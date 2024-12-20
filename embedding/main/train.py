from loguru import logger
from transformers import HfArgumentParser
from peft import LoraConfig, PeftModel, get_peft_model

from src import (
    ModelArgs,
    Data,
    RetrievalDataCollator,
    EvalDataset_Corpus_Iter,
    FileLogger,
    makedirs,
    format_numel_str,
    get_model_and_tokenizer,
)
from src.trainer import RetrievalTrainer, TrainingArgs



def main():
    parser = HfArgumentParser([ModelArgs, TrainingArgs])
    model_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgs
    training_args: TrainingArgs

    model, tokenizer = get_model_and_tokenizer(model_args, device="cpu" if training_args.use_cpu else "cuda")

    with training_args.main_process_first():
        train_dataset = Data.prepare_train_data(
            data_files=model_args.train_data,
            group_size=training_args.train_group_size,
            train_method=training_args.train_method,
            select_pos=training_args.select_pos,
            select_neg=training_args.select_neg,
            seed=training_args.seed,
            query_template=model_args.query_template,
            key_template=model_args.key_template,
            query_max_length=model_args.query_max_length,
            key_max_length=model_args.key_max_length,
            tokenizer=tokenizer,
            cache_dir=model_args.dataset_cache_dir,
            group_by_dataset=training_args.group_by_dataset,
            skip_preprocess=training_args.skip_preprocess,
            # group_by_dataset will modify training_args
            training_args=training_args,
        )

    eval_dataset_corpus_iter = EvalDataset_Corpus_Iter(
        eval_data_files=model_args.eval_data,
        corpus_files=model_args.corpus,
        query_template=model_args.query_template,
        key_template=model_args.key_template,
        query_max_length=model_args.query_max_length,
        key_max_length=model_args.key_max_length,
        tokenizer=tokenizer,
        cache_dir=model_args.dataset_cache_dir,
        main_process_first=training_args.main_process_first
    )

    # NOTE: we may add special tokens to the tokenizer when applying template with {embed} token
    # so we need to update the token embeddings
    if model.model.config.vocab_size < len(tokenizer):
        model.model.resize_token_embeddings(len(tokenizer))

    if training_args.lora_tune:
        # NOTE: enable gradient for inputs but do not update them
        model.model.enable_input_require_grads()

        if isinstance(model.model, PeftModel):
            # the model have lora
            for name, param in model.named_parameters():
                if "lora" in name or "mrl_proj" in name:
                    param.requires_grad_(True)

        else:
            # add a new lora
            config = LoraConfig(
                r=training_args.lora_rank,
                lora_alpha=training_args.lora_alpha,
                target_modules=training_args.lora_targets,
                modules_to_save=training_args.lora_extra_params,
                lora_dropout=training_args.lora_dropout,
                bias="none",
                # NOTE: do not set CAUSAL_LM here because we do not have labels for language modeling
                task_type="EMBEDDING",
            )
            # NOTE: we must apply LoRA on the huggingface transformers
            model.model = get_peft_model(model.model, config)

    if training_args.freeze_embedding:
        assert not training_args.lora_tune, f"Make sure freeze_embedding is activated only in full-parameter tuning!"
        for name, param in model.named_parameters():
            if "embed_tokens" in name:
                param.requires_grad_(False)

    logger.info(f"Trainable Model Params: {format_numel_str(sum(p.numel() for p in model.parameters() if p.requires_grad))}")

    callbacks = []
    if training_args.early_exit_steps is not None:
        from src.trainer import EarlyExitCallBack
        callbacks.append(EarlyExitCallBack())
    if len(callbacks) == 0:
        callbacks = None

    trainer = RetrievalTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        data_collator=RetrievalDataCollator(
            tokenizer=tokenizer,
            query_max_length=model_args.query_max_length,
            key_max_length=model_args.key_max_length,
            group_by_dataset=training_args.group_by_dataset,
            packing=model_args.packing,
        ),
        callbacks=callbacks,
        eval_dataset_corpus_iter=eval_dataset_corpus_iter,
        compute_metrics=None,
        file_logger=FileLogger(makedirs(training_args.log_path))
    )

    # Training
    if train_dataset is not None:
        trainer.train()
    elif len(eval_dataset_corpus_iter) > 0:
        trainer.evaluate()

if __name__ == "__main__":
    main()
