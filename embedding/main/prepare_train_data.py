from loguru import logger
from datetime import timedelta
from transformers import HfArgumentParser
from accelerate import Accelerator, InitProcessGroupKwargs

from src import (
    ModelArgs,
    Data,
    get_model_and_tokenizer,
)
from src.trainer import TrainingArgs



def main():
    parser = HfArgumentParser([ModelArgs, TrainingArgs])
    model_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgs
    training_args: TrainingArgs

    accelerator = Accelerator(cpu=True, kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=100000))])

    model, tokenizer = get_model_and_tokenizer(model_args, device="cuda")

    if accelerator.process_index == 0:
        logger.info("Tokenizing training data...")

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
            skip_preprocess=False,
            # group_by_dataset will modify training_args
            training_args=training_args,
        )

        train_dataset.save_to_disk(training_args.output_dir)

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
