import os
import json
import torch
import datasets
import torch.distributed as dist

from loguru import logger
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Iterator, Tuple, Union
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from .data import EvalDataset_Corpus_Iter
from .args import ModelArgs
from .utils import FileLogger
from .modeling_utils import evaluate



class RetrievalTrainer(Trainer):
    def __init__(self, *args, eval_dataset_corpus_iter: Iterator[Tuple[datasets.Dataset, datasets.Dataset]], model_args: ModelArgs, file_logger: FileLogger, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_dataset_corpus_iter = eval_dataset_corpus_iter
        # handle save/load index/encoding/results
        self.model_args = model_args
        self.file_logger = file_logger

    def compute_loss(self, model, inputs, return_outputs=False):
        # key_mask and length is not used in forward
        inputs.pop("key_mask", None)
        inputs.pop("length", None)
        dataset = inputs.pop("dataset", None)
        if dataset:
            self.accelerator.print(f"Dataset name for current batch: {dataset}")
        kwargs = {
            "train_method": self.args.train_method,
            "listwise_temp": self.args.listwise_temp,
            "pointwise_temp": self.args.pointwise_temp,
            "pointwise_weight": self.args.pointwise_weight,
            "listwise_weight": self.args.listwise_weight,
            "contrastive_weight": self.args.contrastive_weight,
            "distill_weight": self.args.distill_weight,
            "distill_student_temp": self.args.distill_student_temp,
            "distill_teacher_temp": self.args.distill_teacher_temp,
            "use_inbatch_neg": self.args.use_inbatch_neg,
            "use_cross_device_neg": self.args.use_cross_device_neg,
            "filter_inbatch_neg": self.args.filter_inbatch_neg,
        }
        inputs.update(kwargs)
        return super().compute_loss(model, inputs, return_outputs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # NOTE: we should remove the prefix of the outmost model
        new_state_dict = type(state_dict)()
        for k, v in state_dict.items():
            new_state_dict[k.replace("model.base_model", "base_model")] = v

        # NOTE: the mrl projection weights and the tokenizer will also be saved
        self.model.save_pretrained(
            # NOTE: do not use state_dict here because there is an extra "model" prefix in the state_dict
            output_dir, state_dict=new_state_dict, safe_serialization=self.args.save_safetensors
        )

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        self.model_args.save(os.path.join(output_dir, "model_args.json"))
        # save training args in json
        with open(os.path.join(output_dir, "training_args.json"), "w", encoding="utf-8") as f:
            json.dump(self.args.to_dict(), f, indent=2)

    @torch.no_grad()
    def evaluate(self, eval_dataset_corpus_iter: Optional[EvalDataset_Corpus_Iter] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if eval_dataset_corpus_iter is None:
            if self.eval_dataset_corpus_iter is not None:
                eval_dataset_corpus_iter = self.eval_dataset_corpus_iter
            else:
                logger.warning(f"No eval_dataset_corpus_iter provided, skipping evaluation.")
                return
        if len(eval_dataset_corpus_iter) == 0:
            logger.warning(f"No eval_dataset_corpus_iter provided, skipping evaluation.")
            return

        args = self.args
        model_args = self.model_args
        accelerator = self.accelerator
        # tie output_dir
        model_args.output_dir = args.output_dir

        metrics = evaluate(
            model=self.model,
            args=model_args,
            eval_dataset_corpus_iter=eval_dataset_corpus_iter,
            accelerator=accelerator
        )

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        # log to file
        if accelerator.process_index == 0:
            self.file_logger.log(
                metrics=metrics,
                Model_Args=asdict(model_args),
                Training_Args=asdict(args),
                Global_Steps=self.state.global_step
            )

        return metrics


class EarlyExitCallBack(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if args.early_exit_steps is not None and state.global_step > args.early_exit_steps:
            control.should_training_stop = True


@dataclass
class TrainingArgs(TrainingArguments):
    output_dir: str = field(
        default='data/outputs/',
        metadata={'help': 'The output directory where the model predictions and checkpoints will be written.'},
    )

    skip_preprocess: bool = field(
        default=False,
        metadata={'help': 'Skip preprocessing when loading train_data?'}
    )

    group_by_dataset: Optional[str] = field(
        default=None,
        metadata={'help': 'Whether and how to use samples from the same task in each batch (across devices). {random, epoch-random}'}
    )
    train_method: str = field(
        default="listwise",
        metadata={'help': 'How to train the model? {pointwise, listwise, both}'}
    )
    use_inbatch_neg: bool = field(
        default=True,
        metadata={'help': 'Use in-batch negatives?'}
    )
    use_cross_device_neg: Optional[int] = field(
        default=-1,
        metadata={'help': 'Gather negatives from how many processes when distributed training? -1 means all processes'}
    )
    filter_inbatch_neg: bool = field(
        default=True,
        metadata={'help': 'Filter out in-batch positives?'}
    )
    listwise_temp: float = field(
        default=0.02,
        metadata={'help': 'Temperature used for cosine dense metric in listwise training.'}
    )
    pointwise_temp: float = field(
        default=0.2,
        metadata={'help': 'Temperature used for cosine dense metric in pointwise training.'}
    )
    distill_teacher_temp: float = field(
        default=1.,
        metadata={'help': 'Temperature used for teacher scores in distillation.'}
    )
    distill_student_temp: float = field(
        default=1.,
        metadata={'help': 'Temperature used for student scores in distillation.'}
    )
    contrastive_weight: float = field(
        default=0.2,
        metadata={'help': 'Weight for contrastive loss in distillation.'}
    )
    distill_weight: float = field(
        default=0,
        metadata={'help': 'Weight for distillation loss.'}
    )
    freeze_embedding: bool = field(
        default=False,
        metadata={'help': 'Freeze the embedding parameters when full-parameter tuning?'}
    )
    pointwise_weight: float = field(
        default=1.,
        metadata={'help': 'Weight for pointwise loss in hybrid training.'}
    )
    listwise_weight: float = field(
        default=1.,
        metadata={'help': 'Weight for listwise loss in hybrid training.'}
    )

    train_group_size: int = field(
        default=8,
        metadata={'help': 'How many keys in a batch?'}
    )
    select_pos: str = field(
        default="first",
        metadata={'help': 'How to select the positive key from a set of positives?'}
    )
    select_neg: str = field(
        default="first",
        metadata={'help': 'How to select the negative keys from a set of negatives?'}
    )

    lora_tune: bool = field(
        default=False,
        metadata={"help": "Use LoRA fine-tuning?"},
    )
    lora_rank: int = field(
        default=64,
        metadata={'help': 'LoRA rank.'}
    )
    lora_alpha: int = field(
        default=32,
        metadata={'help': 'LoRA scaling factor.'}
    )
    lora_dropout: float = field(
        default=0.,
        metadata={'help': 'LoRA dropout p.'}
    )
    lora_targets: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Module name patterns to add LoRA."},
    )
    lora_extra_params: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Extra trainable parameters except LoRA weights, if low rank training."},
    )

    remove_unused_columns: bool = field(
        default=False,
        metadata={'help': 'Remove columns in the dataset that are not registered in the forward function?'}
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={'help': 'Find unusuable parameters?'}
    )
    ddp_timeout: int = field(
        default=36000,
        metadata={'help': 'Timeout for DDP.'}
    )
    report_to: str = field(
        default="none",
        metadata={'help': 'Log results by external tools?'}
    )
    early_exit_steps: int = field(
        default=None,
        metadata={'help': 'Early exit after this number of steps.'}
    )
    log_path: str = field(
        default="data/outputs/metrics.log",
        metadata={'help': 'Log file path.'}
    )

    # NOTE: newer version of transformers forbid modifying the configs after initilization, we bypass this setting
    # because group_by_dataset will modify the per_device_train_batch_size setting
    def __setattr__(self, name, value):
        super(TrainingArguments, self).__setattr__(name, value)
