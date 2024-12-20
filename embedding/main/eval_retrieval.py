import os
from loguru import logger
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from dataclasses import dataclass, field, asdict
from transformers import HfArgumentParser

from src import (
    ModelArgs,
    Data,
    EvalDataset_Corpus_Iter,
    FileLogger,
    makedirs,
    get_model_and_tokenizer,
    evaluate,
)


@dataclass
class Args(ModelArgs):
    output_dir: str = field(
        default="data/results/retrieval",
        metadata={'help': 'Output directory for results and logs.'}
    )


def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu, kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=100000))])

    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    eval_dataset_corpus_iter = EvalDataset_Corpus_Iter(
        eval_data_files=args.eval_data,
        corpus_files=args.corpus,
        query_template=args.query_template,
        key_template=args.key_template,
        query_max_length=args.query_max_length,
        key_max_length=args.key_max_length,
        tokenizer=tokenizer,
        cache_dir=args.dataset_cache_dir,
        main_process_first=accelerator.main_process_first,
    )

    metrics = evaluate(
        model=model,
        args=args,
        eval_dataset_corpus_iter=eval_dataset_corpus_iter,
        accelerator=accelerator
    )

    if accelerator.process_index == 0:
        log_path = os.path.join(args.output_dir, f"metrics.log")
        file_logger = FileLogger(makedirs(log_path))
        file_logger.log(metrics, Args=asdict(args))
    
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()