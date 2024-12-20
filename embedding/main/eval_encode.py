from loguru import logger
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from src import (
    ModelArgs,
    Data,
    get_model_and_tokenizer,
    split_file_dir_name_ext
)


@dataclass
class Args(ModelArgs):
    output_dir: str = field(
        default="data/results/encode",
        metadata={'help': 'Output directory for results and logs.'}
    )
    save_encode: bool = field(
        default=True,
        metadata={'help': 'Save embeddings?'}
    )
    save_name: str = field(
        default="default",
        metadata={'help': 'Name name of the embedding file.'}
    )


def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu, kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=100000))])

    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    for corpus_path in args.corpus:
        with accelerator.main_process_first():
            corpus = Data.prepare_corpus(
                corpus_path,
                key_template=args.key_template,
                key_max_length=args.key_max_length,
                tokenizer=tokenizer,
                cache_dir=args.dataset_cache_dir,
            )

        _, save_name, _ = split_file_dir_name_ext(corpus_path)
        model.index(
            corpus,

            batch_size=args.batch_size,
            output_dir=args.output_dir,
            save_encode=args.save_encode,
            load_encode=args.load_encode,
            save_name=save_name,
            accelerator=accelerator,
            skip_indexing=True,
        )


if __name__ == "__main__":
    main()