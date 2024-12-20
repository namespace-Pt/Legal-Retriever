import os
import json
import mteb
import torch

from loguru import logger
from accelerate import Accelerator
from dataclasses import dataclass, field, asdict
from transformers import HfArgumentParser

from src import (
    ModelArgs,
    get_model_name,
)
from .eval_mteb import CMTEB_TASKS, MTEB_TASKS

os.environ["TOKENIZERS_PARALLELISM"] = "false"



@dataclass
class Args(ModelArgs):
    output_dir: str = field(
        default="data/results/mteb",
        metadata={'help': 'Output directory for results and logs.'}
    )
    tasks: list[str] = field(
        default_factory=lambda: ["cmteb"],
        metadata={'help': 'Task name list. cmteb stands for all CMTEB tasks.'}
    )
    languages: list[str] = field(
        default_factory=lambda: ["cmn", "cmo"],
        metadata={'help': 'Task language list.'}
    )
    overwrite: bool = field(
        default=False,
        metadata={'help': 'Set to True to overwrite the existing results.'}
    )



def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)

    output_folder = os.path.join(args.output_dir, get_model_name(args.model_name_or_path, args.lora))

    # parse cmteb to all subtasks
    tasks = args.tasks.copy()
    if "cmteb" in tasks:
        tasks.remove("cmteb")
        tasks += sum(CMTEB_TASKS.values(), [])
    if "mteb" in tasks:
        tasks.remove("mteb")
        tasks += sum(MTEB_TASKS.values(), [])
    tasks = mteb.get_tasks(languages=args.languages, tasks=tasks)

    encode_kwargs = {"batch_size": args.batch_size}

    for task in tasks:
        evaluator = mteb.MTEB(tasks=[task])
        task_eval_splits = task.metadata_dict.get("eval_splits", [])
        # load data
        logger.info(f"Loading dataset for {task.metadata_dict['name']}")
        task.check_if_dataset_is_superseeded()
        task.load_data(eval_splits=task_eval_splits)

if __name__ == "__main__":
    main()
