import os
import json
import mteb
import torch
import pandas as pd

from loguru import logger
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from functools import partial
from dataclasses import dataclass, field, asdict
from transformers import HfArgumentParser

from src import (
    ModelArgs,
    Templater,
    get_model_name,
    get_model_and_tokenizer,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


CMTEB_TASKS = {
    "Retrieval": ["T2Retrieval", "MMarcoRetrieval", "DuRetrieval", "CovidRetrieval", "CmedqaRetrieval", "EcomRetrieval", "MedicalRetrieval", "VideoRetrieval"],
    "STS": ["ATEC", "BQ", "LCQMC", "PAWSX", "STSB", "AFQMC", "QBQTC", "STS22.v2"],
    "PairClassification": ["Ocnli", "Cmnli"],
    "Classification": ["TNews", "IFlyTek", "MultilingualSentiment", "JDReview", "OnlineShopping", "Waimai", "AmazonReviewsClassification", "MassiveIntentClassification", "MassiveScenarioClassification"],
    "Reranking": ["T2Reranking", "MMarcoReranking", "CMedQAv1-reranking", "CMedQAv2-reranking"],
    "Clustering": ["CLSClusteringS2S.v2", "CLSClusteringP2P.v2", "ThuNewsClusteringS2S.v2", "ThuNewsClusteringP2P.v2"],
}

MTEB_TASKS = {
    'Retrieval': ['ArguAna', 'ClimateFEVER', 'DBPedia', 'FEVER', 'FiQA2018', 'HotpotQA', 'MSMARCO', 'NFCorpus', 'NQ', 'QuoraRetrieval', 'SCIDOCS', 'SciFact', 'Touche2020', 'TRECCOVID', 'CQADupstackAndroidRetrieval', 'CQADupstackEnglishRetrieval', 'CQADupstackGamingRetrieval', 'CQADupstackGisRetrieval', 'CQADupstackMathematicaRetrieval', 'CQADupstackPhysicsRetrieval', 'CQADupstackProgrammersRetrieval', 'CQADupstackStatsRetrieval', 'CQADupstackTexRetrieval', 'CQADupstackUnixRetrieval', 'CQADupstackWebmastersRetrieval', 'CQADupstackWordpressRetrieval'],
    'Classification': ['AmazonCounterfactualClassification', 'AmazonPolarityClassification', 'AmazonReviewsClassification', 'Banking77Classification', 'EmotionClassification', 'ImdbClassification', 'MassiveIntentClassification', 'MassiveScenarioClassification', 'MTOPDomainClassification', 'MTOPIntentClassification', 'ToxicConversationsClassification', 'TweetSentimentExtractionClassification'],
    'Clustering': ['ArxivClusteringP2P', 'ArxivClusteringS2S', 'BiorxivClusteringP2P', 'BiorxivClusteringS2S', 'MedrxivClusteringP2P', 'MedrxivClusteringS2S', 'RedditClustering', 'RedditClusteringP2P', 'StackExchangeClustering', 'StackExchangeClusteringP2P', 'TwentyNewsgroupsClustering'],
    'PairClassification': ['SprintDuplicateQuestions', 'TwitterSemEval2015', 'TwitterURLCorpus'],
    'Reranking': ['AskUbuntuDupQuestions', 'MindSmallReranking', 'SciDocsRR', 'StackOverflowDupQuestions'],
    'STS': ['BIOSSES', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS17', 'STS22', 'STSBenchmark'],
    'Summarization': ['SummEval']
}


@dataclass
class Args(ModelArgs):
    output_dir: str = field(
        default="data/results/mteb",
        metadata={'help': 'Output directory for results and logs.'}
    )
    result_dir: str = field(
        default=None,
        metadata={'help': 'Subdirectory for saving results.'}
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

    use_efficient: bool = field(
        default=True,
        metadata={'help': 'Whether to use efficient mteb.'}
    )
    use_sentence_transformer: bool = field(
        default=False,
        metadata={'help': 'Whether to use sentence transformer model.'}
    )



def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu, kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=100000))])

    if args.result_dir is not None:
        output_folder = os.path.join(args.output_dir, get_model_name(args.model_name_or_path, args.lora), args.result_dir)
    else:
        output_folder = os.path.join(args.output_dir, get_model_name(args.model_name_or_path, args.lora))

    if args.use_sentence_transformer:
        from sentence_transformers import SentenceTransformer
        if args.dtype == "float16":
            dtype = args.dtype
            if dtype == "float32":
                dtype = torch.float32
            elif dtype == "float16":
                dtype = torch.float16
            elif dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                raise NotImplementedError
        model = SentenceTransformer(args.model_name_or_path, cache_folder=args.model_cache_dir, model_kwargs={"torch_dtype": dtype})
    else:
        model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    # parse cmteb to all subtasks
    tasks = args.tasks.copy()
    if "cmteb" in tasks:
        tasks.remove("cmteb")
        tasks += sum(CMTEB_TASKS.values(), [])
    if "mteb" in tasks:
        tasks.remove("mteb")
        tasks += sum(MTEB_TASKS.values(), [])
    tasks = mteb.get_tasks(languages=args.languages, tasks=tasks)

    if not args.use_sentence_transformer:
        encode_kwargs = {"args": args, "accelerator": accelerator, "batch_size": args.batch_size, "distributed": True}
    else:
        encode_kwargs = {"batch_size": args.batch_size}

    results = []

    templater = Templater(tokenizer)

    for task in tasks:
        logger.info(f"Evaluating {task}...")

        task_name = task.metadata.name
        task_type = task.metadata.type

        if not args.use_sentence_transformer:
            # these two template_fn will be used in Retrieval and Reranking tasks
            encode_kwargs["query_template_fn"] = partial(templater.apply, query_template=args.query_template, dataset=task_name, max_length=args.query_max_length)
            encode_kwargs["key_template_fn"] = partial(templater.apply, key_template=args.key_template, dataset=task_name, max_length=args.key_max_length)
            # this template will be used in other tasks
            encode_kwargs["template_fn"] = partial(templater.apply, query_template=args.query_template, dataset=task_name, max_length=args.query_max_length)

        evaluator = mteb.MTEB(tasks=[task])
        result = evaluator.run(
            model, 
            output_folder=output_folder,
            encode_kwargs=encode_kwargs,
            overwrite_results=args.overwrite,
            cache_dir=args.dataset_cache_dir,
        )

        # NOTE: reset index for every individual retrieval task
        if not args.use_sentence_transformer:
            model.reset()

        accelerator.wait_for_everyone()

        if accelerator.process_index == 0:
            meta = evaluator.create_model_meta(model)
            output_path = evaluator.create_output_folder(meta, output_folder)

            if len(result) == 1:
                scores = result[0].scores
            else:
                # in case the result has existed and is not overwritten
                path = os.path.join(output_path, f"{task_name}.json")
                with open(path, encoding="utf-8") as f:
                    string = f.read()
                    # NOTE: some results file is corrupted
                    if string.endswith("}}"):
                        string = string[:-1]
                    item = json.loads(string)
                    scores = item["scores"]
            main_score = next(iter(scores.values()))[0]["main_score"]
            main_score = round(main_score, 5)
            results.append((task_name, task_type, main_score))
            logger.info(f"({task_type}) {task_name}: {main_score}")

    if accelerator.process_index == 0:
        columns = [res[0] for res in results]
        scores = [res[2] for res in results]

        # average across all datasets
        columns.insert(0, "AVG")
        scores.insert(0, sum(scores) / len(scores))

        # NOTE: add an empty column because the vscode excel viewer may omit the last column
        columns.append("")
        scores.append("")

        df = pd.DataFrame([scores], columns=columns)
        df.to_excel(os.path.join(output_path, "results.xlsx"), float_format="%.5f")

if __name__ == "__main__":
    main()
