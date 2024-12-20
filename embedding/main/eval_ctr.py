import os
import json
import torch
import accelerate
from loguru import logger
from tqdm import tqdm
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from transformers import HfArgumentParser
from sklearn.metrics import roc_auc_score

from src import (
    ModelArgs,
    Data,
    RetrievalDataCollator,
    FileLogger,
    makedirs,
    get_model_and_tokenizer,
)


@dataclass
class Args(ModelArgs):
    output_dir: str = field(
        default="data/results/ctr",
        metadata={'help': 'Output directory for results and logs.'}
    )
    use_real_ctr: bool = field(
        default=False,
        metadata={'help': 'Use real_ctr from data file to compute AUC?'}
    )
    use_oracle_ctr: bool =  field(
        default=False,
        metadata={'help': 'Use oracle_ctr from data file to compute AUC?'}
    )
    # # NOTE: we cannot pack here because we need separate query to compute qauc
    # packing: bool = field(
    #     default=False,
    #     metadata={'help': 'Pack queries and keys to eliminate padding?'}
    # )


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    if args.use_real_ctr or args.use_oracle_ctr:
        args.cpu = True

    accelerator = Accelerator(cpu=args.cpu, kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=100000))])

    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    with accelerator.main_process_first():
        eval_dataset = Data.prepare_train_data(
            data_files=args.eval_data,
            group_size=1,
            train_method="pointwise",
            query_template=args.query_template,
            key_template=args.key_template,
            query_max_length=args.query_max_length,
            key_max_length=args.key_max_length,
            tokenizer=tokenizer,
            cache_dir=args.dataset_cache_dir,
        )

    data_collator = RetrievalDataCollator(
        tokenizer=tokenizer,
        query_max_length=args.query_max_length,
        key_max_length=args.key_max_length,
        packing=args.packing,
    )

    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    dataloader = accelerator.prepare(dataloader)

    all_logits = []
    all_labels = []

    for i, x in enumerate(tqdm(dataloader, desc="Evaluating CTR")):
        # query = tokenizer.batch_decode(x["query"]["input_ids"], skip_special_tokens=True)
        labels = x["pointwise_labels"]
        labels = labels.squeeze(-1).tolist()

        if args.use_real_ctr:
            logits = [ctr[0] for ctr in x["real_ctrs"]]

        elif args.use_oracle_ctr:
            logits = [ctr[0] for ctr in x["oracle_ctrs"]]

        else:
            # (batch_size, d_embed)
            query_embedding = model._encode(x["query"], field="query")
            # (batch_size, d_embed)
            key_embedding = model._encode(x["key"], field="key")
            # (batch_size)
            scores = model._compute_similarity(query_embedding.unsqueeze(1), key_embedding.unsqueeze(1)).flatten()
            logits = torch.sigmoid(scores).tolist()

        if accelerator.num_processes > 1:
            # query = accelerator.gather_for_metrics(query)
            logits = accelerator.gather_for_metrics(logits)
            labels = accelerator.gather_for_metrics(labels)

        # all_queries.extend(query)
        all_logits.extend(logits)
        all_labels.extend(labels)

    if accelerator.process_index == 0:
        # mapping query to auc score
        qauc_dict = defaultdict(lambda: [[],[]])

        assert len(eval_dataset) == len(all_logits), f"Make sure the length of eval_dataset is equal to the length of all_logits"

        for x, logit, label in zip(eval_dataset, all_logits, all_labels):
            query = x["query"]
            qauc_dict[hash(query)][0].append(logit)
            qauc_dict[hash(query)][1].append(label)

        qauc = 0
        qauc_v2 = 0
        valid_query_num = 0
        valid_sample_num = 0

        for query, (logits, labels) in qauc_dict.items():
            if not (1 in labels and 0 in labels):
                continue

            auc = roc_auc_score(labels, logits)
            qauc += auc * len(labels)
            qauc_v2 += auc
            valid_query_num += 1
            valid_sample_num += len(labels)

        qauc /= valid_sample_num
        qauc_v2 /= valid_query_num

        logger.info(f"There are {len(qauc_dict)} queries in total, where {valid_query_num} queries are valid (contain both positive and negative), containing {valid_sample_num} interactions.")

        metrics = {
            "auc": round(roc_auc_score(all_labels, all_logits), 4),
            "qauc": round(qauc, 4),
            "qauc_v2": round(qauc_v2, 4),
        }

        # with open(makedirs(os.path.join(args.output_dir, "results.json")), "w") as f:
        #     json.dump({"labels": all_labels, "predictions": all_logits}, f)

        log_path = os.path.join(args.output_dir, f"metrics.log")
        file_logger = FileLogger(makedirs(log_path))
        file_logger.log(metrics, Args=asdict(args))

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()