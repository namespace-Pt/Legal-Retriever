import os
import torch
import seaborn as sns
from tqdm import tqdm
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from src import (
    ModelArgs,
    Data,
    RetrievalDataCollator,
    makedirs,
    get_model_and_tokenizer,
)


@dataclass
class Args(ModelArgs):
    output_dir: str = field(
        default="data/results/docsim",
        metadata={'help': 'Output directory for results and logs.'}
    )


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

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

    all_scores = []

    for x in tqdm(dataloader, desc="Evaluating Doc Sim"):
        # (batch_size, d_embed)
        key_embedding = model._encode(x["key"], field="key")
        # (batch_size, batch_size)
        scores = model._compute_similarity(key_embedding, key_embedding).tolist()

        if accelerator.num_processes > 1:
            scores = accelerator.gather_for_metrics(scores)

        all_scores.extend(scores)

    if accelerator.process_index == 0:
        result_path = os.path.join(args.output_dir, args.model_short_name, "results.pt")
        path = makedirs(result_path)
        all_scores = torch.tensor(all_scores)
        torch.save(all_scores, path)

        fig_path = os.path.join(args.output_dir, args.model_short_name, "visualization.png")
        ax = sns.histplot(all_scores.flatten().numpy()[:100000])
        ax.figure.savefig(fig_path, format='png', bbox_inches='tight')


if __name__ == "__main__":
    main()