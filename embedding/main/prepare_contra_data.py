import os
import sys

import faiss
from torch.utils.data import DataLoader
import datasets
import json
import random
from glob import glob
from typing import Optional
from dataclasses import dataclass

from transformers import HfArgumentParser
from tqdm import tqdm
from src import ModelArgs, makedirs


@dataclass
class Args(ModelArgs):
    base_dir: str = "/mnt/bn/search-douyin-rank-yg/all_data_from_lf/text_embedding_data/query_doc_info_sample_order_1028_v4"
    output_dir: Optional[str] = None
    doc_as_query_portion: float = 0
    write_to_json: bool = False
    start_file_idx: Optional[int] = None
    end_file_idx: Optional[int] = None


def collate_fn(batch_elem):
    assert len(batch_elem) == 1
    return batch_elem[0]


def collate_query_pos_neg_from_impression(impression, doc_as_query_portion: float = 0):
    # query
    query = impression[0]["query"]
    assert all([i["query"] == query for i in impression])
    search_id = impression[0]["search_id"]

    # click
    strong_pos = []
    # play_time_max > 10s
    pos = []
    # play_time_max < 3s
    neg = []
    # play_time_max < 3s and position < click position
    strong_neg = []
    # position of click
    strong_pos_position = []
    
    neg_candidates = []
    for i, x in enumerate(impression):
        text = x["doc_info"]
        doc_id = x["doc_id"]

        # NOTE: the pos may be absorbed in strong pos
        if x["search_result_click_cnt"] > 0:
            strong_pos.append((text, doc_id))
            strong_pos_position.append(x["position"])
        elif x["play_time_max"] > 10000:
            pos.append((text, doc_id))
        else:
            neg_candidates.append(x)

    if len(strong_pos) < 1 or len(pos) < 1:
    # if len(strong_pos) < 1:
        return None

    min_click_position = min(strong_pos_position)
    for i, x in enumerate(neg_candidates):
        text = x["doc_info"]
        doc_id = x["doc_id"]
        position = x["position"]

        if x["play_time_max"] < 3000: 
            if position < min_click_position:
                strong_neg.append((text, doc_id))
            else:
                neg.append((text, doc_id))

    if doc_as_query_portion > 0 and random.uniform(0, 1) <= doc_as_query_portion:
        query, _ = pos.pop(0)
        result_pos = strong_pos + pos
        result_neg = strong_neg + neg

    else:
        result_pos = strong_pos + pos
        result_neg = strong_neg + neg

    if len(result_pos) < 1 or len(result_neg) < 1:
        return None

    return {
        "search_id": search_id,
        "query": query,
        "pos": [p[0] for p in result_pos],
        "neg": [n[0] for n in result_neg],
        "pos_ids": [p[1] for p in result_pos],
        "neg_ids": [n[1] for n in result_neg]
    }


if __name__ == "__main__":
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    base_dir = args.base_dir
    if args.output_dir is None:
        output_dir = base_dir
    else:
        output_dir = args.output_dir

    doc_as_query_portion = args.doc_as_query_portion

    all_files = glob(f"{base_dir}/part-*.parquet")

    if args.start_file_idx is None:
        start_file_idx = 0
    else:
        start_file_idx = args.start_file_idx
    
    if args.end_file_idx is None:
        end_file_idx = len(all_files)
    else:
        end_file_idx = args.end_file_idx

    data_files = sum([glob(f"{base_dir}/part-{i:05d}-*.parquet") for i in range(start_file_idx, end_file_idx)], [])
    prefix = str() + f"{round(len(data_files) / len(all_files), 2)}s{start_file_idx}e{end_file_idx}"


    print(f"start_idx: {start_file_idx}, end_idx: {end_file_idx}, len: {len(data_files)}")
    dataset = datasets.load_dataset("parquet", data_files=data_files, split="train", streaming=True)

    if doc_as_query_portion > 0:
        output_dir = os.path.join(output_dir, f"{prefix}_{1 - doc_as_query_portion}query_{doc_as_query_portion}doc_pos_neg")
    else:
        output_dir = os.path.join(output_dir, f"{prefix}_query_pos_neg")

    print(f"Will save dataset to {output_dir}! Load with datasets.load_from_disk!")

    random.seed(0)

    path = makedirs(os.path.join(output_dir, "train.jsonl"))

    if args.write_to_json:
        json_file = open(path, "w", encoding="utf-8")

    def dataset_generator():
        impression = []
        prev_search_id = None

        # NOTE: cannot use_num_workers>1 on dataset with n_shards == 1
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn, prefetch_factor=32)
        for i, x in enumerate(tqdm(dataloader)):
            # if i > 100:
            #     break

            search_id = x["search_id"]

            text = ""
            doc = json.loads(x["doc_info"])
            fields = [
                ("title", doc.get('title', '').strip()),
                ("username", doc.get('username', '').strip()),
                ("music", doc.get('music', '').strip()),
                ("poi", doc.get('poi', '').strip()),
                ("challenge", doc.get('challenge', '').strip()),
                ("ocr", doc.get('ocr', '').strip()),
                ("asr", doc.get('asr', '').strip())
            ]
            for field_name, field_value in fields:
                text += f"<{field_name}>{field_value}\n\n"
            x["doc_info"] = text

            # this means a new impression begins
            if search_id != prev_search_id and prev_search_id is not None:
                # assert query != prev_query
                res = collate_query_pos_neg_from_impression(impression, args.doc_as_query_portion)
                impression.clear()

                if res is not None:
                    yield res

                    # NOTE: double assurance
                    if args.write_to_json:
                        json_file.write(json.dumps(res, ensure_ascii=False) + "\n")

            if x["doc_id"] != "t":
                impression.append(x)

            prev_search_id = search_id

    dataset = datasets.Dataset.from_generator(dataset_generator)
    dataset.save_to_disk(output_dir)

    if args.write_to_json:
        json_file.close()
