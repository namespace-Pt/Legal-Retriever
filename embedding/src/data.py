import os
import re
import math
import json
import torch
import random
import datasets
import contextlib
import numpy as np
import ubelt as ub

from loguru import logger
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union, List, Callable, Dict, Any, Mapping
from dataclasses import dataclass
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer

from .template import Templater
from .utils import get_max_length_in_nested_lists, pad_nested_lists, split_file_dir_name_ext


DATASET_NUM_PROC=os.cpu_count()


def parse_text_from_douyin_doc_info(doc_info):
    doc_text = ""
    doc = json.loads(doc_info)
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
        doc_text += f"<{field_name}>{field_value}\n\n"
    return doc_text


class Data:
    def _process_listwise_train_data(
        data,
        group_size: int = 8, 
        select_pos: str = "first", 
        select_neg: str = "random", 
        query_template_fn: Callable = lambda x: x, 
        key_template_fn: Callable = lambda x: x, 
        tokenizer: PreTrainedTokenizer = None,
    ):
        outputs = {"query": [], "key": [], "teacher_scores": []}

        placeholder_none = [None] * len(data['query'])

        for query, pos, neg, pos_scores, neg_scores in zip(data["query"], data["pos"], data["neg"], data.get("pos_scores", placeholder_none), data.get("neg_scores", placeholder_none)):
            # select positive sample
            if select_pos == "random":
                pos_idx = random.choice(range(len(pos)))
            elif select_pos == "first":
                pos_idx = 0
            else:
                raise ValueError(f"select_pos {select_pos} not supported!")
            pos_item = pos[pos_idx]
            if pos_scores is not None:
                assert len(pos_scores) == len(pos)
                pos_score = pos_scores[pos_idx]

            # repeat negative samples if there are not enough
            if len(neg) == 0 and group_size > 1:
                continue

            # NOTE: filter samples that do not have enough negatives
            if len(neg) < group_size - 1:
                # continue
                num = math.ceil((group_size - 1) / len(neg))
                neg = neg * num
                if neg_scores is not None:
                    neg_scores = neg_scores * num
            
            # select negative samples
            if select_neg == "random":
                neg_indices = random.sample(range(len(neg)), group_size - 1)
            elif select_neg == "first":
                neg_indices = list(range(len(neg)))[:group_size - 1]
            else:
                raise ValueError(f"select_neg {select_neg} not supported!")
            neg_items = [neg[i] for i in neg_indices]
            if neg_scores is not None:
                assert len(neg_scores) == len(neg)
                neg_scores = [neg_scores[i] for i in neg_indices]

            # combine pos and neg to form key
            key = [pos_item] + neg_items
            if pos_scores is not None and neg_scores is not None:
                teacher_scores = [pos_score] + neg_scores
            else:
                teacher_scores = None

            # format query/pos/neg with template
            query = query_template_fn(query)
            key = key_template_fn(key)

            outputs["query"].append(query)
            outputs["key"].append(key)
            outputs["teacher_scores"].append(teacher_scores)

            if "length" not in outputs:
                outputs["length"]  = []
            outputs["length"].append(max(len(x) for x in tokenizer(key)['input_ids']))

        return outputs
    
    def _process_pointwise_train_data(
        data,
        query_template_fn: Callable = lambda x: x, 
        key_template_fn: Callable = lambda x: x, 
        tokenizer: PreTrainedTokenizer = None,
    ):
        # NOTE: the pointwise data is directly loaded from the source parquet files

        outputs = {"query": [], "key": [], "length": [], "pointwise_labels": [], "real_ctrs": [], "oracle_ctrs": [], "search_id": [], "key_ids": []}

        for query, doc_info, search_result_click_cnt, real_ctr, oracle_ctr, search_id, doc_id in zip(data["query"], data["doc_info"], data["search_result_click_cnt"], data["real_clk_ctr"], data["video_oracle_ctr"], data["search_id"], data["doc_id"]):
            if doc_info is None:
                continue

            real_ctrs = [float(real_ctr)]
            oracle_ctrs = [float(oracle_ctr)]

            text = parse_text_from_douyin_doc_info(doc_info)

            query = query_template_fn(query)
            key = [key_template_fn(text)]
            label = [int(search_result_click_cnt > 0)]
            key_id = [doc_id]
            # NOTE: pos_raw will be used in computing retrieval metrics
            # pos_raw = [text]

            outputs["search_id"].append(search_id)
            outputs["query"].append(query)
            outputs["key"].append(key)
            outputs["key_ids"].append(key_id)
            outputs["length"].append(len(tokenizer(key[0])['input_ids']))
            outputs["pointwise_labels"].append(label)
            outputs["real_ctrs"].append(real_ctrs)
            outputs["oracle_ctrs"].append(oracle_ctrs)
            # outputs["pos_raw"].append(pos_raw)

        return outputs

    def _process_eval_data(data, query_template_fn: Callable = lambda x: x, tokenizer: PreTrainedTokenizer = None):
        placeholder_none = [None] * len(data['query'])
        # TODO: rerank evaluation
        outputs = {"query": [], "pos_raw": []}
        for query, pos, doc_info in zip(data["query"], data.get("pos", placeholder_none), data.get("doc_info", placeholder_none)):                
            if pos is not None:
                outputs["pos_raw"].append(pos)
            elif doc_info is not None:
                # douyin
                outputs["pos_raw"].append([parse_text_from_douyin_doc_info(doc_info)])
            else:
                outputs["pos_raw"].append([])
            query = query_template_fn(query)
            outputs["query"].append(query)
        return outputs

    def _process_corpus(data, key_template_fn: Callable = lambda x: x):
        # TODO: chain set_transform so that the key can be generated from multiple fields
        if "text" in data:
            text = key_template_fn(data["text"])
            outputs = {"text": text}
        else:
            raise ValueError(f"Expect 'text' field!")
        if "gid" in data:
            outputs["gid"] = data["gid"]
        return outputs

    def _collect_texts_from_train_file(data):
        outputs = {"text": [], "hash": []}
        for query, pos, neg in zip(data["query"], data["pos"], data.get("neg", [None] * len(data['query']))):
            texts = []
            if pos is not None: 
                texts += pos
            if neg is not None:
                texts += neg
            outputs["text"].extend(texts)
            # NOTE: we cannot use python std hash function because it is not deterministic among processes
            outputs["hash"].extend([ub.hash_data(t, hasher="md5") for t in texts])
        return outputs
    
    def _collect_texts_from_douyin_data(data):
        outputs = {"text": [], "hash": []}
        for query, doc_info in zip(data["query"], data["doc_info"]):
            if doc_info is None:
                continue
            text = parse_text_from_douyin_doc_info(doc_info)
            outputs["text"].append(text)
            # NOTE: we cannot use python std hash function because it is not deterministic among processes
            outputs["hash"].append(ub.hash_data(text, hasher="md5"))
        return outputs

    def prepare_train_data(
        data_files: Optional[Union[List[str], str]] = None,
        train_method: str = "listwise",
        group_size: int = 8,
        select_pos: str = "first",
        select_neg: str = "random",
        seed: int = 42,
        query_template: str = "no",
        key_template: str = "no",
        query_max_length: int = 512,
        key_max_length: int = 512,
        tokenizer: PreTrainedTokenizer = None,
        cache_dir: Optional[str] = None,
        group_by_dataset: Optional[str] = None,
        keep_in_memory: Optional[bool] = None,
        skip_preprocess: bool = False,
        training_args: Optional[TrainingArguments] = None,
    ):
        if data_files is None:
            return None

        if isinstance(data_files, list):
            pass
        elif isinstance(data_files, str):
            data_files = [data_files]
        else:
            raise ValueError(f"Invalid training data {data_files}!")
        logger.info(f"Loading training data from {data_files}...")

        data_2_num_sample = {}
        for data_file in data_files:
            match = re.search("\[(\d*)\]", data_file)
            if match:
                max_sample_num = int(match.group(1))
                data_file = re.sub("\[(\d*)\]", "", data_file)
            else:
                max_sample_num = None
            data_2_num_sample[data_file] = max_sample_num

        random.seed(seed)

        train_datasets = []
        offset = 0
        dataset_indices_range = {}
        dataset_dup = {}

        templater = Templater(tokenizer=tokenizer)

        assert train_method in ["listwise", "pointwise", "both"]

        for data_file, max_sample_num in data_2_num_sample.items():

            # NOTE: we always use the name of the parent folder as the dataset name regardless of the dataset is a single file or a folder
            dataset_name = split_file_dir_name_ext(data_file)[0].name

            if os.path.isdir(data_file) and os.path.exists(os.path.join(data_file, "dataset_info.json")):
                logger.info(f"Loading {dataset_name} dataset from disk dataset ({data_file})...")
                # the dataset may be save_to_disk in advance
                dataset = datasets.load_from_disk(data_file)
            elif os.path.isdir(data_file):
                logger.info(f"Loading {dataset_name} dataset from parquet ({data_file}/*.parquet)...")
                # FIXME: very ugly, we need to specify the dataset format
                # FIXME: how to organize multiple files in one dataset and multiple files for different datasets?
                dataset = datasets.load_dataset('parquet', data_files=f"{data_file}/*.parquet", split='train', cache_dir=cache_dir)
            else:
                logger.info(f"Loading {dataset_name} dataset from json ({data_file})...")
                dataset = datasets.load_dataset('json', data_files=data_file, split='train', cache_dir=cache_dir)

            # NOTE: get template fn
            query_template_fn = partial(templater.apply, query_template=query_template, dataset=dataset_name, max_length=query_max_length)
            key_template_fn = partial(templater.apply, key_template=key_template, dataset=dataset_name, max_length=key_max_length)

            # NOTE: NLI dataset has only 1 negative

            if train_method in ["listwise", "both"]:
                process_fn = partial(
                    Data._process_listwise_train_data,
                    group_size=group_size,
                    select_pos=select_pos,
                    select_neg=select_neg,
                    query_template_fn=query_template_fn,
                    key_template_fn=key_template_fn,
                    tokenizer=tokenizer,
                )
            else:
                process_fn = partial(
                    Data._process_pointwise_train_data,
                    query_template_fn=query_template_fn,
                    key_template_fn=key_template_fn,
                    tokenizer=tokenizer,
                )

            if not skip_preprocess:
                # map to filter
                dataset = dataset.map(process_fn, batched=True, num_proc=DATASET_NUM_PROC, remove_columns=dataset.column_names, batch_size=32, keep_in_memory=keep_in_memory)

            if max_sample_num is not None and len(dataset) > max_sample_num:
                dataset = dataset.train_test_split(max_sample_num, seed=seed)["test"]

            # the start and end index of the current dataset
            if dataset_name in dataset_indices_range:
                # NOTE: we allow duplicated dataset to balance the portion of different datasets
                dataset_dup[dataset_name] += 1
                dataset_indices_range[f"{dataset_name}_{dataset_dup[dataset_name]}"] = (offset, offset + len(dataset))
            else:
                dataset_indices_range[dataset_name] = (offset, offset + len(dataset))
                dataset_dup[dataset_name] = 0

            offset += len(dataset)

            train_datasets.append(dataset)

        dataset = datasets.concatenate_datasets(train_datasets)

        # all samples in the same batch come from the same dataset
        if group_by_dataset:
            assert training_args.dataloader_num_workers == 0, f"Make sure dataloader num_workers is 0 when using group_by_dataset!"

            dataset = GroupByDatasetTrainDataset(
                dataset, 
                dataset_indices_range, 
                batch_size=training_args.per_device_train_batch_size, 
                seed=seed, 
                group_by_dataset=group_by_dataset, 
                num_processes=training_args.world_size,
                process_index=training_args.process_index,
            )
            # NOTE: the dataset already yields one entire batch with batch_size, so we need to use train_batch_size=1
            training_args.per_device_train_batch_size = 1

        return dataset

    def prepare_eval_data(data_file: Optional[Union[List[str], str]] = None, query_template: str = "no", query_max_length: int = 512, tokenizer: PreTrainedTokenizer = None, cache_dir: Optional[str] = None):
        # listwise evaluation data
        if data_file is None:
            return None

        assert isinstance(data_file, str)
        logger.info(f"Loading evaluation data from {data_file}...")

        match = re.search("\[(\d*)\]", data_file)
        if match:
            max_sample_num = int(match.group(1))
            data_file = re.sub("\[(\d*)\]", "", data_file)
        else:
            max_sample_num = None

        # NOTE: we always use the name of the parent folder as the dataset name regardless of the dataset is a single file or a folder
        dataset_name = split_file_dir_name_ext(data_file)[0].name

        do_filter = False
        if os.path.isdir(data_file) and os.path.exists(os.path.join(data_file, "dataset_info.json")):
            logger.info(f"Loading {dataset_name} dataset from disk dataset ({data_file})...")
            # the dataset may be save_to_disk in advance
            dataset = datasets.load_from_disk(data_file)
        elif os.path.isdir(data_file):
            # douyin dataset
            logger.info(f"Loading {dataset_name} dataset from parquet ({data_file}/*.parquet)...")
            # FIXME: very ugly, we need to specify the dataset format
            # FIXME: how to organize multiple files in one dataset and multiple files for different datasets?
            dataset = datasets.load_dataset('parquet', data_files=f"{data_file}/*.parquet", split='train', cache_dir=cache_dir)
            # filter out negative samples and empty documents
            do_filter = True
        else:
            logger.info(f"Loading {dataset_name} dataset from json ({data_file})...")
            dataset = datasets.load_dataset('json', data_files=data_file, split='train', cache_dir=cache_dir)

        if max_sample_num is not None and len(dataset) > max_sample_num:
            dataset = dataset.train_test_split(max_sample_num, shuffle=False)["test"]

        if do_filter:
            dataset = dataset.filter(lambda data: [click > 0 and doc is not None for click, doc in zip(data["search_result_click_cnt"], data["doc_info"])], num_proc=DATASET_NUM_PROC, batched=True)

        # NOTE: get template fn
        templater = Templater(tokenizer=tokenizer)
        query_template_fn = partial(templater.apply, query_template=query_template, dataset=dataset_name, max_length=query_max_length)

        process_fn = partial(
            Data._process_eval_data,
            query_template_fn=query_template_fn,
            tokenizer=tokenizer,
        )
        dataset.set_transform(process_fn)
        return dataset

    def prepare_corpus(data_file, key_template: str = "no", key_max_length: int = 512, tokenizer: PreTrainedTokenizer = None, cache_dir=None):
        if data_file is None:
            return None
        
        assert isinstance(data_file, str)

        match = re.search("\[(\d*)\]", data_file)
        if match:
            max_sample_num = int(match.group(1))
            data_file = re.sub("\[(\d*)\]", "", data_file)
        else:
            max_sample_num = None

        if os.path.isdir(data_file) and os.path.exists(os.path.join(data_file, "dataset_info.json")):
            logger.info(f"Loading corpus from disk dataset ({data_file})...")
            # the dataset may be save_to_disk in advance
            dataset = datasets.load_from_disk(data_file)
        elif os.path.isdir(data_file):
            # FIXME: very ugly
            logger.info(f"Loading corpus from parquet ({data_file}/*.parquet)...")
            dataset = datasets.load_dataset('parquet', data_files=f"{data_file}/*.parquet", split='train', cache_dir=cache_dir)
        elif data_file.endswith("parquet"):
            logger.info(f"Loading corpus from parquet ({data_file}...")
            dataset = datasets.load_dataset('parquet', data_files=data_file, split='train', cache_dir=cache_dir)
        else:
            logger.info(f"Loading corpus from json ({data_file})...")
            dataset = datasets.load_dataset('json', data_files=data_file, split='train', cache_dir=cache_dir)

        # NOTE: we always use the name of the parent folder as the dataset name regardless of the dataset is a single file or a folder
        dataset_name = split_file_dir_name_ext(data_file)[0].name

        if max_sample_num is not None and len(dataset) > max_sample_num:
            dataset = dataset.train_test_split(max_sample_num, shuffle=False)["test"]

        # NOTE: get template fn
        templater = Templater(tokenizer=tokenizer)
        key_template_fn = partial(templater.apply, key_template=key_template, dataset=dataset_name, max_length=key_max_length)

        if "query" in dataset.column_names:
            if "pos" in dataset.column_names:
                # in case we are using a train file as corpus (STS datasets)
                dataset = dataset.map(Data._collect_texts_from_train_file, batched=True, num_proc=DATASET_NUM_PROC, remove_columns=dataset.column_names, batch_size=32)
                logger.info(f"Deduplicating dataset of size {len(dataset)}...")
            elif "doc_info" in dataset.column_names:
                # in case we are using douyin parquet data as corpus
                dataset = dataset.map(Data._collect_texts_from_douyin_data, batched=True, num_proc=DATASET_NUM_PROC, remove_columns=dataset.column_names, batch_size=32)
            else:
                raise NotImplementedError(f"Reading corpus from train/test data requires either 'pos' column or 'doc_info' column in dataset!")

            # NOTE: deduplicate
            _ , unique_indices = np.unique(dataset["hash"], return_index=True, axis=0)
            dataset = dataset.select(unique_indices.tolist())
            dataset = dataset.remove_columns(["hash"])
            # print(f"rank {torch.distributed.get_rank()} unique_indices {unique_indices[:10]} dataset {dataset[0]}")

        if "text" in dataset.column_names:
            pass
        elif "content" in dataset.column_names:
            dataset = dataset.rename_column("content", "text")
        elif "doc_info" in dataset.column_names:
            # douyin dataset
            def text_format_fn(x):
                outputs = {"text": []}
                for doc_info in x["doc_info"]:
                    outputs["text"].append(parse_text_from_douyin_doc_info(doc_info))
                outputs["gid"] = x["gid"]
                return outputs
            dataset = dataset.map(text_format_fn, batched=True, num_proc=DATASET_NUM_PROC, remove_columns=dataset.column_names, batch_size=100, desc="Formatting Text")
        else:
            raise NotImplementedError("Corpus format not supported!")

        process_fn = partial(
            Data._process_corpus,
            key_template_fn=key_template_fn,
        )
        dataset.set_transform(process_fn)

        return dataset


class GroupByDatasetTrainDataset(torch.utils.data.Dataset):
    """Dataset to yield a batch of data at one time. All samples in the same batch comes from the same dataset.
    
    Args:
        group_by_dataset: 
            random: 
                At every step, randomly sample a dataset, then randomly sample a batch from that dataset. The dataset may not be enumerated given one-epoch training.
            epoch: 
                Before the epoch begin, organize all data from the same dataset into units of size (batch_size * num_processes), then shuffle these units. At every step, slice one unit. The units will be re-generated for every epoch.
            epoch-static:
                Before the epoch begin, organize all data from the same dataset into units of size (batch_size * num_processes), then shuffle these units. At every step, slice one unit. The units will not be re-generated for every epoch, instead, they will only be shuffled.
            epoch-random:
                At every step, sample a dataset based on the remaining length of unused data, then slice out (batch_size * num_processes) data from the dataset. All datasets will be enumerated given one-epoch training.
    """
    def __init__(self, dataset, dataset_indices_range, batch_size, seed, group_by_dataset, process_index=0, num_processes=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_by_dataset = group_by_dataset
        self.process_index = process_index
        self.num_processes = num_processes

        self.dataset_indices_range = dataset_indices_range

        self.deterministic_generator = np.random.default_rng(seed)
        # different devices must sample different data batch
        self.nondeterministic_generator = np.random.default_rng(seed + process_index)

        # shuffle the indices
        if "random" in self.group_by_dataset:
            self.sample_range = [np.arange(*x) for x in self.dataset_indices_range.values()]
            for x in self.sample_range:
                # NOTE: we must make sure every processes use the same shuffling order
                self.deterministic_generator.shuffle(x)
    
    def create_epoch(self):
        """
        Group data from the same dataset into (batch_size * num_processes) units, then shuffle all units.
        """
        epoch = []
        for k, x in self.dataset_indices_range.items():
            dataset_range = np.arange(*x)
            # NOTE: we must make sure every processes use the same shuffling order
            self.deterministic_generator.shuffle(dataset_range)
            num_batches, remainer = divmod(len(dataset_range), self.batch_size * self.num_processes)
            # Truncate
            if remainer != 0:
                dataset_range = dataset_range[:num_batches * self.batch_size * self.num_processes]

            batches = dataset_range.reshape(num_batches, self.batch_size * self.num_processes).tolist()
            for i in range(len(batches)):
                batches[i] = (k, batches[i])
            epoch.extend(batches)
        # shuffle among datasets, also make sure different processes share the same shuffling results
        self.deterministic_generator.shuffle(epoch)
        self.epoch = epoch
        self.step = 0
        self.steps_per_epoch = len(epoch)

    def __getitem__(self, idx):        
        if self.group_by_dataset == "random":
            sample_prob = [len(x) / len(self.dataset) for x in self.sample_range]

            dataset_name = self.deterministic_generator.choice(range(len(self.sample_range)), size=1, p=sample_prob)[0]
            sample_range = self.sample_range[dataset_name]

            batch_indices = self.nondeterministic_generator.choice(sample_range, size=self.batch_size, replace=False)
            batch_data = self.dataset[batch_indices.tolist()]

        elif self.group_by_dataset == "epoch":
            if not hasattr(self, "epoch") or self.step > self.steps_per_epoch - 1:
                self.create_epoch()

            dataset_name, batch_indices = self.epoch[self.step]
            batch_indices = batch_indices[self.process_index * self.batch_size: (self.process_index + 1) * self.batch_size]
            batch_data = self.dataset[batch_indices]
            self.step += 1
        
        elif self.group_by_dataset == "epoch-static":
            if not hasattr(self, "epoch"):
                # the data within each batch is static once created
                self.create_epoch()
            
            if self.step > self.steps_per_epoch - 1:
                self.deterministic_generator.shuffle(self.epoch)
                self.step = 0

            dataset_name, batch_indices = self.epoch[self.step]
            batch_indices = batch_indices[self.process_index * self.batch_size: (self.process_index + 1) * self.batch_size]
            batch_data = self.dataset[batch_indices]
            self.step += 1
        
        elif self.group_by_dataset == "epoch-random":
            sample_scope = [len(x) for x in self.sample_range]
            sample_prob = [x / sum(sample_scope) for x in sample_scope]

            dataset_name = self.deterministic_generator.choice(range(len(self.sample_range)), size=1, p=sample_prob)[0]
            sample_range = self.sample_range[dataset_name]

            # sequential sample (the indices are already shuffled)
            batch_indices = sample_range[self.process_index * self.batch_size: (self.process_index + 1) * self.batch_size]
            batch_data = self.dataset[batch_indices.tolist()]
            # update indices
            remaining_indices = sample_range[self.num_processes * self.batch_size:]
            if len(remaining_indices) < self.batch_size * self.num_processes:
                remaining_indices = np.array([])
            self.sample_range[dataset_name] = remaining_indices
            # restore all indices if they are all sampled
            if all(len(x) == 0 for x in self.sample_range):
                self.sample_range = [np.arange(*x) for x in self.dataset_indices_range.values()]
                for x in self.sample_range:
                    self.deterministic_generator.shuffle(x)
        else:
            raise NotImplementedError(f"Organize method {self.group_by_dataset} is not implemented!")

        batch_data["dataset"] = dataset_name

        return batch_data
    
    def __len__(self):
        return len(self.dataset) // self.batch_size


@dataclass
class RetrievalDataCollator:
    """
    Data collator for retrieval.
    """
    tokenizer: PreTrainedTokenizer = None
    query_template_fn: Optional[Callable] = None
    key_template_fn: Optional[Callable] = None
    query_max_length: Optional[int] = None
    key_max_length: Optional[int] = None
    group_by_dataset: Optional[str] = None
    return_cross: Optional[bool] = None
    packing: Optional[bool] = None

    def __call__(self, batch_elem):
        first_elem = batch_elem[0]
        return_batch = {}
        
        for k, v in first_elem.items():
            if self.group_by_dataset:
                # here the data have already been grouped
                batch_value = batch_elem[0][k]
            else:
                batch_value = [elem[k] for elem in batch_elem]
            
            # collate training/evaluating
            if k == "query":
                if self.query_template_fn is not None:
                    batch_value = self.query_template_fn(batch_value)

                query = batch_value
                # NOTE: we do not need the individual query and key when requiring cross data
                if self.return_cross:
                    continue

                if self.packing:
                    # NOTE: we will use position_ids to detect the boundary of each query
                    input_ids = self.tokenizer(
                        batch_value,
                        truncation=True,
                        max_length=self.query_max_length,
                    ).input_ids
                    position_ids = [list(range(len(x))) for x in input_ids]
                    # 1, sum_of_all_query_length
                    batch_value = {
                        "input_ids": torch.tensor([sum(input_ids, [])]), 
                        "position_ids": torch.tensor([sum(position_ids, [])])
                    }

                else:
                    batch_value = self.tokenizer(
                        batch_value,
                        padding=True,
                        truncation=True,
                        max_length=self.query_max_length,
                        return_tensors="pt",
                    )

            elif k == "key":
                # in case the keys are of different sizes for different queries when reranking
                max_length = get_max_length_in_nested_lists(batch_value)
                batch_value, key_mask = pad_nested_lists(batch_value, max_length, "<|padding_content|>", "right")
                # NOTE: one query corresponds to multiple keys, which will be concatenated
                batch_value = sum(batch_value, [])

                if self.key_template_fn is not None:
                    batch_value = self.key_template_fn(batch_value)

                key = batch_value
                # key_mask assigns 1 to valid keys and 0 to padded keys
                return_batch["key_mask"] = torch.tensor(key_mask)
                # NOTE: we do not need the individual query and key when requiring cross data
                if self.return_cross:
                    continue

                if self.packing:
                    # NOTE: we will use position_ids to detect the boundary of each query
                    input_ids = self.tokenizer(
                        batch_value,
                        truncation=True,
                        max_length=self.key_max_length,
                    ).input_ids
                    position_ids = [list(range(len(x))) for x in input_ids]
                    # 1, sum_of_all_query_length
                    batch_value = {
                        "input_ids": torch.tensor([sum(input_ids, [])]), 
                        "position_ids": torch.tensor([sum(position_ids, [])])
                    }

                else:
                    batch_value = self.tokenizer(
                        batch_value,
                        padding=True,
                        truncation=True,
                        max_length=self.key_max_length,
                        return_tensors="pt",
                    )

            elif k == "key_index":
                max_length = get_max_length_in_nested_lists(batch_value)
                batch_value, _ = pad_nested_lists(batch_value, max_length, -1, "right")

            elif k == "text":
                if self.key_template_fn is not None:
                    batch_value = self.key_template_fn(batch_value)

                if self.packing:
                    # NOTE: we will use position_ids to detect the boundary of each query
                    input_ids = self.tokenizer(
                        batch_value,
                        truncation=True,
                        max_length=self.key_max_length,
                    ).input_ids
                    position_ids = [list(range(len(x))) for x in input_ids]
                    # 1, sum_of_all_query_length
                    batch_value = {
                        "input_ids": torch.tensor([sum(input_ids, [])]), 
                        "position_ids": torch.tensor([sum(position_ids, [])])
                    }
                
                else:
                    # collate corpus
                    batch_value = self.tokenizer(
                        batch_value,
                        padding=True,
                        truncation=True,
                        max_length=self.key_max_length,
                        return_tensors="pt",
                    )
            
            elif k == "pointwise_labels":
                batch_value = torch.tensor(batch_value)

            elif any(v is None for v in batch_value):
                # in case that some data have teacher_scores but others do not
                batch_value = None

            else:
                pass

            return_batch[k] = batch_value

        if self.return_cross:
            query_num = len(query)
            key_num = len(key)
            assert key_num % query_num == 0
            group_size = key_num // query_num
            new_query = []
            for i in range(key_num):
                new_query.append(query[i // group_size])

            if self.packing:
                logger.warning(f"Packing is not supported for query-doc cross data by now.")

            # TODO: support packing
            return_batch["cross"] = self.tokenizer(
                new_query, key, 
                padding=True, 
                truncation=True,
                max_length=self.key_max_length + self.query_max_length,
                return_tensors="pt"
            )
            return_batch["batch_size"] = len(query)

        return return_batch


@dataclass
class EvalDataSuite:
    eval_dataset: Optional[datasets.Dataset] = None
    eval_data_file: Optional[str] = None
    corpus: Optional[datasets.Dataset] = None
    corpus_file: Optional[str] = None


class EvalDataset_Corpus_Iter:
    """An iterator to yield one eval_dataset and corresponding corpus at one time (to save memory)."""
    def __init__(self, eval_data_files, corpus_files, query_template: str = "no", key_template: str = "no", query_max_length: int = 512, key_max_length: int = 512,  tokenizer: PreTrainedTokenizer = None, cache_dir: Optional[str] = None, main_process_first: Callable = contextlib.nullcontext):
        if eval_data_files is None:
            eval_data_files = []
        if corpus_files is None:
            corpus_files = []
        if not isinstance(eval_data_files, list):
            eval_data_files = [eval_data_files]
        if not isinstance(corpus_files, list):
            corpus_files = [corpus_files]
        assert len(eval_data_files) == len(corpus_files)

        self.eval_data_files = eval_data_files
        self.corpus_files = corpus_files
        self.query_template = query_template
        self.key_template = key_template
        self.query_max_length = query_max_length
        self.key_max_length = key_max_length
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.main_process_first = main_process_first
    
    def __len__(self):
        return len(self.eval_data_files)

    def __iter__(self):
        for eval_data_file, corpus_file in zip(self.eval_data_files, self.corpus_files):
            with self.main_process_first():
                eval_dataset = Data.prepare_eval_data(eval_data_file, query_template=self.query_template, query_max_length=self.query_max_length, tokenizer=self.tokenizer, cache_dir=self.cache_dir)
                corpus = Data.prepare_corpus(corpus_file, key_template=self.key_template, key_max_length=self.key_max_length, tokenizer=self.tokenizer, cache_dir=self.cache_dir)
            # NOTE: we may use [xxx] to get a subset of eval_data, so make sure this pattern is eliminated
            eval_data_file = re.sub("\[(\d*)\]", "", eval_data_file)
            yield EvalDataSuite(
                eval_dataset=eval_dataset,
                corpus=corpus,
                eval_data_file=eval_data_file,
                corpus_file=corpus_file
            )
