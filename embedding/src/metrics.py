import os
import json
import torch
import random
import inspect
import datasets
import numpy as np

from loguru import logger
from fuzzywuzzy import fuzz
from contextlib import nullcontext
from typing import List, Optional, Union
from tqdm import tqdm

from .utils import makedirs, split_file_dir_name_ext, dataset_no_transform



class Metrics:
    """Class for computing metrics and some post-processings."""
    def compute_metrics(metric_names: List[str], *args, **kwargs):
        all_implemented_fns = [x[0] for x in inspect.getmembers(Metrics, predicate=inspect.isfunction) if not x[0].startswith("_")]
        metrics = {}
        for metric_name in metric_names:
            # call corresponding method
            if metric_name in all_implemented_fns:
                metric_fn = getattr(Metrics, metric_name)
                metric = metric_fn(*args, **kwargs)
                # NOTE: some metric_fn are only used for post-processing and saving results, which return None by default
                if metric is not None:
                    metrics.update(metric)
            else:
                raise NotImplementedError(f"Metric {metric_name} not implemented!")
        return metrics

    def _get_save_path(eval_data_file: str, output_dir: Optional[str] = None, field: str = "result", save_name: str = None):
        """
        if output_dir is None:
            -> {eval_data_dir}/{eval_data_name}.{field}.{save_name}.jsonl
        else:
            -> {output_dir}/{eval_data_name}.{field}.{save_name}.jsonl
        """
        eval_data_dir, eval_data_name, eval_data_ext = split_file_dir_name_ext(eval_data_file)
        if output_dir is None:
            output_dir = eval_data_dir
        # NOTE: we may input the gz file, but it will be automatically uncompressed after loading, 
        # so we remove the extension name before .gz
        if eval_data_ext == ".gz":
            eval_data_name = ".".join(eval_data_name.split(".")[:-1])
        fields = [eval_data_name, field]
        if save_name is not None:
            fields.append(save_name)
        save_path = os.path.join(output_dir, ".".join(fields) + ".jsonl")
        makedirs(save_path)
        return save_path

    def _save_retrieval_result(result_path: str, retrieval_indices: Union[List[List[int]], np.ndarray], retrieval_scores: Optional[Union[List[List[float]], np.ndarray]] = None):
        with open(result_path, "w", encoding="utf-8") as f:
            for i in range(len(retrieval_indices)):
                res = {
                    "indices": retrieval_indices[i],
                }
                if retrieval_scores is not None:
                    res["scores"] = retrieval_scores[i]
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

    def _load_retrieval_result(result_path: str):
        logger.info(f"Loading retrieval results from {result_path}...")
        retrieval_indices = []
        retrieval_scores = []
        with open(result_path, encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                retrieval_indices.append(x["indices"])
                if "scores" in x:
                    retrieval_scores.append(x["scores"])
        if len(retrieval_scores) == 0 and len(retrieval_indices) > 0:
            retrieval_scores = None
        return retrieval_indices, retrieval_scores

    def _clean_retrieval_result(retrieval_indices: Union[List[List[int]], np.ndarray], retrieval_scores: Optional[Union[List[List[float]], np.ndarray]] = None):
        """Remove -1 (means no valid documents to retrieve) from indices and the corresponding documents."""
        cleaned_retrieval_indices = []
        if retrieval_scores is None:
            cleaned_retrieval_scores = None
        else:
            cleaned_retrieval_scores = []

        for i in range(len(retrieval_indices)):
            indices = retrieval_indices[i]
            if retrieval_scores is not None:
                scores = retrieval_scores[i]

            if -1 in indices:
                if isinstance(indices, np.ndarray):
                    valid_pos = indices > -1
                    indices = indices[valid_pos].tolist()
                    if retrieval_scores is not None:
                        scores = scores[valid_pos].tolist()
                else:
                    valid_pos = [i for i, x in enumerate(indices) if x > -1]
                    indices = [indices[i] for i in valid_pos]
                    if retrieval_scores is not None:
                        scores = [scores[i] for i in valid_pos]
            
            cleaned_retrieval_indices.append(indices)
            if retrieval_scores is not None:
                cleaned_retrieval_scores.append(scores)

        return retrieval_indices, retrieval_scores

    def _get_labels(eval_dataset: datasets.Dataset, corpus: datasets.Dataset):
        """Get label_indices and label_rels from the eval_dataset."""
        label_indices = []
        label_rels = []

        # NOTE: first, we examine if there is a field named 'pos_raw', which stores the raw text of the positive document
        # if there is, then we use it to map the text to corpus offsets
        # otherwise, we assume there is a field named 'pos' in the un-transformed eval_dataset, which acts the same as 'pos_raw'

        test_sample = eval_dataset[0]
        with dataset_no_transform(eval_dataset):
            test_sample_without_transform = eval_dataset[0]
        
        if "pos_indices" in test_sample_without_transform:
            keep_transform = False
        elif "pos_raw" in test_sample:
            keep_transform = True
        elif "pos" in test_sample_without_transform:
            keep_transform = False
        else:
            raise ValueError(f"Expect 'pos_indices' in the original test dataset, or 'pos_raw' in the transformed test dataset, or 'pos' in the original test dataset.")
        
        # NOTE: determine whether we need to reverse the dataset transform_fn

        with dataset_no_transform(eval_dataset) if not keep_transform else nullcontext(), dataset_no_transform(corpus):
            if "pos_indices" in test_sample_without_transform:
                for x in eval_dataset:
                    indices = x.get("pos_indices", None)
                    if indices is not None:
                        label_indices.append(indices)
                        rels = x.get("pos_rels", [1 for _ in indices])
                        label_rels.append(rels)
            else:
                doc2idx = {}
                for i, c in enumerate(tqdm(corpus, desc="Collecting Text from Corpus")):
                    doc2idx[c["text"]] = i
                for x in tqdm(eval_dataset, desc="Collecting Positive from Eval Data"):
                    if "pos_raw" in x:
                        indices = [doc2idx[p] for p in x["pos_raw"]]
                    else:
                        indices = [doc2idx[p] for p in x["pos"]]
                    label_indices.append(indices)
                    rels = [1 for _ in indices]
                    label_rels.append(rels)
        return label_indices, label_rels

    def mrr(retrieval_indices: Union[List[List[int]], np.ndarray], label_indices: List[List[int]], retrieval_scores: Optional[Union[List[List[float]], np.ndarray]] = None, label_rels: Optional[List[List[float]]] = None, cutoffs: List[int] = [1, 10, 100], **kwargs):
        mrrs = np.zeros(len(cutoffs))
        counts = 0

        retrieval_indices, retrieval_scores = Metrics._clean_retrieval_result(retrieval_indices, retrieval_scores)

        assert len(retrieval_indices) == len(label_indices), f"Make sure the number of predictions ({len(retrieval_indices)}) equals the number of labels ({len(label_indices)})!"

        for i in range(len(retrieval_indices)):
            preds = retrieval_indices[i]
            targets = label_indices[i]

            # filter out irrelevant targets
            if label_rels is not None:
                targets = [target for j, target in enumerate(targets) if label_rels[i][j] > 0]

            if len(targets) == 0:
                continue
            
            targets = set(targets)

            jump = False
            counts += 1

            for i, pred in enumerate(preds, 1):
                if pred in targets:
                    for k, cutoff in enumerate(cutoffs):
                        if i <= cutoff:
                            mrrs[k] += 1 / i
                    jump = True
                if jump:
                    break

        mrrs /= counts

        metric = {}
        for i, cutoff in enumerate(cutoffs):
            mrr = mrrs[i]
            metric[f"mrr@{cutoff}"] = round(mrr, 4)
        return metric


    def recall(retrieval_indices: Union[List[List[int]], np.ndarray], label_indices: List[List[int]], retrieval_scores: Optional[Union[List[List[float]], np.ndarray]] = None, label_rels: Optional[List[List[float]]] = None, cutoffs: List[int] = [1, 10, 100], **kwargs):
        recalls = np.zeros(len(cutoffs))
        counts = 0

        retrieval_indices, retrieval_scores = Metrics._clean_retrieval_result(retrieval_indices, retrieval_scores)

        assert len(retrieval_indices) == len(label_indices), f"Make sure the number of predictions ({len(retrieval_indices)}) equals the number of labels ({len(label_indices)})!"

        for i in range(len(retrieval_indices)):
            preds = retrieval_indices[i]
            targets = label_indices[i]

            # filter out irrelevant targets
            if label_rels is not None:
                targets = [target for j, target in enumerate(targets) if label_rels[i][j] > 0]

            if len(targets) == 0:
                continue
            
            counts += 1

            for k, cutoff in enumerate(cutoffs):
                recall = np.intersect1d(targets, preds[:cutoff])
                recalls[k] += len(recall) / len(targets)

        recalls /= counts

        metric = {}
        for i, cutoff in enumerate(cutoffs):
            recall = recalls[i]
            metric[f"recall@{cutoff}"] = round(recall, 4)
        return metric
    
    def ndcg(retrieval_indices: Union[List[List[int]], np.ndarray], label_indices: List[List[int]], retrieval_scores: Optional[Union[List[List[float]], np.ndarray]] = None, label_rels: Optional[List[List[float]]] = None, cutoffs: List[int] = [1, 10, 100], **kwargs):
        ndcgs = np.zeros(len(cutoffs))
        counts = 0

        retrieval_indices, retrieval_scores = Metrics._clean_retrieval_result(retrieval_indices, retrieval_scores)

        assert len(retrieval_indices) == len(label_indices), f"Make sure the number of predictions ({len(retrieval_indices)}) equals the number of labels ({len(label_indices)})!"

        for i in range(len(retrieval_indices)):
            preds = retrieval_indices[i]
            targets = label_indices[i]
            # filter out irrelevant targets
            if label_rels is not None:
                assert len(targets) == len(label_rels[i]), f"Make sure the number of targets {len(targets)} equals the number of scores {len(label_rels[i])}"
                target_scores = label_rels[i]
            else:
                # all are equally relevant by default
                target_scores = [1 for _ in targets]
            target2score = {target: score for target, score in zip(targets, target_scores)}

            if len(targets) == 0:
                continue

            targets = set(targets)

            dcg = np.zeros(len(cutoffs))
            idcg = np.zeros(len(cutoffs))
            counts += 1

            for i, pred in enumerate(preds, 1):
                if pred in targets:
                    for k, cutoff in enumerate(cutoffs):
                        if i <= cutoff:
                            # get the relevance score of the pred
                            dcg[k] += (2 ** target2score[pred] - 1) / np.log2(i + 1)

            # descendingly sort positives to acquire the ideal ranking
            ideal_ranking = sorted(target_scores, reverse=True)
            for j, y in enumerate(ideal_ranking, 1):
                for k, cutoff in enumerate(cutoffs):
                    if j <= cutoff:
                        idcg[k] += (2 ** y - 1) / np.log2(j + 1)

            ndcgs += dcg / idcg

        ndcgs /= counts

        metric = {}
        for i, cutoff in enumerate(cutoffs):
            ndcg = ndcgs[i]
            metric[f"ndcg@{cutoff}"] = round(ndcg, 4)
        return metric

    def collate_key(
        eval_data_file: str,
        eval_dataset: datasets.Dataset, 
        corpus: datasets.Dataset, 
        retrieval_indices: Union[List[List[int]], np.ndarray], 
        retrieval_scores: Optional[Union[List[List[float]], 
        np.ndarray]] = None, 
        save_name: Optional[str] = None, 
        output_dir: Optional[str] = None, 
        save_to_output: bool = False,
        **kwargs
    ):
        """
        Collate retrieval results for evaluation. 
        Append a 'keys' column in the eval_data where each key is a piece of retrieved text;
        Delete 'pos' and 'neg' column.
        If output_dir is None, save at {eval_data}.keys.{save_name}.jsonl
        Else, save at {output_dir}.keys.{save_name}.jsonl
        """
        assert len(retrieval_indices) == len(eval_dataset), f"Make sure the number of predictions ({len(retrieval_indices)}) match the number of evaluation data ({len(eval_dataset)})."

        retrieval_indices, retrieval_scores = Metrics._clean_retrieval_result(retrieval_indices, retrieval_scores)

        if save_to_output:
            save_path = Metrics._get_save_path(eval_data_file, output_dir, field="key", save_name=save_name)
        else:
            save_path = Metrics._get_save_path(eval_data_file, None, field="key", save_name=save_name)

        logger.info(f"Saving retrieved keys to {save_path}...")

        with open(save_path, "w", encoding="utf-8") as f, dataset_no_transform(eval_dataset), dataset_no_transform(corpus):
            for i, x in enumerate(tqdm(eval_dataset, desc="Collating Retrieved Keys")):
                item = {
                    "query": x["query"]
                }
                preds = retrieval_indices[i]
                item["key"] = corpus[preds]["text"]
                item["key_indices"] = preds
                if retrieval_scores is not None:
                    item["key_scores"] = retrieval_scores[i]
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def collate_neg(
        eval_data_file: str,
        eval_dataset: datasets.Dataset, 
        corpus: datasets.Dataset, 
        retrieval_indices: Union[List[List[int]], np.ndarray], 
        retrieval_scores: Optional[Union[List[List[float]], 
        np.ndarray]] = None, 
        save_name: Optional[str] = None, 
        output_dir: Optional[str] = None, 
        save_to_output: bool = False,
        neg_num: int = 20, 
        neg_keep_original: Optional[int] = 0,
        neg_filter_answer: bool = False,
        neg_filter_fuzz: Optional[int] = None,
        **kwargs
    ):
        """
        Collate hard negatives for training. 
        Append 'pos' and 'neg' columns in the eval_data where each element is a piece of retrieved text;
        Save at {output_dir}.neg.{save_name}.jsonl
        """
        random.seed(42)

        assert len(retrieval_indices) == len(eval_dataset), f"Make sure the number of predictions ({len(retrieval_indices)}) match the number of evaluation data ({len(eval_dataset)})."

        # retrieval_indices, retrieval_scores = Metrics._clean_retrieval_result(retrieval_indices, retrieval_scores)

        if save_to_output:
            save_path = Metrics._get_save_path(eval_data_file, output_dir, field="neg", save_name=save_name)
        else:
            save_path = Metrics._get_save_path(eval_data_file, None, field="neg", save_name=save_name)

        logger.info(f"Saving {neg_num} negatives to {save_path}...")
    
        with open(save_path, "w", encoding="utf-8") as f, dataset_no_transform(eval_dataset), dataset_no_transform(corpus):
            dataset = DummyCollateNegDataset(eval_dataset=eval_dataset, corpus=corpus, retrieval_indices=retrieval_indices, retrieval_scores=retrieval_scores)
            # NOTE: use dataloader to speed up!
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=512, 
                collate_fn=dummy_collate_fn, 
                shuffle=False,
                drop_last=False,
                pin_memory=False,
                num_workers=32,
            )
            for x in tqdm(dataloader, desc="Collating Hard Negatives"):
                for item in x:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")



class DummyCollateNegDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        eval_dataset: datasets.Dataset, 
        corpus: datasets.Dataset, 
        retrieval_indices: List[List[int]], 
        retrieval_scores: List[List[int]],
        neg_num: int = 20, 
        neg_keep_original: Optional[int] = 0,
        neg_filter_answer: bool = False,
        neg_filter_fuzz: Optional[int] = None,
    ):
        self.dataset = eval_dataset
        self.corpus = corpus
        self.retrieval_indices = retrieval_indices
        self.retrieval_scores = retrieval_scores
        self.neg_num = neg_num
        self.neg_keep_original = neg_keep_original
        self.neg_filter_answer = neg_filter_answer
        self.neg_filter_fuzz = neg_filter_fuzz

        assert len(eval_dataset) == len(retrieval_indices), f"Make sure the dataset length {len(eval_dataset)} equals the retrieval results length {len(retrieval_indices)}!"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]

        pos = x.get("pos", [])
        # NOTE: the pos_scores and neg_scores will be deleted by default because we cannot guarantee all pos docs and neg docs are returned by the retriever
        item = {
            "query": x["query"],
            "pos": pos
        }
        if "pos_indices" in x:
            pos_indices = set(x["pos_indices"])
            item["pos_indices"] = x["pos_indices"]
        else:
            pos_indices = set()
        if "pos_rels" in x:
            item["pos_rels"] = x["pos_rels"]

        indices = self.retrieval_indices[idx]

        # first filter out positive documents
        indices = [index for index in indices if index not in pos_indices]

        # get text
        preds = self.corpus[indices]["text"]

        pos_lower = set([p.lower() for p in pos] + [x["query"].lower()])

        # NOTE: we may want to keep several original negatives, in that case, we should assure new negatives are different against any kept ones
        original_neg = x.get("neg", [])
        if len(original_neg) and self.neg_keep_original:
            original_neg_num = min(self.neg_keep_original, len(original_neg))
            original_neg_sampled_indices = random.sample(range(len(original_neg)), original_neg_num)
            original_neg = [original_neg[j] for j in original_neg_sampled_indices]
        else:
            original_neg_num = 0
        original_neg_lower = set([n.lower() for n in original_neg])

        valid_indices = []
        deduplicate_neg = set()

        for j, pred in enumerate(preds):
            pred = pred.lower()
            # 1. remove duplicated negatives
            if pred in deduplicate_neg:
                continue
            # 2. remove negative that is the same as any positive
            if pred in pos_lower:
                continue
            # 3. remove negative that contains the answer when neg_filter_answer
            if self.neg_filter_answer and "answers" in x:
                if any(answer in pred for answer in x["answers"]):
                    continue
            # 4. remove negative that are too similar to any positive (meansure by fuzzy score)
            if self.neg_filter_fuzz:
                if any(fuzz.ratio(pred, p) > self.neg_filter_fuzz for p in pos_lower):
                    continue
            # 5. remove negative that is the same as any kept original negatives
            if pred in original_neg_lower:
                continue
            valid_indices.append(j)
            deduplicate_neg.add(pred)

        # only slice out the first neg_num docs
        valid_indices = valid_indices[:self.neg_num]

        valid_indices = valid_indices[:(self.neg_num - original_neg_num)]
        new_neg = [preds[j] for j in valid_indices]

        # mix
        item["neg"] = new_neg + original_neg
        return item


def dummy_collate_fn(batch):
    return batch
