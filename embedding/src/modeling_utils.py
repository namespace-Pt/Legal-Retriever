import os
import torch
import datasets
import accelerate
import numpy as np

from loguru import logger
from tqdm import tqdm
from contextlib import nullcontext
from typing import Optional, List, Union, Mapping

from .data import EvalDataset_Corpus_Iter
from .args import ModelArgs
from .metrics import Metrics
from .utils import makedirs, dataset_no_transform


def optional_grad_ctx(with_grad: bool = False):
    if with_grad:
        return nullcontext()
    else:
        return torch.no_grad()


def move_to_device(data, device: Union[str, int, torch.device]):
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: move_to_device(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(move_to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        return data.to(**kwargs)
    else:
        return data


def save_to_memmap(path: str, shape: tuple, array: np.ndarray, indices: np.ndarray, batch_size: int = 100000, accelerator: Optional[accelerate.Accelerator] = None):
    """
    Save numpy array to memmap file.

    indices: the mapping from line index in array to the line index in the final memmap
    """
    if accelerator is None or accelerator.process_index == 0:
        if os.path.exists(path):
            os.remove(path)
        else:
            makedirs(path)
        memmap = np.memmap(
            path,
            shape=shape,
            mode="w+",
            dtype=array.dtype
        )
        del memmap
    
    if accelerator is not None:
        accelerator.wait_for_everyone()

    logger.info(f"Saving array at {path}...")
    memmap = np.memmap(
        path,
        shape=shape,
        mode="r+",
        dtype=array.dtype
    )
    assert len(indices) == array.shape[0], f"Array size {array.shape[0]} and indices size {len(indices)} not matched!"
    if len(indices) > batch_size:
        for i in tqdm(range(0, len(indices), batch_size), leave=False, ncols=100):
            batch_indices = indices[i: i + batch_size]
            memmap[batch_indices] = array[i: i + batch_size]
    else:
        memmap[indices] = array

    if accelerator is not None:
        accelerator.wait_for_everyone()


def get_inputs_length(inputs: Union[str, List[str], Mapping[str, torch.Tensor]]):
    if isinstance(inputs, str):
        return 1    
    elif isinstance(inputs, list) and isinstance(inputs[0], str):
        return len(inputs)
    elif isinstance(inputs, Mapping):
        length = len(next(iter(inputs.values())))
        assert all(len(x) == length for x in inputs.values()), f"Make sure all values have the same length in inputs."
        return length
    else:
        raise ValueError(f"Expected inputs of type str, list[str], or dict, got {type(inputs)}!")


def distribute_inputs(inputs: Union[str, List[str], Mapping[str, torch.Tensor]], num_processes: int = 1, process_index: int = 0):
    """
    Distribute the inputs to all processes, used for multi-gpu inference.

    Returns:
        local_inputs
        is_distributed: True if the inputs have been distributed, False otherwise.
        (local_start_idx, local_end_idx): the indices range of the local inputs
    """
    if isinstance(inputs, str):
        local_start_idx, local_end_idx = -1, -1
        local_inputs = inputs
        is_distributed = False
    elif isinstance(inputs, list) and isinstance(inputs[0], str):
        if num_processes > 1 and len(inputs) >= num_processes:
            num_instances_per_process = len(inputs) / num_processes
            local_start_idx = round(num_instances_per_process * process_index)
            local_end_idx = round(num_instances_per_process * (process_index + 1))
            local_inputs = inputs[local_start_idx: local_end_idx]
            is_distributed = True
        else:
            local_start_idx, local_end_idx = -1, -1
            local_inputs = inputs
            is_distributed = False
    elif isinstance(inputs, Mapping):
        length = len(next(iter(inputs.values())))
        assert all(len(x) == length for x in inputs.values()), f"Make sure all values have the same length in inputs."
        if num_processes > 1 and len(inputs) >= num_processes:
            num_instances_per_process = length / num_processes
            local_start_idx = round(num_instances_per_process * process_index)
            local_end_idx = round(num_instances_per_process * (process_index + 1))
            local_inputs = type(inputs)({k: v[local_start_idx: local_end_idx] for k, v in inputs.items()})
            is_distributed = True
        else:
            local_start_idx, local_end_idx = -1, -1
            local_inputs = inputs
            is_distributed = False
    else:
        raise ValueError(f"Expected inputs of type str, list[str], or dict, got {type(inputs)}!")
    return local_inputs, is_distributed, (local_start_idx, local_end_idx)


def batchify_inputs(inputs, batch_size: int = 512):
    """
    Yield one batch of inputs at a time, along with its starting/ending index.
    """
    if isinstance(inputs, str):
        yield [inputs], (0, 1)
    elif isinstance(inputs, list) and isinstance(inputs[0], str):
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i: i + batch_size]
            yield batch_inputs, (i, min(i + batch_size, len(inputs)))
    elif isinstance(inputs, Mapping):
        length = len(next(iter(inputs.values())))
        assert all(len(x) == length for x in inputs.values()), f"Make sure all values have the same length in inputs."
        for i in range(0, length, batch_size):
            batch_inputs = type(inputs)({k: v[i: i + batch_size] for k, v in inputs.items()})
            yield batch_inputs, (i, min(i + batch_size, length))
    else:
        raise ValueError(f"Expected inputs of type str, list[str], or dict, got {type(inputs)}!")


@torch.no_grad()
def evaluate(model, args: ModelArgs, eval_dataset_corpus_iter: EvalDataset_Corpus_Iter, accelerator: Optional[accelerate.Accelerator] = None):
    """Evaluate the model according to eval_method."""
    
    if hasattr(model, "eval"):
        model.eval()

    torch.cuda.empty_cache()

    # # NOTE: very important to reset group_by_dataset
    # group_by_dataset = data_collator.group_by_dataset
    # data_collator.group_by_dataset = None

    metrics = {}

    # NOTE: the location of results should be related to encode_method
    output_dir = os.path.join(args.output_dir, args.encode_method)

    for eval_data in eval_dataset_corpus_iter:
        eval_data_file = eval_data.eval_data_file
        eval_dataset = eval_data.eval_dataset
        corpus_file = eval_data.corpus_file
        corpus = eval_data.corpus

        result_path = Metrics._get_save_path(eval_data_file, output_dir, field="result", save_name=args.save_name)

        if args.load_result:
            retrieval_indices, retrieval_scores = Metrics._load_retrieval_result(result_path)

        else:
            if args.eval_method == "retrieval":
                # index corpus
                model.index(
                    corpus,

                    batch_size=args.batch_size,
                    output_dir=output_dir,

                    # for dense retrieval
                    index_factory=args.faiss_index_factory,
                    save_index=args.save_index,
                    load_index=args.load_index,
                    save_encode=args.save_encode,
                    load_encode=args.load_encode,
                    save_name=args.save_name,
                    accelerator=accelerator,

                    # for bm25 retrieval
                    threads=args.anserini_threads,
                    language=args.anserini_language,
                    storeDocvectors=args.anserini_store_docvec,
                    load_collection=args.load_collection,
                )

                # num_samples, hits
                retrieval_indices, retrieval_scores = model.search(
                    eval_dataset, 
                    hits=args.hits,

                    batch_size=args.batch_size,
                    show_progress=True,

                    # for dense retrieval
                    accelerator=accelerator,

                    # for bm25 retrieval
                    threads=args.anserini_threads,
                    parallelism=args.anserini_parallel
                )
            else:
                raise NotImplementedError(f"Eval method {args.eval_method} not implemented!")
            
            if accelerator.process_index == 0 and args.save_result:
                Metrics._save_retrieval_result(result_path, retrieval_indices, retrieval_scores)

        if accelerator.process_index == 0:
            label_indices, label_rels = Metrics._get_labels(eval_dataset, corpus)
            dataset_metrics = Metrics.compute_metrics(
                metric_names=args.metrics,
                eval_data_file=eval_data_file,
                eval_dataset=eval_dataset,
                corpus=corpus, 
                retrieval_indices=retrieval_indices,
                retrieval_scores=retrieval_scores,
                label_indices=label_indices,
                label_rels=label_rels,
                save_name=args.save_name,
                save_to_output=args.save_to_output,
                neg_num=args.neg_num,
                neg_filter_answer=args.neg_filter_answer,
                neg_filter_fuzz=args.neg_filter_fuzz,
                neg_keep_original=args.neg_keep_original,
            )

            metrics[eval_data_file] = dataset_metrics

    # NOTE: broadcast metrics across devices
    if accelerator.num_processes > 1:
        # wrap in list for broadcast
        metrics = [metrics]
        accelerate.utils.broadcast_object_list(metrics, from_process=0)
        metrics = metrics[0]

    accelerator.wait_for_everyone()
    # data_collator.group_by_dataset = group_by_dataset

    torch.cuda.empty_cache()

    return metrics


class DummyCorpusDataset(torch.utils.data.Dataset):
    """Simple dataset to wrap the raw strings with a 'text' field in dict."""
    def __init__(self, corpus):
        self.corpus = corpus
    def __len__(self):
        return len(self.corpus)
    def __getitem__(self, idx):
        return {"text": self.corpus[idx]}


class ShardSampler:
    """
    The sampler returns a shard of the input corpus.
    """
    def __init__(self, corpus: Union[datasets.Dataset, torch.utils.data.Dataset], num_replicas: int, rank: int) -> None:
        """
        Args:
            num_replicas: number of splits
            rank: the current process id

        Attributes:
            start: the starting index
            end: the ending index
        """
        super().__init__()
        corpus_length = len(corpus)
        len_per_worker = corpus_length / num_replicas
        # force to set rank==0 because when world_size==1 the local_rank is -1 by default
        if num_replicas == 1:
            rank = 0
        self.start = round(len_per_worker * rank)
        self.end = round(len_per_worker * (rank + 1))
        self.rank = rank

    def __iter__(self):
        start = self.start
        end = self.end
        return iter(range(start, end, 1))

    def __len__(self):
        return self.end - self.start


def collate_text_return_length(batch):
    lengths = []
    for elem in batch:
        text = ''.join(elem["text"].strip().split())
        lengths.append(len(text))
    return lengths


class BalancedShardSampler:
    """
    The sampler returns a shard of the input corpus. The text lengths in each shard are roughly the same and are organized descendingly.
    """
    def __init__(self, corpus: Union[datasets.Dataset, torch.utils.data.Dataset], num_replicas: int = 1, rank: int = 0):
        # 1. get the length of each text in the corpus
        if "length" in corpus[0]:
            lengths = []
            for x in corpus:
                lengths.append(x["length"])
        else:
            lengths = []
            # NOTE: we simply use the character length as sequence length so we do not need to call the tokenizer (which is slow)
            with dataset_no_transform(corpus):
                # for x in tqdm(corpus):
                #     text = ''.join(x["text"].strip().split())
                #     lengths.append(len(text))

                dataloader = torch.utils.data.DataLoader(
                    corpus,
                    batch_size=1024,
                    collate_fn=collate_text_return_length,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=False,
                    num_workers=8,
                )
                for x in tqdm(dataloader, desc="Computing Lengths"):
                    lengths.extend(x)

        lengths = np.array(lengths)

        # 2. sort by length
        descending_length_indices = lengths.argsort()[::-1].tolist()
        # descending_length_indices = list(range(len(lengths)))

        # 3. distribute the texts in a zig-zag way
        # [5,4,3,2,1] num_replicas=2 => [5,2,1], [4,3]
        local_indices = []
        bin_range = range(num_replicas)
        for i, idx in enumerate(descending_length_indices):
            # switch the iteration order of the bin_range
            if i % num_replicas == 0:
                if (i // num_replicas) % 2 == 0:
                    bin_indices = iter(bin_range)
                else:
                    bin_indices = reversed(bin_range)
            bin_idx = next(bin_indices)
            if bin_idx == rank:
                local_indices.append(idx)

        self.local_indices = local_indices
    
    def __iter__(self):
        return iter(self.local_indices)

    def __len__(self):
        return len(self.local_indices)
