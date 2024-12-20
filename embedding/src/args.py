import os
import json
import glob
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Union
from .utils import split_file_dir_name_ext


@dataclass
class BaseArgs:
    model_cache_dir: str = field(
        default=None,
        metadata={'help': 'Default path to save language models.'}
    )
    dataset_cache_dir: str = field(
        default=None,
        metadata={'help': 'Default path to save huggingface datasets.'}
    )
    data_root: str = field(
        default=None, 
        metadata={'help': 'The base directory storing all data used for training and evaluation. If specified, make sure all train_data, eval_data, and corpus are path relative to data_root!'},
    )
    train_data: Optional[List[str]] = field(
        default=None,
        metadata={'help': 'Training json files.'},
    )
    eval_data: Optional[List[str]] = field(
        default=None,
        metadata={'help': 'Evaluation json files.'},
    )
    corpus: Optional[List[str]] = field(
        default=None,
        metadata={'help': 'Corpus json files.'}
    )

    model_name_or_path: str = field(
<<<<<<< HEAD
        default="/data/peitian/Data/hf-models/bge-m3",
=======
        default="/mnt/bn/search-douyin-rank-yg/all_data_from_lf/llm_models/Qwen2-0.5B.with_fasttokenizer/",
>>>>>>> cb248046b6b6c650cd0d48d3076d95b11a8804eb
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    access_token: Optional[str] = field(
        default="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",
        metadata={'help': 'Huggingface access token.'}
    )
    attn_impl: Optional[str] = field(
<<<<<<< HEAD
        default="eager",
=======
        default="flash_attention_2",
>>>>>>> cb248046b6b6c650cd0d48d3076d95b11a8804eb
        metadata={'help': 'The implementation of attention.'}
    )
    chat_template: str = field(
        default="hf",
        metadata={'help': 'Instruction template name in fastchat.'}
    )
    padding_side: str = field(
        default="left",
        metadata={'help': 'Tokenizer padding side.'}
    )

    lora: Optional[str] = field(
        default=None,
        metadata={'help': 'LoRA ID.'},
    )
    lora_unload: bool = field(
        default=True,
        metadata={'help': 'Merge and unload LoRA?'},
    )

    dtype: str = field(
        default="float16",
        metadata={'help': 'Data type for embeddings.'}
    )
    cpu: bool = field(
        default=False,
        metadata={'help': 'Use cpu?'}
    )

    query_template: str = field(
        default="no",
        metadata={'help': 'The template name in data.py or the template string.'}
    )
    key_template: str = field(
        default="no",
        metadata={'help': 'The template name in data.py or the template string.'}
    )
    hits: int = field(
        default=10,
        metadata={'help': 'How many keys to retrieve/rerank?'}
    )
    metrics: List[str] = field(
        default_factory=lambda: ["mrr", "recall", "ndcg"],
        metadata={'help': 'List of metrics'}
    )
    cutoffs: List[int] = field(
        default_factory=lambda: [1, 10],
        metadata={'help': 'Cutoffs to evaluate retrieval metrics.'}
    )

    load_result: bool = field(
        default=False,
        metadata={'help': 'Load retrieval results directly?'}
    )
    save_result: bool = field(
        default=True,
        metadata={'help': 'Save retrieval results?'}
    )
    save_name: Optional[str] = field(
        default=None,
        metadata={'help': 'Name suffix of the json file when saving the collated retrieval results and indexes.'}
    )
    # result_dir: Optional[str] = field(
    #     default=None,
    #     metadata={'help': 'Sub-directory name where the results (retrieved keys, mined negatives, embeddings, and indexes) will be saved. Default to model_name_or_path. ({output_dir/result_dir/xxx})'}
    # )
    save_to_output: bool = field(
        default=False,
        metadata={'help': 'Save the result/key/negative to output_dir? If not true, they will be saved next to the eval_data.'}
    )
    neg_num: int = field(
        default=20,
        metadata={'help': 'Maximum negative number to mine.'}
    )
    neg_filter_answer: bool = field(
        default=False,
        metadata={'help': 'Remove negatives that contain the desired answer when collating negatives?'}
    )
    neg_filter_fuzz: Optional[int] = field(
        default=None,
        metadata={'help': 'Remove negatives whose fuzz score against positives are above this specified threshold.'}
    )
    neg_keep_original: Optional[int] = field(
        default=None,
        metadata={'help': 'How many original negatives to preserve in the newly collated negatives?'}
    )
    
    def resolve_path(self, path):
        """Resolve any path starting with 'embedding:' to relative path against data_root."""
        pattern = "embedding:"
        # resolve relative data paths when necessary
        if isinstance(path, list):
            for i, x in enumerate(path):
                if x.startswith(pattern):
                    path[i] = os.path.join(self.data_root, x.replace(pattern, ""))
        else:
            if path.startswith(pattern):
                path = os.path.join(self.data_root, path.replace(pattern, ""))

        # expand glob
        if isinstance(path, list):
            new_path = []
            for p in path:
                if "*" in p:
                    new_path.extend(glob.glob(p))
                else:
                    new_path.append(p)
            path = new_path
        else:
            if "*" in path:
                path = glob.glob(path)

        return path

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @property
    def model_short_name(self):
        if self.lora is None:
            parent, model_name, _ = split_file_dir_name_ext(self.model_name_or_path)
            _, parent_name, _ = split_file_dir_name_ext(parent)
        else:
            parent, model_name, _ = split_file_dir_name_ext(self.lora)
            _, parent_name, _ = split_file_dir_name_ext(parent)
        return "--".join([parent_name, model_name])

    def __post_init__(self):        
        if self.train_data is not None:
            self.train_data = self.resolve_path(self.train_data)

        if self.eval_data is not None:
            self.eval_data = self.resolve_path(self.eval_data)

        if self.corpus is not None:
            self.corpus = self.resolve_path(self.corpus)

        if hasattr(self, "output_dir") and self.output_dir is not None:
            self.output_dir = self.resolve_path(self.output_dir)


@dataclass
class DenseRetrievalArgs(BaseArgs):
    query_max_length: int = field(
        default=512,
        metadata={'help': 'Max query length.'}
    )
    key_max_length: int = field(
        default=512,
        metadata={'help': 'Max key length.'}
    )
    pooling_method: str = field(
<<<<<<< HEAD
        default="first",
=======
        default="last",
>>>>>>> cb248046b6b6c650cd0d48d3076d95b11a8804eb
        metadata={'help': 'Pooling methods to aggregate token embeddings for a sequence embedding. {first, last, mean}'}
    )
    normalize: bool = field(
        default=True,
        metadata={'help': 'L2-normalize embeddings?'}
    )
    faiss_index_factory: str = field(
        default="Flat",
        metadata={'help': 'Index factory string for faiss.'}
    )
    batch_size: int = field(
        default=128,
        metadata={'help': 'Batch size for encoding, indexing, and retrieval.'}
    )
    packing: bool = field(
        default=False,
        metadata={'help': 'Pack queries and keys to eliminate padding?'}
    )
    mrl_dims: Optional[List[int]] = field(
        default=None,
        metadata={'help': 'Transform the embedding to a given set of dimensions through linear projection.'}
    )
    mrl_2layer_proj: Optional[bool] = field(
        default=None,
        metadata={'help': 'Use two-layer MLP to transform MRL?'}
    )

    load_encode: bool = field(
        default=False,
        metadata={'help': 'Load cached embeddings?'}
    )
    save_encode: bool = field(
        default=False,
        metadata={'help': 'Save embeddings?'}
    )
    load_index: bool = field(
        default=False,
        metadata={'help': 'Load cached index?'}
    )
    save_index: bool = field(
        default=False,
        metadata={'help': 'Save index?'}
    )


@dataclass
class BM25RetrievalArgs(BaseArgs):
    anserini_dir: str = field(
        default='/share/peitian/Apps/anserini',
        metadata={'help': 'Anserini installation directory.'}
    )
    bm25_k1: float = field(
        default=0.82,
        metadata={'help': 'BM25 k1.'}
    )
    bm25_b: float = field(
        default=0.68,
        metadata={'help': 'BM25 b.'}
    )
    anserini_store_docvec: bool = field(
        default=False,
        metadata={'help': 'Store document vector? Useful when you want to inspect the word-level statistics (tf-idf) after index construction.'}
    )
    anserini_language: str = field(
        default="en",
        metadata={'help': 'Language.'}
    )
    anserini_threads: int = field(
        default=32,
        metadata={'help': 'Indexing/Searching thread number.'}
    )
    anserini_parallel: int = field(
        default=4,
        metadata={'help': 'Indexing/Searching process number.'}
    )
    load_index: bool = field(
        default=False,
        metadata={'help': 'Load index?'}
    )
    load_collection: bool = field(
        default=False,
        metadata={'help': 'Load collection?'}
    )


@dataclass
class ModelArgs(DenseRetrievalArgs, BM25RetrievalArgs):
    encode_method: str = field(
        default="dense",
        metadata={'help': 'How to retrieve? {dense, bm25, random, no, crossenc}'}
    )
    eval_method: str = field(
        default="retrieval",
        metadata={'help': 'How to evaluate? {retrieval, rerank, encode}'},
    )
