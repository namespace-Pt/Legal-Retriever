import faiss
import torch

from .args import DenseRetrievalArgs, BM25RetrievalArgs, ModelArgs
from .data import Data, RetrievalDataCollator, GroupByDatasetTrainDataset, EvalDataset_Corpus_Iter
from .metrics import Metrics
from .template import Templater
from .modeling_dense import DenseRetriever, FaissIndex
from .utils import FileLogger, makedirs, rmdirs, split_file_dir_name_ext, get_model_name, get_max_length_in_nested_lists, pad_nested_lists, mask_nested_lists, normalize_text, dataset_no_transform, format_numel_str
from .modeling_utils import evaluate, move_to_device

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


def get_model_and_tokenizer(model_args, device="cpu", evaluation_mode=True, **kwargs):
    import torch
    from loguru import logger
    from dataclasses import asdict
    from transformers.integrations import is_deepspeed_zero3_enabled

    from .args import ModelArgs
    model_args: ModelArgs

    model_args_dict = asdict(model_args)
    model_args_dict.update(**kwargs)

    model_name_or_path = model_args_dict["model_name_or_path"]

    logger.info(f"Loading model and tokenizer from {model_name_or_path}...")

    tokenizer_kwargs = {
        "padding_side": model_args_dict["padding_side"],
        "token": model_args_dict["access_token"],
        "trust_remote_code": True,
        "cache_dir": model_args_dict["model_cache_dir"],
    }

    dtype = model_args_dict["dtype"]
    if dtype == "float32":
        dtype = torch.float32
    elif dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise NotImplementedError
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": {"": device} if not is_deepspeed_zero3_enabled() else None,
        "attn_implementation": model_args_dict["attn_impl"],
        "token": model_args_dict["access_token"],
        "trust_remote_code": True,
        "cache_dir": model_args_dict["model_cache_dir"],
    }

    if model_args_dict["encode_method"] == "dense":
        from src.modeling_dense import DenseRetriever as model_class
        kwargs = {
            "model_name_or_path": model_name_or_path,
            "pooling_method": model_args_dict["pooling_method"],
            "normalize": model_args_dict["normalize"],
            "query_max_length": model_args_dict["query_max_length"],
            "key_max_length": model_args_dict["key_max_length"],
            "packing": model_args_dict["packing"],
            "mrl_dims": model_args_dict["mrl_dims"],
            "mrl_2layer_proj": model_args_dict["mrl_2layer_proj"],
        }
        kwargs["model_kwargs"] = model_kwargs
        kwargs["tokenizer_kwargs"] = tokenizer_kwargs
        # NOTE: load lora inside the model because the inner model is the transformers model
        kwargs["lora_kwargs"] = {
            "lora": model_args_dict["lora"],
            "lora_unload": model_args_dict["lora_unload"]
        }

    elif model_args_dict["encode_method"] == "bm25":
        from src.modeling_bm25 import BM25Retriever as model_class
        tokenizer_kwargs["pretrained_model_name_or_path"] = model_name_or_path
        kwargs = {
            "anserini_dir": model_args_dict["anserini_dir"],
            "k1": model_args_dict["bm25_k1"],
            "b": model_args_dict["bm25_b"],
            "tokenizer_kwargs": tokenizer_kwargs
        }

    elif model_args_dict["encode_method"] == "cross-encoder":
        raise NotImplementedError
    else:
        raise NotImplementedError

    model = model_class(**kwargs)

    return model, model.tokenizer
