import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from functools import partial
from typing import Optional, Dict, Union, Mapping
from transformers import AutoModel, AutoTokenizer
from .template import Templater

logger = logging.getLogger("EmbeddingModel")


class EmbeddingModel(torch.nn.Module):
    MRL_NAME = "mrl_proj_{mrl_dim}"

    def __init__(
        self, 
        model_path, 
        model_kwargs, 
        tokenizer_kwargs: Dict[str, str] = {}, 
        pooling_method: str = "first", 
        normalize: bool = True, 
        mrl_dims: Optional[list[int]] = None, 
        mrl_2layer_proj: Optional[bool] = None, 
        query_max_length: int = 256, 
        key_max_length: int = 256, 
        query_template: str = "no", 
        key_template: str = "no"
    ):
        super().__init__()
        # if "torch_dtype" in model_kwargs:
        #     if model_kwargs["torch_dtype"] == "float16":
        #         model_kwargs["torch_dtype"] = torch.float16
        #     elif model_kwargs["torch_dtype"] == "bfloat16":
        #         model_kwargs["torch_dtype"] = torch.bfloat16
        #     elif model_kwargs["torch_dtype"] == "float32":
        #         model_kwargs["torch_dtype"] = torch.float32

        logger.info(f"Loading model from {model_path}\npooling_method: {pooling_method}, normalize: {normalize}, mrl_dims: {mrl_dims}, query_max_length: {query_max_length}, key_max_length: {key_max_length}, query_template: {query_template}, key_template: {key_template}")

        model = AutoModel.from_pretrained(model_path, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.pooling_method = pooling_method
        self.normalize = normalize
        self.mrl_dims = mrl_dims
        self.query_max_length = query_max_length
        self.key_max_length = key_max_length
        templater = Templater(tokenizer)
        self.query_template_fn = partial(templater.apply, query_template=query_template, max_length=query_max_length)
        self.key_template_fn = partial(templater.apply, query_template=key_template, max_length=key_max_length)
        
        if mrl_dims is not None:
            for mrl_dim in mrl_dims:
                mrl_name = EmbeddingModel.MRL_NAME.format(mrl_dim=mrl_dim)
                if mrl_2layer_proj:
                    projection = torch.nn.Sequential(
                        torch.nn.Linear(model.config.hidden_size, (model.config.hidden_size + mrl_dim) // 2),
                        torch.nn.SiLU(),
                        torch.nn.Linear((model.config.hidden_size + mrl_dim) // 2, mrl_dim),
                    )
                    setattr(self, mrl_name, projection)
                else:
                    projection = torch.nn.Linear(model.config.hidden_size, mrl_dim)
                    setattr(self, mrl_name, projection)
                self.get_mrl_proj(mrl_dim).to(device=model.device, dtype=model.dtype)

                mrl_weight_path = os.path.join(model_path, f"{mrl_name}.pt")

                if os.path.exists(mrl_weight_path):
                    weight = torch.load(mrl_weight_path, map_location=model.device)
                    self.get_mrl_proj(mrl_dim).load_state_dict(weight)
                else:
                    logger.warning(f"MRL projection weight for {mrl_name} not found! Use random initialization instead.")        

    @property
    def device(self):
        return self.model.device

    @property
    def mrl_dim(self):
        if self.mrl_dims is None:
            return None
        else:
            return self.mrl_dims[0]

    @property
    def ndim(self):
        if self.mrl_dim is None or self.mrl_dim == False:
            return self.model.config.hidden_size
        else:
            return self.mrl_dim

    def get_mrl_proj(self, mrl_dim: int = None):
        if mrl_dim is None or mrl_dim == False:
            return lambda x: x
        else:
            return getattr(self, f"mrl_proj_{mrl_dim}")
        
    def _pool(
        self, 
        last_hidden_states: torch.Tensor, 
        pooling_method: str = "last",
        attention_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
    ):
        """
        Pool the last_hidden_states along the sequence dimension. Handle packed inputs as well.
        """
        if position_ids is None:
            # NOTE: no packing
            # last_hidden_states: batch_size (* group_size), seq_len, d_embed
            if pooling_method == "first":
                embedding = last_hidden_states[:, 0]
            elif pooling_method == "last":
                embedding = last_hidden_states[:, -1]
            elif pooling_method == "mean":
                last_hidden_states = last_hidden_states.masked_fill(
                    ~attention_mask[..., None].bool(), 0.0)
                embedding = last_hidden_states.sum(
                    dim=1) / attention_mask.sum(dim=1, keepdim=True)
            elif pooling_method == "weighted-mean":
                attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
                s = torch.sum(last_hidden_states * attention_mask_.unsqueeze(-1).float(), dim=1)
                d = attention_mask_.sum(dim=1, keepdim=True).float()
                embedding = s / d
            else:
                raise NotImplementedError(f"Pooling_method {pooling_method} not implemented!")
        else:
            # NOTE: packing
            # last_hidden_states: 1, all_seq_len, d_embed
            # position_ids: 1, all_seq_len

            # all_seq_len, d_embed
            last_hidden_states = last_hidden_states[0]
            position_ids = position_ids[0]

            sequence_start_pos = position_ids == 0

            if pooling_method == "first":
                embedding = last_hidden_states[sequence_start_pos]
            elif pooling_method == "last":
                indices = torch.arange(len(position_ids), device=position_ids.device) - 1
                indices = indices[sequence_start_pos]
                # get the index of the last token in each sequence
                indices[:-1] = indices[1:].clone()
                indices[-1] = len(position_ids) - 1
                embedding = last_hidden_states[indices]
            elif pooling_method == "mean":
                embedding = torch.zeros_like(last_hidden_states[sequence_start_pos])
                indices = sequence_start_pos.cumsum(-1) - 1
                # accumulate hidden states of the same sequence
                embedding.index_add_(0, indices, last_hidden_states)
                # compute sequence lengths
                zero_indices = sequence_start_pos.nonzero(as_tuple=True)[0]
                indices = torch.cat([zero_indices, torch.tensor([len(position_ids)], dtype=zero_indices.dtype, device=zero_indices.device)])
                lengths = indices[1:] - indices[:-1]
                # mean over sequence
                embedding = embedding / lengths.unsqueeze(-1)
            else:
                raise NotImplementedError(f"Pooling method {pooling_method} is currently not supported for packed inputs!")

        return embedding
        
    @torch.no_grad()
    def encode(self, sentences, field: str = "key", mrl_dim: Optional[int] = None, batch_size: int = 128, do_template: bool = False):
        if mrl_dim is None:
            mrl_dim = self.mrl_dim
        
        is_single = False
        if isinstance(sentences, str):
            is_single = True
            sentences = [sentences]

        if batch_size is None:
            batch_size = len(sentences)
            
        lengths = np.array([len(x) for x in sentences])
        descending_length_indices = lengths.argsort()[::-1].tolist()

        def collate_fn(data):
            text = data
            if do_template:
                if field == "query":
                    text = self.key_template_fn(text)
                if field == "key":
                    text = self.query_template_fn(text)
            max_length = self.query_max_length if field == "query" else self.key_max_length
            inputs = self.tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            return inputs

        # use dataloader to speed up tokenization
        dataloader = torch.utils.data.DataLoader(
            sentences,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            sampler=Sampler(descending_length_indices),
            pin_memory=True,
            num_workers=16,
        )

        all_embeddings = np.zeros((len(sentences), self.ndim), dtype=np.float32)
        
        start_idx = end_idx = 0

        for batch_inputs in tqdm(dataloader, desc="Encoding"):
            batch_inputs = move_to_device(batch_inputs, self.device)
            hidden_states = self.model(**batch_inputs).last_hidden_state
            embedding = self._pool(hidden_states, self.pooling_method, batch_inputs["attention_mask"])

            if self.normalize:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            end_idx = start_idx + len(embedding)
            indices = descending_length_indices[start_idx: end_idx]
            all_embeddings[indices] = embedding.cpu().numpy()
            start_idx = end_idx

        if is_single:
            return all_embeddings[0]

        return all_embeddings


class Sampler:
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __iter__(self):
        return iter(self.indices)


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