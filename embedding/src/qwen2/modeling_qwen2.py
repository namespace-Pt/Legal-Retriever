import torch
from typing import Optional, Mapping, Union, List, Callable
from transformers import Qwen2PreTrainedModel, Qwen2Model


class Qwen2EmbeddingModel(Qwen2PreTrainedModel):
    MRL_NAME = "mrl_proj_{mrl_dim}"

    def __init__(self, config):
        super().__init__(config)

        model = Qwen2Model(config)
        self.model = model

        self.mrl_dims = config.mrl_dims
        self.mrl_2layer_proj = config.mrl_2layer_proj
        self.normalize = config.normalize
        self.pooling_method = config.pooling_method

        for mrl_dim in self.mrl_dims:
            mrl_name = Qwen2EmbeddingModel.MRL_NAME.format(mrl_dim=mrl_dim)
            if self.mrl_2layer_proj:
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
        
    def forward(self, input_ids, attention_mask=None, position_ids=None):
        # NOTE: ugly hard code because exporting to onnx forbids custom inputs
        mrl_dim = self.mrl_dims[0]

        if position_ids is not None:
            # NOTE: packing case
            # (1, all_sequence_length, d_embed)
            embeddings = self.model(input_ids=input_ids, position_ids=position_ids).last_hidden_state
            # (batch_size, d_embed)
            embedding = self._pool(embeddings, pooling_method=self.pooling_method, position_ids=position_ids)

        else:
            # NOTE: padding case
            # (batch_size, seq_len, d_embed)
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            # (batch_size, d_embed)
            embedding = self._pool(embeddings, pooling_method=self.pooling_method, attention_mask=attention_mask)

        # NOTE: transform to a given dimension
        # (batch_size, seq_len, d_mrl)
        embedding = self.get_mrl_proj(mrl_dim)(embedding)

        if self.normalize:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding