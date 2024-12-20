import os
import gc
import faiss
import torch
import accelerate
import datasets
import numpy as np
import torch.distributed as dist

from loguru import logger
from packaging import version
from tqdm import tqdm
from peft import PeftModel
from typing import Optional, Mapping, Union, List, Callable
from transformers import AutoModel, AutoTokenizer

from .data import RetrievalDataCollator
from .modeling_utils import optional_grad_ctx, move_to_device, save_to_memmap, distribute_inputs, BalancedShardSampler, DummyCorpusDataset


MRL_NAME = "mrl_proj_{mrl_dim}"


class DenseRetriever(torch.nn.Module):
    # involked in MTEB evaluation
    _is_efficient = True

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-icl-en",
        pooling_method: str = "last",
        normalize: bool = True,
        query_max_length: int = 1024,
        key_max_length: int = 1024,
        packing: bool = False,
        model_kwargs: dict = {},
        tokenizer_kwargs: dict = {},
        lora_kwargs: dict = {},
        mrl_dims: Optional[List[int]] = None,
        mrl_2layer_proj: Optional[bool] = None,
    ):
        super().__init__()
        model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)            
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # NOTE: in case we add special tokens to the tokenizer
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))

        if lora_kwargs.get("lora", None) is not None:
            logger.info(f"Loading LoRA from {lora_kwargs['lora']}...")
            lora_path = lora_kwargs["lora"]
            model = PeftModel.from_pretrained(
                model, 
                lora_path,
                torch_dtype=model.dtype,
                device_map={"": model.device},
            )
            if lora_kwargs.get("lora_unload", False):
                model = model.merge_and_unload()

        self.model = model
        self.tokenizer = tokenizer
        self.pooling_method = pooling_method
        self.normalize = normalize
        self.query_max_length = query_max_length
        self.key_max_length = key_max_length
        self.packing = packing

        if self.packing:
            assert self.model.config._attn_implementation == "flash_attention_2", f"Make sure to use flash_attention_2 when enabling packing!"
            import transformers
            hf_version = version.parse(transformers.__version__)
            assert hf_version >= version.parse("4.44.0"), f"Make sure the transformers version is at least 4.44 when enabliing packing!"

        if mrl_dims is not None:
            for mrl_dim in mrl_dims:
                mrl_name = MRL_NAME.format(mrl_dim=mrl_dim)
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

                if lora_kwargs.get("lora", None) is not None:
                    mrl_weight_path = os.path.join(lora_path, f"{mrl_name}.pt")
                else:
                    mrl_weight_path = os.path.join(model_name_or_path, f"{mrl_name}.pt")

                if os.path.exists(mrl_weight_path):
                    weight = torch.load(mrl_weight_path, map_location=model.device)
                    self.get_mrl_proj(mrl_dim).load_state_dict(weight)
                else:
                    logger.warning(f"MRL projection weight for {mrl_name} not found! Use random initialization instead.")
        else:
            mrl_dims = [None]

        self.mrl_dims = mrl_dims

        self._index = None

        self.eval()

    @property
    def config(self):
        return self.model.config

    @property
    def mrl_dim(self):
        return self.mrl_dims[0]

    @property
    def ndim(self):
        if self.mrl_dim is None or self.mrl_dim == False:
            return self.model.config.hidden_size
        else:
            return self.mrl_dim

    @property
    def device(self):
        return self.model.device

    @property
    def num_processes(self):
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1

    @property
    def process_index(self):
        if dist.is_initialized():
            return dist.get_rank()
        else:
            return 0
        
    @property
    def node_rank(self):
        if dist.is_initialized():
            return int(os.environ["GROUP_RANK"])
        else:
            return 0

    def get_mrl_proj(self, mrl_dim: Optional[int] = None):
        if mrl_dim is None or mrl_dim == False:
            return lambda x: x
        else:
            return getattr(self, f"mrl_proj_{mrl_dim}")

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        self.model.gradient_checkpointing_disable(*args, **kwargs)

    def save_pretrained(self, output_dir, *args, **kwargs):
        self.model.save_pretrained(output_dir, *args, **kwargs)
        self.tokenizer.save_pretrained(output_dir)
        for mrl_dim in self.mrl_dims:
            mrl_proj = self.get_mrl_proj(mrl_dim)
            if mrl_dim is not None:
                torch.save(mrl_proj.state_dict(), os.path.join(output_dir, f"{MRL_NAME.format(mrl_dim=mrl_dim)}.pt"))

    def _gather_tensors(self, local_tensor):
        """
        Gather tensors from all processes.

        Args:
            local_tensor: the tensor that needs to be gathered

        Returns:
            concatenation of local_tensor in each process
        """
        if local_tensor is None:
            return None
        all_tensors = [torch.empty_like(local_tensor)
                       for _ in range(self.num_processes)]
        dist.all_gather(all_tensors, local_tensor.contiguous())
        # NOTE: assign gradient?
        all_tensors[self.process_index] = local_tensor
        return torch.stack(all_tensors, dim=0)

    def _gather_node_ranks(self):
        node_rank = torch.tensor([self.node_rank], device=self.device)
        all_node_ranks = [torch.empty_like(node_rank) for _ in range(self.num_processes)]
        dist.all_gather(all_node_ranks, node_rank)
        all_node_ranks = torch.cat(all_node_ranks)
        return all_node_ranks
    
    def _gather_process_indices(self):
        process_index = torch.tensor([self.process_index], device=self.device)
        all_indices = [torch.empty_like(process_index) for _ in range(self.num_processes)]
        dist.all_gather(all_indices, process_index)
        all_indices = torch.cat(all_indices)
        return all_indices

    def _prepare(self, inputs: Union[str, List[str], Mapping[str, torch.Tensor]], field: str = "key"):
        """
        Convert inputs into tokenized input_ids. Pack inputs if self.packing is True.

        Returns:
            a dictionary (no packing): {
                "input_ids": batch_size, seq_len
                "attention_mask": batch_size, seq_len
            }

            or, 

            a dictionary (packing): {
                "input_ids": 1, all_seq_len
                "position_ids": 1, all_seq_len
            }
        """

        # wrap the single input string with a list
        if isinstance(inputs, str):
            inputs = [inputs]

        if isinstance(inputs, list) and isinstance(inputs[0], str):
            if field == "query":
                if self.packing:
                    input_ids = self.tokenizer(inputs, truncation=True, max_length=self.query_max_length).input_ids
                    position_ids = [list(range(len(x))) for x in input_ids]
                    # 1, all_seq_len
                    inputs = {
                        "input_ids": torch.tensor([sum(input_ids, [])]), 
                        "position_ids": torch.tensor([sum(position_ids, [])])
                    }
                else:
                    inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=self.query_max_length)

            elif field == "key":
                if self.packing:
                    input_ids = self.tokenizer(inputs, truncation=True, max_length=self.key_max_length).input_ids
                    position_ids = [list(range(len(x))) for x in input_ids]
                    # 1, all_seq_len
                    inputs = {
                        "input_ids": torch.tensor([sum(input_ids, [])]), 
                        "position_ids": torch.tensor([sum(position_ids, [])])
                    }
                else:
                    inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=self.key_max_length)

        elif isinstance(inputs, Mapping) and "input_ids" in inputs:
            # NOTE: we do not check the length of Mapping inputs, because it may be packed
            pass

        else:
            raise ValueError(f"Expected inputs of type str, list[str], or dict, got {type(inputs)}!")
        
        inputs = move_to_device(inputs, self.device)

        return inputs

    def _pool(
        self, 
        last_hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
    ):
        """
        Pool the last_hidden_states along the sequence dimension. Handle packed inputs as well.
        """
        if position_ids is None:
            # no packing
            # last_hidden_states: batch_size (* group_size), seq_len, d_embed
            if self.pooling_method == "first":
                embedding = last_hidden_states[:, 0]
            elif self.pooling_method == "last":
                embedding = last_hidden_states[:, -1]
            elif self.pooling_method == "mean":
                last_hidden_states = last_hidden_states.masked_fill(
                    ~attention_mask[..., None].bool(), 0.0)
                embedding = last_hidden_states.sum(
                    dim=1) / attention_mask.sum(dim=1, keepdim=True)
            elif self.pooling_method == "weighted-mean":
                attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
                s = torch.sum(last_hidden_states * attention_mask_.unsqueeze(-1).float(), dim=1)
                d = attention_mask_.sum(dim=1, keepdim=True).float()
                embedding = s / d
            else:
                raise NotImplementedError(f"Pooling_method {self.pooling_method} not implemented!")
        else:
            # packing
            # last_hidden_states: 1, all_seq_len, d_embed
            # position_ids: 1, all_seq_len

            # all_seq_len, d_embed
            last_hidden_states = last_hidden_states[0]
            position_ids = position_ids[0]

            sequence_start_pos = position_ids == 0

            if self.pooling_method == "first":
                embedding = last_hidden_states[sequence_start_pos]
            elif self.pooling_method == "last":
                indices = torch.arange(len(position_ids), device=position_ids.device) - 1
                indices = indices[sequence_start_pos]
                # get the index of the last token in each sequence
                indices[:-1] = indices[1:].clone()
                indices[-1] = len(position_ids) - 1
                embedding = last_hidden_states[indices]
            elif self.pooling_method == "mean":
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
                raise NotImplementedError(f"Pooling method {self.pooling_method} is currently not supported for packed inputs!")

        return embedding

    def _encode(self, inputs: Union[str, List[str], Mapping], field: str = "query", normalize: Optional[bool] = None, mrl_dim: Optional[Union[int, bool]] = None):
        """Encode inputs into embeddings

        Args:
            inputs: can be string, list of strings, or BatchEncoding results from tokenizer
            field: query or key
            normalize: l2-normalize the embedding? If None, default to self.normalize. If False, no normalization will be applied.
            mrl_dim: the mrl projection dimension. If None, default to self.mrl_dim. If False, no mrl projection will be applied.
        Returns:
            Tensor: [batch_size, d_embed]
        """
        if normalize is None:
            normalize = self.normalize
        if mrl_dim is None:
            mrl_dim = self.mrl_dim

        with optional_grad_ctx(self.training):
            inputs = self._prepare(inputs, field=field)

            if "position_ids" in inputs:
                # 1, all_seq_len, d_embed
                embeddings = self.model(**inputs).last_hidden_state
                # num_seq, d_embed
                embedding = self._pool(embeddings, position_ids=inputs["position_ids"])
                
            else:
                # batch_size, seq_len, d_embed
                embeddings = self.model(**inputs).last_hidden_state
                embedding = self._pool(embeddings, inputs["attention_mask"])

            # NOTE: transform to a given dimension
            embedding = self.get_mrl_proj(mrl_dim)(embedding)

            if normalize:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            return embedding
    
    def _compute_similarity(self, query_embedding, key_embedding):
        # NOTE: use fp32 for matmul
        return torch.matmul(query_embedding.float(), key_embedding.transpose(-2, -1).float())

    def _compute_listwise_loss(
        self, 
        # (batch_size, d_embed)
        query_embedding: torch.Tensor, 
        # (batch_size * group_size, d_embed)
        key_embedding: torch.Tensor,
        teacher_scores: Optional[torch.Tensor] = None,
        listwise_temp: float = 0.02,
        contrastive_weight: float = 1.,
        distill_student_temp: float = 1.,
        distill_teacher_temp: float = 1.,
        distill_weight: float = 1.,
        use_inbatch_neg: Optional[bool] = None,
        use_cross_device_neg: Optional[str] = None,
        filter_inbatch_neg: Optional[bool] = None,
        **kwargs,
    ):
        # if self.process_index == 0:
        #     print(f"Listwise Temperature: {listwise_temp}")

        if teacher_scores is not None and distill_weight > 0:
            do_distill = True
            student_query = query_embedding.unsqueeze(1)    # B, 1, D
            student_key = key_embedding.view(student_query.shape[0], -1, student_query.shape[-1])   # B, N, D
            student_scores = self._compute_similarity(student_query, student_key).squeeze(1)         # B, N
            student_scores = student_scores  / listwise_temp

            student_scores = torch.nn.functional.log_softmax(student_scores / distill_student_temp, dim=-1)
            teacher_scores = torch.nn.functional.softmax(teacher_scores / distill_teacher_temp, dim=-1)
            distill_loss = torch.nn.functional.kl_div(student_scores, teacher_scores, reduction="batchmean")
        else:
            do_distill = False

        if contrastive_weight > 0:
            do_contrastive = True

            if use_inbatch_neg:
                if use_cross_device_neg is not None:
                    # gather with grad
                    # n_nodes * n_procs, batch_size, d_embed
                    query_embedding = self._gather_tensors(query_embedding)
                    # n_nodes * n_procs, batch_size * (1 + n_neg), d_embed
                    key_embedding = self._gather_tensors(key_embedding)
                    if use_cross_device_neg == "all":
                        query_embedding = query_embedding.flatten(0, 1)
                        key_embedding = key_embedding.flatten(0, 1)
                    elif use_cross_device_neg != -1:
                        # slice out embeddings in the current node
                        # n_nodes * n_procs
                        all_process_indices = self._gather_process_indices()
                        group_indices = all_process_indices // use_cross_device_neg
                        current_group_index = group_indices[self.process_index]
                        same_group_indicator = group_indices == current_group_index
                        # if self.process_index in [0, 7]:
                        #     print(f"Rank {self.process_index} Indicator: {same_group_indicator} Shape: {query_embedding.shape}")
                        # n_procs
                        query_embedding = query_embedding[same_group_indicator].flatten(0,1)
                        key_embedding = key_embedding[same_group_indicator].flatten(0,1)
                    else:
                        raise ValueError(f"Invalid setting of use_cross_device_neg={use_cross_device_neg}! Please use 'all' or 'intra-node'!")

                scores = self._compute_similarity(query_embedding, key_embedding)
                scores = scores / listwise_temp

                group_size = key_embedding.shape[0] // query_embedding.shape[0]

                if filter_inbatch_neg:
                    batch_size = query_embedding.shape[0]
                    min_value = torch.finfo(scores.dtype).min
                    # NOTE: filter out in-batch positives (set the corresponding scores to -inf)
                    # batch_size, d_embed
                    pos_embedding = key_embedding[0::group_size]
                    # batch_size, batch_size * group_size
                    l2_dist = torch.cdist(pos_embedding, key_embedding, p=2)
                    # NOTE: mask out true positives
                    # batch_size, batch_size * group_size
                    l2_dist[range(batch_size), range(0, batch_size * group_size, group_size)] = 1
                    scores = scores.masked_fill(l2_dist == 0, min_value)

                # in batch negative
                labels = torch.arange(query_embedding.shape[0], device=self.device)
                labels = labels * group_size
                contrastive_loss = torch.nn.functional.cross_entropy(scores, labels)
            else:
                scores = self._compute_similarity(
                    query_embedding.unsqueeze(1), 
                    key_embedding.reshape(query_embedding.shape[0], -1, query_embedding.shape[-1])
                ).squeeze(1)
                scores = scores / listwise_temp
                labels = torch.zeros(scores.shape[0], device=scores.device, dtype=torch.long)
                contrastive_loss = torch.nn.functional.cross_entropy(scores, labels)

        else:
            do_contrastive = False

        if do_distill and do_contrastive:
            loss = contrastive_loss * contrastive_weight + distill_loss * distill_weight
        elif do_distill:
            loss = distill_loss
        elif do_contrastive:
            loss = contrastive_loss
        else:
            raise ValueError(f"Neither distill or contrastive learning is enabled!")
        
        return loss

    def _compute_pointwise_loss(
        self, 
        # (batch_size, d_embed)
        query_embedding: torch.Tensor, 
        # (batch_size * group_size, d_embed)
        key_embedding: torch.Tensor, 
        # (batch_size)
        pointwise_labels: Optional[torch.Tensor] = None,
        pointwise_temp: float=0.2,
        **kwargs
    ):
        # if self.process_index == 0:
        #     print(f"Pointwise Temperature: {pointwise_temp}")

        # (batch_size, batch_size * group_size)
        query_embedding = query_embedding.unsqueeze(1)    # B, 1, D
        key_embedding = key_embedding.view(query_embedding.shape[0], -1, query_embedding.shape[-1])   # B, N, D
        scores = self._compute_similarity(query_embedding, key_embedding).squeeze(1)         # B, N
        logits = torch.nn.functional.sigmoid(scores / pointwise_temp)

        if pointwise_labels is not None:
            assert logits.shape[1] == 1
            # NOTE: align dtype
            loss = torch.nn.functional.binary_cross_entropy(logits.squeeze(1), pointwise_labels.to(logits.dtype))
        else:
            # the first column is positive, others are negatives
            labels = torch.ones_like(logits)
            labels[:, 1:] = 0
            loss = torch.nn.functional.binary_cross_entropy(logits, labels)
        return loss

    def forward(
        self, 
        query: Mapping[str, torch.Tensor], 
        key: Mapping[str, torch.Tensor], 
        train_method: str = "listwise",
        **kwargs
    ):
        """
        Forward computation.
        """
        # NOTE: we first get the raw embedding WITHOUT normalization and mrl projection
        # batch_size, d_embed
        raw_query_embedding = self._encode(query, field="query", normalize=False, mrl_dim=False)
        # batch_size * group_size, d_embed
        raw_key_embedding = self._encode(key, field="key", normalize=False, mrl_dim=False)

        pointwise_weight = kwargs["pointwise_weight"]
        listwise_weight = kwargs["listwise_weight"]

        mrl_losses = []

        for mrl_dim in self.mrl_dims:
            mrl_loss = 0

            query_embedding = self.get_mrl_proj(mrl_dim)(raw_query_embedding)
            key_embedding = self.get_mrl_proj(mrl_dim)(raw_key_embedding)

            if self.normalize:
                query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
                key_embedding = torch.nn.functional.normalize(key_embedding, p=2, dim=1)

            if train_method in ["listwise", "both"] and listwise_weight > 0:
                contra_loss = self._compute_listwise_loss(query_embedding, key_embedding, **kwargs)
                mrl_loss += listwise_weight * contra_loss
                if self.process_index == 0:
                    print(f"Contrastive Loss: {contra_loss}")
            if train_method in ["pointwise", "both"] and pointwise_weight > 0:
                point_loss = self._compute_pointwise_loss(query_embedding, key_embedding, **kwargs)
                mrl_loss += pointwise_weight * point_loss
                if self.process_index == 0:
                    print(f"Pointwise Loss: {point_loss}")

            mrl_losses.append(mrl_loss)
        
        loss = sum(mrl_losses) / len(mrl_losses)

        return {"loss": loss}

    @torch.no_grad()
    def encode(
        self, 
        inputs: Union[str, List[str]], 
        field: str = "query", 
        template_fn: Optional[Callable] = None, 
        distributed: Optional[bool] = None, 
        batch_size: Optional[int] = None,
        show_progress: Optional[bool] = None,
        **kwargs
    ):
        """Encode inputs into embeddings. Usually involked by MTEB evaluation.

        Args:
            inputs: can be string, list of strings, or BatchEncoding results from tokenizer
            field: query or key
            template_fn: function to decorate the inputs before encoding
            distributed: split the inputs to all processes, where each process encodes local inputs and then gather the results
            batch_size: batch size for encoding

        Returns:
            np.ndarray (float32): [num_inputs, d_embed]
        """
        if isinstance(inputs, str):
            is_single = True
            inputs = [inputs]
        else:
            is_single = False

        embeddings = np.zeros((len(inputs), self.ndim), dtype=np.float32)

        if distributed:
            inputs, is_distributed, (local_start_idx, local_end_idx) = distribute_inputs(inputs, num_processes=self.num_processes, process_index=self.process_index)
        else:
            is_distributed = False
            local_start_idx = 0

        if batch_size is None:
            batch_size = len(inputs)

        dataloader = torch.utils.data.DataLoader(
            inputs,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            num_workers=0,
        )

        start_idx = local_start_idx
        for batch_inputs in tqdm(dataloader, desc="Encoding", disable=not show_progress):
            # NOTE: apply template
            if template_fn is not None:
                batch_inputs = template_fn(batch_inputs)

            embedding = self._encode(batch_inputs, field=field).to(device="cpu", dtype=torch.float32).numpy()

            end_idx = start_idx + len(embedding)

            # print(f"Rank {self.process_index} start {start_idx} end {end_idx} embedding.shape {embedding.shape}")

            # gather embeddings from all processes (we can do this once at the end, but that may incur intensive communication overhead)
            # so, we gather embeddings for every batch
            if is_distributed:
                obj_list = [None for _ in range(self.num_processes)]
                obj = (embedding, start_idx, end_idx)
                dist.all_gather_object(obj_list, obj)

                for j, (embedding, s, e) in enumerate(obj_list):
                    embeddings[s: e] = embedding
            else:
                embeddings[start_idx: end_idx] = embedding
            
            start_idx = end_idx

        if is_single:
            return embeddings[0]
        else:
            return embeddings

    @torch.no_grad()
    def index(
        self, 
        corpus: Union[datasets.Dataset, list[str]], 
        template_fn: Optional[Callable] = None,
        index_factory: str = "Flat", 
        save_index: bool = False,
        load_index: bool = False,
        save_encode: bool = False,
        load_encode: bool = False,
        batch_size: int = 128,
        output_dir: str = "data/results",
        save_name: Optional[str] = None,
        accelerator: Optional[accelerate.Accelerator] = None,
        skip_indexing: Optional[bool] = None,
        **kwargs,
    ):
        """
        Index the corpus.

        1. split the corpus into disjoint shards
        2. encode local shard on each process
        3. save the encoded result or the index if specified
        """
        num_shards = self.num_processes
        shard_idx = self.process_index

        if save_name is None:
            name = ""
        else:
            name = f".{save_name}"
        encode_path = os.path.join(output_dir, f"embeddings{name}.memmap")
        index_path = os.path.join(output_dir, "index{name}.{index_factory}.{shard_idx}-{num_shards}.faiss")

        if accelerator is not None:
            accelerator.wait_for_everyone()

        if isinstance(corpus, datasets.Dataset) and "text" in corpus.column_names:
            pass
        elif isinstance(corpus, list) and isinstance(corpus[0], str):
            corpus = DummyCorpusDataset(corpus)
        else:
            raise ValueError(f"Make sure the corpus is a dataset with 'text' column, or a list of strings.")

        # splits the corpus into disjoint shards, the tokens within each shard are roughly the same
        sampler = BalancedShardSampler(corpus, num_shards, shard_idx)

        # the mapping from local shard to the corpus
        local_shard_indices = np.array(sampler.local_indices)

        if accelerator is not None:
            accelerator.wait_for_everyone()

        if load_encode:
            logger.info("Loading embeddings from {encode_path}...")
            encoded_corpus = np.memmap(
                encode_path,
                mode="r",
                dtype=np.float16
            ).reshape(len(corpus), self.ndim)[local_shard_indices]

        elif not load_index:
            # use multiple workers to speed up encoding
            dataloader = torch.utils.data.DataLoader(
                corpus,
                batch_size=batch_size,
                collate_fn=RetrievalDataCollator(
                    query_max_length=self.query_max_length,
                    key_max_length=self.key_max_length,
                    key_template_fn=template_fn,
                    tokenizer=self.tokenizer,
                    packing=self.packing,
                ),
                sampler=sampler,
                pin_memory=True,
                shuffle=False,
                drop_last=False,
                num_workers=16,
            )

            offset = 0
            # NOTE: use fp16 to save disk space with neglectable performance drop
            encoded_corpus = np.zeros((len(sampler), self.ndim), dtype=np.float16)

            for step, inputs in enumerate(tqdm(dataloader, desc="Indexing")):
                # print(f"Rank {self.process_index} Token sum: {inputs['text']['attention_mask'].sum()}")
                embeddings = self._encode(inputs["text"], field="key")   # batch_size, ndim
                encoded_corpus[offset: offset + embeddings.shape[0]] = embeddings.to(device="cpu", dtype=torch.float16).numpy()
                offset += embeddings.shape[0]

                # if step > 5:
                #     break

            if save_encode:
                save_to_memmap(
                    encode_path,
                    shape=(len(corpus), self.ndim),
                    array=encoded_corpus,
                    indices=local_shard_indices,
                    accelerator=accelerator
                )
        
        torch.cuda.empty_cache()

        if skip_indexing:
            return encoded_corpus

        index = FaissIndex(self.device)
        if load_index:
            index.load(index_path)
        else:
            # NOTE: hard-coded to use inner product (because ip is the correct choice whether normalize or not)
            index.build(encoded_corpus, index_factory, metric="ip", shard_idx=shard_idx, num_shards=num_shards, local_shard_indices=local_shard_indices)

        if save_index:
            index.save(index_path)

        self._index = index

        if accelerator is not None:
            accelerator.wait_for_everyone()

        return encoded_corpus

    @torch.no_grad()
    def search(
        self, 
        inputs: Union[str, List[str], datasets.Dataset], 
        hits: int = 100, 
        template_fn: Optional[Callable] = None,
        batch_size: int = 128,
        show_progress: Optional[bool] = None,
        **kwargs
    ):
        """
        Search the corpus.

        1. search top-k in local shard
        2. merge top-k result from all shards (merge then sort then top-k)
        """
        assert self._index is not None, "Make sure there is an indexed corpus!"

        if isinstance(inputs, str):
            inputs = [inputs]

        def collate_fn(data):
            if isinstance(data[0], str):
                queries = data
            elif isinstance(data[0], dict):
                queries = [x["query"] for x in data]
            else:
                raise NotImplementedError(f"Unrecognized data type: {type(data[0])}")
            if template_fn is not None:
                queries = template_fn(queries)
            queries = self.tokenizer(queries, padding=True, truncation=True, max_length=self.query_max_length, return_tensors="pt")
            return queries

        # use dataloader to speed up tokenization
        dataloader = torch.utils.data.DataLoader(
            inputs,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            num_workers=16,
        )

        all_scores = []
        all_indices = []

        for i, batch_inputs in enumerate(tqdm(dataloader, desc="Searching", disable=not show_progress)):
            # NOTE: hard-coded to use fp16 to save memory
            embeddings = self._encode(batch_inputs, field="query").to(device="cpu", dtype=torch.float16).numpy()
            scores, indices = self._index.search(embeddings, hits)
            # offset
            indices = self._index.local_shard_indices[indices]
            scores = scores.tolist()
            indices = indices.tolist()
            all_scores.extend(scores)
            all_indices.extend(indices)

        retrieval_indices = []
        retrieval_scores = []

        for i in tqdm(range(0, len(all_scores), batch_size), desc="Merging"):
            j = i + batch_size
            scores = all_scores[i:j]
            indices = all_indices[i:j]

            if self._index.num_shards > 1:
                # gather and merge results from all processes
                # move to cpu for faster sorting and merging
                # do not use gather_for_metrics because we do not want to drop any items at last
                scores = accelerate.utils.gather_object([scores])
                indices = accelerate.utils.gather_object([indices])
            else:
                scores = [scores]
                indices = [indices]

            for batch_idx in range(len(scores[0])):
                # merge all candidates
                indice = sum([indices[i][batch_idx] for i in range(self._index.num_shards)], [])
                score = sum([scores[i][batch_idx] for i in range(self._index.num_shards)], [])
                # take care of -1s, which may be returned by faiss
                pair = sorted(zip(indice, score), key=lambda x: x[1] if x[0] >= 0 else -float('inf'), reverse=True)[:hits]
                retrieval_indices.append([x[0] for x in pair])
                retrieval_scores.append([x[1] for x in pair])

        return retrieval_indices, retrieval_scores
    
    @torch.no_grad()
    def rerank(
        self, 
        query: Union[str, List[str], Mapping], 
        key: Union[str, List[str], Mapping], 
        key_mask: Optional[torch.Tensor] = None,
    ):
        """
        Rerank key according query.

        1. compute query/key embedding
        2. compute similarity
        3. mask padded keys (if there is any)
        """
        query_embedding = self._encode(query, field="query")
        key_embedding = self._encode(key, field="key")
        key_embedding = key_embedding.view(query_embedding.shape[0], -1, query_embedding.shape[-1])   # batch_size, key_num, embedding_dim
        scores = torch.einsum("bnd,bd->bn", key_embedding, query_embedding)    # batch_size, key_num
        # mask padded candidates
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask.bool(), torch.finfo(key_embedding.dtype).min)

        scores, indices = scores.sort(dim=-1, descending=True)
        # NOTE: set the indices to -1 so that this prediction is ignored when computing metrics
        indices[scores == torch.finfo(scores.dtype).min] = -1
        return indices, scores

    def reset(self):
        """Remove index object associated with the model."""
        del self._index
        gc.collect()
        torch.cuda.empty_cache()
        self._index = None



class FaissIndex:
    def __init__(self, device) -> None:
        if isinstance(device, torch.device):
            if device.index is None:
                device = "cpu"
            else:
                device = device.index
        self.device = device

    def build(self, doc_embeddings, index_factory, metric, num_shards: int = 1, shard_idx: int = 0, local_shard_indices:Optional[np.ndarray] = None):
        if metric == "l2":
            metric = faiss.METRIC_L2
        elif metric in ["ip", "cos"]:
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise NotImplementedError(f"Metric {metric} not implemented!")
        
        index = faiss.index_factory(doc_embeddings.shape[1], index_factory, metric)
        
        if self.device != "cpu":
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            # logger.info("using fp16 on GPU...")
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.device, index, co)

        index.train(doc_embeddings)
        index.add(doc_embeddings)
        self.index = index
        self.num_shards = num_shards
        self.shard_idx = shard_idx
        self.local_shard_indices = local_shard_indices
        return index
    
    def add(self, doc_embeddings):
        self.index.add(doc_embeddings)

    def load(self, index_path):
        logger.info(f"loading index from {index_path}...")
        index = faiss.read_index(index_path)
        if self.device != "cpu":
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.device, index, co)
        self.index = index
        return index

    def save(self, index_path):
        logger.info(f"saving index at {index_path}...")
        if isinstance(self.index, faiss.GpuIndex):
            index = faiss.index_gpu_to_cpu(self.index)
        else:
            index = self.index
        faiss.write_index(index, index_path)

    def search(self, query, hits):
        return self.index.search(query, k=hits)