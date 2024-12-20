#!/bin/bash

########### make bash recognize aliases ##########
shopt -s expand_aliases
shopt -s extglob

########### build environment if necessary ##########
if [ ! -f /opt/tiger/miniconda/bin/activate ]; then
bash /opt/tiger/DouyinSearchEmb/build.sh
fi


# set huggingface mirror
# export HF_ENDPOINT=https://hf-mirror.com

# NOTE: set this to avoid OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# activate environment
source /opt/tiger/miniconda/bin/activate /opt/tiger/miniconda/envs/llm/

python3 --version

cd /opt/tiger/DouyinSearchEmb

# 关闭命令执行的详细输出模式（如果之前开启过）
set +x

# NCCL_DEBUG=INFO
# NCCL_P2P_LEVEL=NVL

NODE_RANK="$ARNOLD_ID"
NPROC_PER_NODE=$ARNOLD_WORKER_GPU
NNODES=${1:-$ARNOLD_WORKER_NUM}
MASTER_ADDR=$ARNOLD_WORKER_0_HOST
MASTER_PORT=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)

DDP="--nproc_per_node $NPROC_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $NODE_RANK"

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"


TRAIN_DATA_FOLDER=/mnt/bn/search-douyin-rank-yg/all_data_from_lf/text_embedding_data/query_doc_info_sample_order_1028_v4_contra_qd
TOKENIZED_TRAIN_DATA=$TRAIN_DATA_FOLDER/qwen2-v1-512-listwise-g2-tokenized

if [ ! -d $TOKENIZED_TRAIN_DATA ]; then

torchrun $DDP -m main.prepare_train_data \
--model_name_or_path /mnt/bn/search-douyin-rank-yg/all_data_from_lf/llm_models/qwen-2-1.5b \
--output_dir $TOKENIZED_TRAIN_DATA \
--train_data $TRAIN_DATA_FOLDER/*query_pos_neg \
--padding_side left \
--query_max_length 512 \
--key_max_length 512 \
--query_template v1 \
--key_template v1 \
--train_method listwise \
--train_group_size 2 \
--select_pos first \
--select_neg random

fi


output_name=qwen_2_0.5b-pointwise-d64_2layer-v4-g2-bs256-fullparam-temp0.2

torchrun $DDP -m main.train --deepspeed data/config/qwen2/zero2.json \
--output_dir /mnt/bn/search-douyin-rank-yg/all_data_from_lf/peitian_data/data/outputs/$output_name \
--model_name_or_path /mnt/bn/search-douyin-rank-yg/all_data_from_lf/llm_models/Qwen2-0.5B.with_fasttokenizer/ \
--train_data $TOKENIZED_TRAIN_DATA \
--skip_preprocess \
--padding_side left \
--dtype float16 \
--attn_impl flash_attention_2 \
--pooling_method last \
--batch_size 128 \
--train_method both \
--train_group_size 2 \
--query_max_length 512 \
--key_max_length 512 \
--packing \
--mrl_dims 64 \
--mrl_2layer_proj True \
--query_template v1 \
--key_template v1 \
--select_pos first \
--select_neg random \
--distill_weight 0 \
--pointwise_weight 1 \
--listwise_weight 0 \
--use_inbatch_neg \
--use_cross_device_neg -1 \
--filter_inbatch_neg \
--pointwise_temp 0.2 \
--per_device_train_batch_size 256 \
--num_train_epochs 1 \
--learning_rate 1e-4 \
--lr_scheduler_type cosine_with_min_lr \
--warmup_steps 0 \
--weight_decay 0.01 \
--fp16 \
--gradient_checkpointing \
--gradient_checkpointing_kwargs '{"use_reentrant": false}' \
--save_strategy steps \
--save_steps 0.5 \
--eval_strategy no \
--logging_steps 1 \
--save_only_model





for lora in /mnt/bn/search-douyin-rank-yg/all_data_from_lf/peitian_data/data/outputs/$output_name/checkpoint-*
do

for eval_data in /mnt/bn/search-douyin-rank-yg/all_data_from_lf/text_embedding_data/query_doc_info_sample_order_1102_eval_v2
do


torchrun $DDP -m main.eval_retrieval --eval_data $eval_data --corpus $eval_data --model_name_or_path $lora --mrl_dims 64 --mrl_2layer_proj True

torchrun $DDP -m main.eval_ctr --eval_data $eval_data --model_name_or_path $lora --mrl_dims 64 --mrl_2layer_proj True

done

done


if [ $NODE_RANK = 0 ]; then
cat data/results/retrieval/metrics.log >> /mnt/bn/search-douyin-rank-yg/all_data_from_lf/peitian_data/data/results/retrieval/metrics.log
cat data/results/ctr/metrics.log >> /mnt/bn/search-douyin-rank-yg/all_data_from_lf/peitian_data/data/results/ctr/metrics.log
fi