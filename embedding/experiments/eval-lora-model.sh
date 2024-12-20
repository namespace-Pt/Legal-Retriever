#!/bin/bash

########### make bash recognize aliases ##########
shopt -s expand_aliases
shopt -s extglob

source ~/.bashrc

# set huggingface mirror
# export HF_ENDPOINT=https://hf-mirror.com

# NOTE: set this to avoid OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# activate environment
source /opt/tiger/miniconda/bin/activate /opt/tiger/miniconda/envs/llm/

python3 --version

cd /opt/tiger/DouyinSearchEmb

NODE_RANK="$ARNOLD_ID"
NPROC_PER_NODE=$ARNOLD_WORKER_GPU
NNODES=$ARNOLD_WORKER_NUM
MASTER_ADDR=$ARNOLD_WORKER_0_HOST
MASTER_PORT=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)

DDP="--nproc_per_node $NPROC_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $NODE_RANK"

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"


lora=$1
model=${2:-"/mnt/bn/search-douyin-rank-yg/all_data_from_lf/llm_models/Qwen2-0.5B.with_fasttokenizer"}
eval_data=/mnt/bn/search-douyin-rank-yg/all_data_from_lf/text_embedding_data/query_doc_info_sample_order_1102_eval_v2


torchrun $DDP -m main.eval_retrieval --eval_data $eval_data --corpus $eval_data --model_name_or_path $model --lora $lora --mrl_dims 64 --mrl_2layer_proj True

torchrun $DDP -m main.eval_ctr --eval_data $eval_data --model_name_or_path $model --lora $lora --mrl_dims 64 --mrl_2layer_proj True


if [ $NODE_RANK = 0 ]; then
cat data/results/retrieval/metrics.log >> /mnt/bn/search-douyin-rank-yg/all_data_from_lf/peitian_data/data/results/retrieval/metrics.log
cat data/results/ctr/metrics.log >> /mnt/bn/search-douyin-rank-yg/all_data_from_lf/peitian_data/data/results/ctr/metrics.log
fi