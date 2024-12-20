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


model=/mnt/bn/search-douyin-rank-yg/all_data_from_lf/peitian_data/data/outputs/qwen_2_0.5b-listwise-d64_2layer-v4-g2-bs256-fullparam-crossneg_8-temp0.02/checkpoint-6699

# output_dir=/mnt/bn/search-douyin-rank-yg/all_data_from_lf/infer_data/search_l1_doc_info_text_1102_shard1_subset_for_debug
output_dir=/mnt/bn/search-douyin-rank-yg/all_data_from_lf/infer_data/search_l1_doc_info_text_1102_test_for_debug

eval_data=$output_dir/queries.json
corpus=$output_dir

# 1. 直接加载刷库产生的embedding
# torchrun $DDP -m main.eval_retrieval --eval_data $eval_data --corpus $corpus --model_name_or_path $model --mrl_dims 64 --mrl_2layer_proj True --metrics collate_key --hits 10 --output_dir $output_dir --save_name listwise --load_encode

# 2. 不加载，重新编码
torchrun $DDP -m main.eval_retrieval --eval_data $eval_data --corpus $corpus --model_name_or_path $model --mrl_dims 64 --mrl_2layer_proj True --metrics collate_key --hits 10 --output_dir $output_dir
