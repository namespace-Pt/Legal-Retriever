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


eval_data=/mnt/bn/search-douyin-rank-yg/all_data_from_lf/text_embedding_data/query_doc_info_sample_order_1102_eval_v1[100000]

for model in /mnt/bn/search-douyin-rank-yg/all_data_from_lf/peitian_data/data/outputs/qwen_2_0.5b-listwise-d64_2layer-v4-g2-bs256-fullparam-crossneg_8-temp0.2/checkpoint-6699 /mnt/bn/search-douyin-rank-yg/all_data_from_lf/peitian_data/data/outputs/qwen_2_0.5b-listwise-d64_2layer-v4-g2-bs256-fullparam-crossneg_8-temp0.02/checkpoint-6699
do

torchrun --nproc_per_node 8 --master_port 12345 -m main.eval_doc_sim --eval_data $eval_data --model_name_or_path $model --mrl_dims 64 --mrl_2layer_proj True

done