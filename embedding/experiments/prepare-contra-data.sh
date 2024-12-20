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


interval=100

base_dir=/mnt/bn/search-douyin-rank-yg/all_data_from_lf/text_embedding_data/query_doc_info_sample_order_1028_v4

for i in $(seq 0 $interval 1900); do

j=$((i + $interval))

python -m main.prepare_contra_data --start_file_idx $i --end_file_idx $j --base_dir $base_dir --output_dir ${base_dir}_contra_qd &

done


# python -m main.prepare_contra_data --output_dir /mnt/bn/search-douyin-rank-yg/all_data_from_lf/text_embedding_data/query_doc_info_sample_order_1028_v4_contra_qd_dd_refine_0.1 --end_file_idx 100
# python -m main.prepare_contra_data --output_dir /mnt/bn/search-douyin-rank-yg/all_data_from_lf/text_embedding_data/query_doc_info_sample_order_1028_v4_contra_qd_dd_refine_0.1 --start_file_idx 100 --end_file_idx 200 --doc_as_query_portion 1