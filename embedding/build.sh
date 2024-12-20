#!/bin/bash

shopt -s expand_aliases
shopt -s extglob


# install screen
if [[ $(type -t conda) == "" ]]; then
sudo apt-get install screen -y
fi


export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118


########## Setup Conda ##########

if [ ! -d "/opt/tiger/miniconda" ]; then
bash /mnt/bn/search-douyin-rank-yg/all_data_from_lf/peitian_data/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/tiger/miniconda
fi

if [[ $(type -t conda) == "" ]]; then

source ~/.bashrc
/opt/tiger/miniconda/bin/conda init
source ~/.bashrc

# echo """
# channels:
#   - defaults
# show_channel_urls: true
# default_channels:
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
# custom_channels:
#   conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
#   pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
# """ > ~/.condarc

fi


if [ ! -d "/opt/tiger/miniconda/envs/llm" ]; then
/opt/tiger/miniconda/bin/conda create -n llm python=3.10 -y
fi


########## Setup aliases ##########
source ~/.bashrc

if [[ $(type -t pta) == "" ]]; then 
echo """
# * Peitian's Alias Start
alias pts='screen -r -d pt'
pta () {
    conda activate /opt/tiger/miniconda/envs/\$1
}
pte () {
    if [[ \$1 ]]
        then
    export CUDA_VISIBLE_DEVICES=\$1
        else
    unset CUDA_VISIBLE_DEVICES
        fi
}
ptgit () {
  cd /opt/tiger/DouyinSearchEmb
  git pull origin peitian
}
# * Peitian's Alias End
""" >> ~/.bashrc
fi


########## Setup python environment ##########

source /opt/tiger/miniconda/bin/activate /opt/tiger/miniconda/envs/llm

if [[ $(python -c "import faiss; print(faiss.__version__)") != "1.7.4" ]]; then

if [[ $(type -t nvidia-smi) == "" ]]; then

python --version

###### CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.44.2 deepspeed accelerate datasets peft seaborn rouge fuzzywuzzy jieba python-Levenshtein ipykernel wandb langchain-openai langchain fastapi loguru nvitop mteb openpyxl sentencepiece notebook ubelt
/opt/tiger/miniconda/bin/conda install -c conda-forge faiss-cpu=1.7.4 -y

else

###### GPU
pip install torch torchvision torchaudio
pip install transformers==4.44.2 deepspeed accelerate datasets peft seaborn rouge fuzzywuzzy jieba python-Levenshtein ipykernel wandb langchain-openai langchain fastapi loguru nvitop mteb openpyxl sentencepiece notebook ubelt
# pip install flash-attn==2.5.8 --no-build-isolation
pip install /mnt/bn/search-douyin-rank-yg/all_data_from_lf/peitian_data/flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
/opt/tiger/miniconda/bin/conda install -c conda-forge faiss-gpu=1.7.4 -y

fi
fi