#!/bin/bash

# 直接指定环境下的 python 路径
ENV_PYTHON="/opt/miniconda3/envs/dl_env/bin/python"
BASE_PATH="comp/HyLiOSR"

SCRIPTS=(
    "chikusei/model_chikusei.py"
    "Indian_pines/model_Indian_pines.py"
    "PaviaU/model_PaviaU.py"
    "Salina/model_Salina.py"
    "whu_hanchuan/model_whu_hanchuan.py"
    "whu_honghu/model_whu_honghu.py"
    "whu_longkou/model_whu_longkou.py"
)

for script in "${SCRIPTS[@]}"; do
    full_path="$BASE_PATH/$script"
    script_name=$(basename "$script" .py)
    session_name="woAtt_$script_name"

    echo "Launching $full_path in tmux session: $session_name"

    tmux new-session -d -s "$session_name" "$ENV_PYTHON $full_path"
done