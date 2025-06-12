#!/bin/bash
WORLD_SIZE=4
eval "$('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate swift
echo "env activated"
cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/huangjing/code/mllm/UI/aguvis
RES_PATH='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/mllm/GUIRoboTron-Speech/results/0605/qwen25_omni_audio_v1_screenspot.jsonl'
screenspot_imgs="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/common/datasets/code/data/screen/ScreenSpot/eval/images"
screenspot_audios="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/common/datasets/code/data/screen/ScreenSpot/eval/audios"
screenspot_test="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/common/datasets/code/data/screen/ScreenSpot/eval/metadata_audio"
CUDA_VISIBLE_DEVICES=0 python ./eval/screenspot_test_aguivs_qwen25omni_eval.py --result_path $RES_PATH \
    --screenspot_imgs $screenspot_imgs \
    --screenspot_audios $screenspot_audios \
    --screenspot_test $screenspot_test \
    --task all