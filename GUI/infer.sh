export OMP_NUM_THREADS=8
echo $OMP_NUM_THREADS
cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/mllm/GUIRoboTron-Speech
eval "$('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate swift
pwd
export PYTHONPATH=$PYTHONPATH:/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/mllm/GUIRoboTron-Speech/swift

NPROC_PER_NODE=4
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR='127.0.0.1'
MASTER_PORT=12345
export MIN_PIXELS=200704
export MAX_PIXELS=937664
CUDA_VISIBLE_DEVICES=0,1,2,3 python swift/cli/infer.py \
    --model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/mllm/GUIRoboTron-Speech/output/qwen25_omni_unify_audio/v0-20250516-223506/checkpoint-10790 \
    --infer_backend pt \
    --max_batch_size 2 \
    --max_new_tokens 512 \
    --max_length 8192 \
    --system 'You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.' \
    --custom_dataset_info /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/mllm/GUIRoboTron-Speech/GUI/data/dataset_info.json \
    --val_dataset screenspot_web screenspot_desktop screenspot_mobile \
    --result_path results/0605/qwen25_omni_audio_v1_screenspot.jsonl \
