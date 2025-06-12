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
CUDA_VISIBLE_DEVICES=6,7 python swift/cli/infer.py \
    --model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/mllm/GUIRoboTron-Speech/output/qwen25_omni_stage2_d1/v0-20250509-144417/checkpoint-7709 \
    --infer_backend pt \
    --max_batch_size 2 \
    --max_new_tokens 512 \
    --max_length 8192 \
    --custom_dataset_info /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/mllm/GUIRoboTron-Speech/GUI/data/dataset_info.json \
    --val_dataset ac_ins_dev_text \
    --result_path results/stage2/omni_stage2_d1_ac_ins_dev_text.jsonl \


