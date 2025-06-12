rm -r ~/.cache
export OMP_NUM_THREADS=8
echo $OMP_NUM_THREADS
cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/mllm/GUIRoboTron-Speech
eval "$('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate swift
pwd
export PYTHONPATH=$PYTHONPATH:/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/mllm/GUIRoboTron-Speech/
export http_proxy=http://10.229.18.23:3128
export https_proxy=http://10.229.18.23:3128

JOB_ARGS=($(python GUI/get_job_args_lc.py))
echo "JOB_ARGS: ${JOB_ARGS[@]}"
export FORCE_TORCHRUN=1
export NNODES="${JOB_ARGS[0]}"
export GPUS_PER_NODE="${JOB_ARGS[1]}"
export MASTER_ADDR="${JOB_ARGS[2]}"
export MASTER_PORT="${JOB_ARGS[3]}"
export NODE_RANK="${JOB_ARGS[4]}"
export MIN_PIXELS=200704
export MAX_PIXELS=937664
torchrun --nproc_per_node="${GPUS_PER_NODE}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}"  swift/cli/sft.py \
    --model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/models/Qwen2.5-Omni-7B \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --target_modules all-linear \
    --freeze_vit true \
    --split_dataset_ratio 0.0 \
    --gradient_accumulation_steps 4 \
    --eval_steps 50000 \
    --save_steps 5000 \
    --save_total_limit 15 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output/qwen25_omni_stage2_d1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 64 \
    --custom_dataset_info /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/mllm/GUIRoboTron-Speech/GUI/data2/dataset_info.json \
    --dataset aitw_l1 amex_l1 android_control_l1 coat_l1 odyssey_l1 guiact_l1 mind2web_l1 miniwob_l1 \
    aitw_l2 amex_l2 coat_l2 odyssey_l2 guiact_l2 mind2web_l2 miniwob_l2 \
    aitw_l3 amex_l3 coat_l3 odyssey_l3 guiact_l3 mind2web_l3 miniwob_l3 \
    --deepspeed zero2
