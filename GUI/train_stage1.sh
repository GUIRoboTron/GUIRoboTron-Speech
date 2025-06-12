rm -r ~/.cache
export OMP_NUM_THREADS=8
echo $OMP_NUM_THREADS
cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/mllm/GUIRoboTron-Speech
eval "$('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate swift
pwd
export PYTHONPATH=$PYTHONPATH:/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/mllm/GUIRoboTron-Speech/


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
    --gradient_accumulation_steps 8 \
    --eval_steps 10000 \
    --save_steps 2000 \
    --save_total_limit 15 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output/qwen25_omni_audio_1160k \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 64 \
    --system 'You are an agent receiving audio commands for screen grounding. Identify the commands in the audio input and locate the corresponding position coordinates in the image.' \
    --custom_dataset_info /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/code/mllm/GUIRoboTron-Speech/GUI/data/dataset_info.json \
    --dataset ui_refexp guienv webui350k omniact_fix ricoig16k widget_captioning ricosca en_aw en_uibert win macos amex2 win macos macos macos macos \
    --deepspeed zero2
# 
# https://blog.csdn.net/Chrsitina_S/article/details/134921892 zero3
# --local_repo_path
# freeze_aligner=True,
# freeze_llm=False,
# freeze_parameters=['thinker.audio_tower', 'thinker.visual', 'talker', 'token2wav'],
# freeze_parameters_ratio=0.0,
# freeze_vit=True,

# 自定义dataset_info路径
# --custom_dataset_info xxx.json0