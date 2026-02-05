#!/bin/bash
# run_tq_profiler_validation_npu.sh
# 验证 TransferQueue Profiler 标记功能的脚本 (NPU 版本)
# 使用 gsm8k 数据 + qwen 0.5b 模型 + vllm rollout
# 需要在 NPU 机器上运行

set -x

PROJECT_DIR="$(pwd)"

# 启用 TQ Profiler 标记
export TQ_PROFILER_ENABLED=1
export TQ_TRACE_ENABLED=1

python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/verl/trainer/config" \
    --config-name='ppo_trainer' \
    algorithm.adv_estimator=grpo \
    data.dataloader_num_workers=0 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.train_batch_size=32 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='tq_profiler_validation' \
    trainer.experiment_name='qwen0.5b-gsm8k-tq-profiler-npu' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.device=npu \
    data.train_files=$PROJECT_DIR/data/gsm8k/train.parquet \
    data.val_files=$PROJECT_DIR/data/gsm8k/test.parquet \
    global_profiler.tool=npu \
    global_profiler.steps='[0]' \
    global_profiler.save_path=./outputs/tq_profiler_validation_npu \
    transfer_queue.enable=True \
    "$@"

echo "====================================="
echo "验证完成！"
echo "请使用 MindStudio Insight 查看 profiling 结果："
echo "  ./outputs/tq_profiler_validation_npu/"
echo ""
echo "搜索 TQ_GET 和 TQ_PUT 标记验证是否生效"
echo "====================================="
