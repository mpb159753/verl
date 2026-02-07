#!/bin/bash
# run_multi_config_profiler_npu.sh
# 多配置 TransferQueue Profiler 脚本 (NPU 版本)
# 运行 5 种不同配置的 profile 日志采集
# 需要在 NPU 机器上使用单机 8 卡运行

set -x
set -e

# =====================================================
# 模型路径配置 (请填写绝对路径)
# =====================================================
MODEL_QWEN_0_5B="/path/to/Qwen2.5-0.5B-Instruct"
MODEL_QWEN_7B="/path/to/Qwen2.5-7B-Instruct"
MODEL_QWEN_14B="/path/to/Qwen2.5-14B-Instruct"

# =====================================================
# 数据集路径配置 (请填写绝对路径)
# =====================================================
DATASET_GSM8K_TRAIN="/path/to/gsm8k/train.parquet"
DATASET_GSM8K_TEST="/path/to/gsm8k/test.parquet"
DATASET_ULTRACHAT_TRAIN="/path/to/ultrachat_200k/train.parquet"
DATASET_ULTRACHAT_TEST="/path/to/ultrachat_200k/test.parquet"
DATASET_LONGBENCH_TRAIN="/path/to/longbench/train.parquet"
DATASET_LONGBENCH_TEST="/path/to/longbench/test.parquet"
DATASET_SHAREGPT4V_TRAIN="/path/to/sharegpt4v/train.parquet"
DATASET_SHAREGPT4V_TEST="/path/to/sharegpt4v/test.parquet"

# =====================================================
# Profile 配置
# =====================================================
PROFILE_BASE_OUTPUT="./outputs/multi_config_profiler_npu"
TOTAL_STEPS=20
PROFILE_STEPS='[10,15]'

# =====================================================
# 环境变量配置
# =====================================================
export TRANSFER_QUEUE_ENABLE=1
export TQ_PROFILER_ENABLED=1
export TQ_TRACE_ENABLED=1
export TORCHDYNAMO_DISABLE=1

# =====================================================
# 运行单个测试配置的函数
# =====================================================
run_profile_test() {
    local TEST_ID=$1
    local MODEL_PATH=$2
    local TRAIN_DATA=$3
    local VAL_DATA=$4
    local GLOBAL_BATCH_SIZE=$5
    local MAX_SEQ_LENGTH=$6
    local TP_SIZE=$7
    local MICRO_BATCH=$8

    local PROFILE_OUTPUT="${PROFILE_BASE_OUTPUT}/${TEST_ID}"
    
    # 计算 max_prompt_length 和 max_response_length (各为 seq_length 的一半)
    local HALF_SEQ_LENGTH=$((MAX_SEQ_LENGTH / 2))
    
    echo "======================================"
    echo "开始运行测试: ${TEST_ID}"
    echo "模型: ${MODEL_PATH}"
    echo "数据集: ${TRAIN_DATA}"
    echo "Global Batch Size: ${GLOBAL_BATCH_SIZE}"
    echo "Max Seq Length: ${MAX_SEQ_LENGTH}"
    echo "TP Size: ${TP_SIZE}"
    echo "Micro Batch: ${MICRO_BATCH}"
    echo "Profile 输出: ${PROFILE_OUTPUT}"
    echo "======================================"
    
    mkdir -p "${PROFILE_OUTPUT}"
    
    python3 -m verl.experimental.transfer_queue.main_ppo \
        --config-name='transfer_queue_ppo_trainer' \
        algorithm.adv_estimator=grpo \
        data.train_files="${TRAIN_DATA}" \
        data.val_files="${VAL_DATA}" \
        data.dataloader_num_workers=0 \
        data.max_prompt_length="${HALF_SEQ_LENGTH}" \
        data.max_response_length="${HALF_SEQ_LENGTH}" \
        data.train_batch_size="${GLOBAL_BATCH_SIZE}" \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size="${GLOBAL_BATCH_SIZE}" \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH}" \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.rollout.enforce_eager=true \
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH}" \
        actor_rollout_ref.actor.profiler.tool=npu \
        actor_rollout_ref.actor.profiler.enable=True \
        actor_rollout_ref.actor.profiler.save_path="${PROFILE_OUTPUT}" \
        actor_rollout_ref.actor.profiler.all_ranks=True \
        '++actor_rollout_ref.actor.profiler.tool_config.npu.level=level0' \
        '++actor_rollout_ref.actor.profiler.tool_config.npu.analysis=False' \
        '++actor_rollout_ref.actor.profiler.tool_config.npu.contents=[]' \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console"]' \
        trainer.project_name='multi_config_profiler' \
        trainer.experiment_name="${TEST_ID}" \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_training_steps="${TOTAL_STEPS}" \
        trainer.device=npu \
        global_profiler.tool=npu \
        global_profiler.steps="${PROFILE_STEPS}" \
        global_profiler.save_path="${PROFILE_OUTPUT}" \
        transfer_queue.enable=true

    echo "======================================"
    echo "测试 ${TEST_ID} 完成！"
    echo "Profile 输出位于: ${PROFILE_OUTPUT}"
    echo "======================================"
}

# =====================================================
# 离线分析函数
# =====================================================
run_offline_analysis() {
    local TEST_ID=$1
    local PROFILE_OUTPUT="${PROFILE_BASE_OUTPUT}/${TEST_ID}"
    
    echo "======================================"
    echo "开始离线分析: ${TEST_ID}"
    echo "======================================"
    
    # 分析第 10 step
    local STEP_10_DIR="${PROFILE_OUTPUT}/10/e2e"
    if [ -d "${STEP_10_DIR}" ]; then
        echo "分析 Step 10 数据: ${STEP_10_DIR}"
        python3 -c "
from torch_npu.profiler.profiler import analyse
analyse(profiler_path='${STEP_10_DIR}', max_process_number=8)
"
    else
        echo "警告: Step 10 目录不存在: ${STEP_10_DIR}"
    fi
    
    # 分析第 15 step
    local STEP_15_DIR="${PROFILE_OUTPUT}/15/e2e"
    if [ -d "${STEP_15_DIR}" ]; then
        echo "分析 Step 15 数据: ${STEP_15_DIR}"
        python3 -c "
from torch_npu.profiler.profiler import analyse
analyse(profiler_path='${STEP_15_DIR}', max_process_number=8)
"
    else
        echo "警告: Step 15 目录不存在: ${STEP_15_DIR}"
    fi
    
    echo "======================================"
    echo "离线分析 ${TEST_ID} 完成！"
    echo "======================================"
}

# =====================================================
# 主程序: 运行所有测试配置
# =====================================================

# 解析命令行参数
RUN_ANALYSIS=false
SKIP_PROFILE=false
TEST_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --analyse|--analysis)
            RUN_ANALYSIS=true
            shift
            ;;
        --skip-profile)
            SKIP_PROFILE=true
            shift
            ;;
        --test)
            TEST_FILTER="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--analyse] [--skip-profile] [--test T-XX]"
            exit 1
            ;;
    esac
done

# 定义所有测试配置
# 格式: TEST_ID MODEL TRAIN_DATA VAL_DATA GLOBAL_BATCH MAX_SEQ TP_SIZE MICRO_BATCH
declare -a TESTS=(
    "T-01 ${MODEL_QWEN_0_5B} ${DATASET_GSM8K_TRAIN} ${DATASET_GSM8K_TEST} 512 1024 1 8"
    "T-02 ${MODEL_QWEN_7B} ${DATASET_ULTRACHAT_TRAIN} ${DATASET_ULTRACHAT_TEST} 1024 4096 1 2"
    "T-03 ${MODEL_QWEN_7B} ${DATASET_LONGBENCH_TRAIN} ${DATASET_LONGBENCH_TEST} 2048 8192 1 1"
    "T-04 ${MODEL_QWEN_14B} ${DATASET_ULTRACHAT_TRAIN} ${DATASET_ULTRACHAT_TEST} 1024 4096 8 1"
    "T-05 ${MODEL_QWEN_7B} ${DATASET_SHAREGPT4V_TRAIN} ${DATASET_SHAREGPT4V_TEST} 64 65536 1 1"
)

# 运行 profiling
if [ "${SKIP_PROFILE}" = false ]; then
    for test_config in "${TESTS[@]}"; do
        read -r TEST_ID MODEL TRAIN VAL BATCH SEQ TP MICRO <<< "${test_config}"
        
        # 如果指定了测试过滤器，只运行匹配的测试
        if [ -n "${TEST_FILTER}" ] && [ "${TEST_ID}" != "${TEST_FILTER}" ]; then
            continue
        fi
        
        run_profile_test "${TEST_ID}" "${MODEL}" "${TRAIN}" "${VAL}" "${BATCH}" "${SEQ}" "${TP}" "${MICRO}"
    done
fi

# 运行离线分析
if [ "${RUN_ANALYSIS}" = true ]; then
    for test_config in "${TESTS[@]}"; do
        read -r TEST_ID _ <<< "${test_config}"
        
        # 如果指定了测试过滤器，只分析匹配的测试
        if [ -n "${TEST_FILTER}" ] && [ "${TEST_ID}" != "${TEST_FILTER}" ]; then
            continue
        fi
        
        run_offline_analysis "${TEST_ID}"
    done
fi

echo ""
echo "=============================================="
echo "所有任务完成！"
echo ""
echo "Profile 输出目录: ${PROFILE_BASE_OUTPUT}"
echo ""
echo "测试配置汇总:"
echo "  T-01: Qwen2.5-0.5B-Instruct + GSM8K (Batch=512, Seq=1024, TP=1, Micro=8)"
echo "  T-02: Qwen2.5-7B-Instruct + UltraChat_200k (Batch=1024, Seq=4096, TP=1, Micro=2)"
echo "  T-03: Qwen2.5-7B-Instruct + LongBench (Batch=2048, Seq=8192, TP=1, Micro=1)"
echo "  T-04: Qwen2.5-14B-Instruct + UltraChat_200k (Batch=1024, Seq=4096, TP=8, Micro=1)"
echo "  T-05: Qwen2.5-7B-Instruct + ShareGPT4V (Batch=64, Seq=65536, TP=1, Micro=1)"
echo ""
echo "用法示例:"
echo "  运行所有测试:         ./run_multi_config_profiler_npu.sh"
echo "  运行单个测试:         ./run_multi_config_profiler_npu.sh --test T-01"
echo "  仅运行离线分析:       ./run_multi_config_profiler_npu.sh --skip-profile --analyse"
echo "  运行测试后进行分析:   ./run_multi_config_profiler_npu.sh --analyse"
echo "=============================================="
