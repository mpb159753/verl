#!/bin/bash
# run_multi_config_main_ppo_npu.sh
# 多配置 Main PPO Profiler 脚本 (NPU 版本)
# 运行多种不同配置的 profile 日志采集 (不使用 TransferQueue)
# 覆盖 tiny → large 不同数据体量，以及 2/4/8 机集群规模
#
# 测试矩阵说明:
#   S 系列 (S-01~S-05): 固定 2 机 16 卡, 覆盖 tiny→large 不同 put 体积
#   N 系列 (N-xx-yy):   固定 medium/large 级别, 对比 2/4/8 机跨机通信开销
#     - N-M-02/04/08: 14B medium (~1.5GB) 在 2/4/8 机
#     - N-L-04/08:    14B large  (~5.9GB) 在 4/8 机

set -x
set -e

# =====================================================
# 路径配置 (相对于代码根目录)
# =====================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 模型路径
MODEL_QWEN_0_5B="${SCRIPT_DIR}/model/Qwen2.5-0.5B-Instruct"
MODEL_QWEN_7B="${SCRIPT_DIR}/model/Qwen2.5-7B-Instruct"
MODEL_QWEN_14B="${SCRIPT_DIR}/model/Qwen2.5-14B-Instruct"

# 数据集路径
DATASET_GSM8K_TRAIN="${SCRIPT_DIR}/dataset/gsm8k/train.parquet"
DATASET_GSM8K_TEST="${SCRIPT_DIR}/dataset/gsm8k/test.parquet"
DATASET_MATH_TRAIN="${SCRIPT_DIR}/dataset/math/train.parquet"
DATASET_MATH_TEST="${SCRIPT_DIR}/dataset/math/test.parquet"

# 合并数据集 (GSM8K + MATH, ~15k 条)
DATASET_COMBINED_TRAIN="[${DATASET_GSM8K_TRAIN},${DATASET_MATH_TRAIN}]"
DATASET_COMBINED_TEST="[${DATASET_GSM8K_TEST},${DATASET_MATH_TEST}]"

# =====================================================
# Profile 配置
# =====================================================
PROFILE_BASE_OUTPUT="./outputs/multi_config_main_ppo_npu"
TOTAL_STEPS=6
PROFILE_STEPS='[2,4]'

# =====================================================
# 日志配置
# =====================================================
LOG_DIR="${LOG_DIR:-./logs/multi_config_main_ppo_npu}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
echo "日志将保存到: ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

# =====================================================
# 环境变量配置
# =====================================================
export TORCHDYNAMO_DISABLE=1

# =====================================================
# 资源清理函数
# =====================================================
cleanup_resources() {
    echo "======================================"
    echo "清理残留资源..."
    echo "======================================"
    
    # 停止 Ray (忽略错误)
    echo "停止 Ray..."
    ray stop --force 2>/dev/null || true
    
    # 终止可能残留的训练进程
    echo "终止残留 Python 进程..."
    pkill -9 -f "verl.trainer.main_ppo" 2>/dev/null || true
    pkill -9 -f "python.*main_ppo" 2>/dev/null || true
    pkill -9 -f "WorkerDict" 2>/dev/null || true
    pkill -9 -f "TaskRunner" 2>/dev/null || true
    
    # 等待端口释放
    echo "等待端口释放..."
    sleep 5
    
    # 清理共享内存 (可选)
    echo "清理共享内存..."
    rm -rf /dev/shm/ray_* 2>/dev/null || true
    rm -rf /dev/shm/plasma_* 2>/dev/null || true
    
    echo "资源清理完成"
}

# 设置退出时的清理钩子
cleanup_on_exit() {
    echo ""
    echo "======================================"
    echo "脚本退出，执行清理..."
    echo "======================================"
    # 停止 Ray
    ray stop --force 2>/dev/null || true
    echo "清理完成"
}
trap cleanup_on_exit EXIT

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
    local NNODES=$9
    local ROLLOUT_N=${10}
    local WITH_STACK_FLAG=${11}
    
    # 根据 WITH_STACK_FLAG 设置 contents 参数
    local PROFILE_CONTENTS='[]'
    if [ "${WITH_STACK_FLAG}" = "true" ]; then
        PROFILE_CONTENTS='["stack", "module", "npu", "cpu"]'
    fi

    local PROFILE_OUTPUT="${PROFILE_BASE_OUTPUT}/${TEST_ID}"
    
    # 计算 max_prompt_length 和 max_response_length (各为 seq_length 的一半)
    local HALF_SEQ_LENGTH=$((MAX_SEQ_LENGTH / 2))
    
    # 计算预估 put 大小 (effective_batch × seqlen × 44B)
    local EFFECTIVE_BATCH=$((GLOBAL_BATCH_SIZE * ROLLOUT_N))
    local PUT_ELEMENTS=$((EFFECTIVE_BATCH * MAX_SEQ_LENGTH))
    local PUT_BYTES=$((PUT_ELEMENTS * 44))
    local PUT_MB=$((PUT_BYTES / 1024 / 1024))
    
    echo "======================================"
    echo "开始运行测试: ${TEST_ID}"
    echo "模型: ${MODEL_PATH}"
    echo "数据集: ${TRAIN_DATA}"
    echo "Global Batch Size: ${GLOBAL_BATCH_SIZE}"
    echo "Max Seq Length: ${MAX_SEQ_LENGTH}"
    echo "TP Size: ${TP_SIZE}"
    echo "Micro Batch: ${MICRO_BATCH}"
    echo "Nodes: ${NNODES}"
    echo "Rollout N: ${ROLLOUT_N}"
    echo "With Stack: ${WITH_STACK_FLAG}"
    echo "Effective Batch (batch×n): ${EFFECTIVE_BATCH}"
    echo "估算 gen_sequences put 大小: ~${PUT_MB}MB"
    echo "Profile 输出: ${PROFILE_OUTPUT}"
    echo "======================================"
    
    mkdir -p "${PROFILE_OUTPUT}"
    
    python3 -m verl.trainer.main_ppo \
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
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.max_model_len="${MAX_SEQ_LENGTH}" \
        actor_rollout_ref.rollout.enforce_eager=true \
        actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH}" \
        actor_rollout_ref.actor.profiler.tool=npu \
        actor_rollout_ref.actor.profiler.enable=True \
        actor_rollout_ref.actor.profiler.save_path="${PROFILE_OUTPUT}" \
        actor_rollout_ref.actor.profiler.all_ranks=True \
        '++actor_rollout_ref.actor.profiler.tool_config.npu.level=level0' \
        '++actor_rollout_ref.actor.profiler.tool_config.npu.analysis=False' \
        "++actor_rollout_ref.actor.profiler.tool_config.npu.contents=${PROFILE_CONTENTS}" \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console"]' \
        trainer.project_name='multi_config_profiler_main' \
        trainer.experiment_name="${TEST_ID}" \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes="${NNODES}" \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_training_steps="${TOTAL_STEPS}" \
        trainer.device=npu \
        global_profiler.tool=npu \
        global_profiler.steps="${PROFILE_STEPS}" \
        global_profiler.save_path="${PROFILE_OUTPUT}" \
        transfer_queue.enable=false

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
    
    # 分析 profile steps 对应的目录
    for STEP in 2 4; do
        local STEP_DIR="${PROFILE_OUTPUT}/${STEP}/e2e"
        if [ -d "${STEP_DIR}" ]; then
            echo "分析 Step ${STEP} 数据: ${STEP_DIR}"
            python3 -c "
from torch_npu.profiler.profiler import analyse
analyse(profiler_path='${STEP_DIR}', export_type='db')
"
        else
            echo "警告: Step ${STEP} 目录不存在: ${STEP_DIR}"
        fi
    done
    
    echo "======================================"
    echo "离线分析 ${TEST_ID} 完成！"
    echo "======================================"
}

# =====================================================
# 分析后清理原始数据函数
# =====================================================
cleanup_profile_raw_data() {
    local TEST_ID=$1
    local PROFILE_OUTPUT="${PROFILE_BASE_OUTPUT}/${TEST_ID}"
    
    echo "======================================"
    echo "清理原始 profile 数据: ${TEST_ID}"
    echo "======================================"
    
    for STEP in 2 4; do
        local STEP_DIR="${PROFILE_OUTPUT}/${STEP}/e2e"
        if [ -d "${STEP_DIR}" ]; then
            # 确认 ascend_profiler_output 存在后再清理
            if [ -d "${STEP_DIR}/ascend_profiler_output" ]; then
                local BEFORE_SIZE=$(du -sh "${STEP_DIR}" 2>/dev/null | cut -f1)
                
                # 删除 ascend_profiler_output 以外的所有内容
                find "${STEP_DIR}" -mindepth 1 -maxdepth 1 \
                    ! -name 'ascend_profiler_output' \
                    -exec rm -rf {} +
                
                local AFTER_SIZE=$(du -sh "${STEP_DIR}" 2>/dev/null | cut -f1)
                echo "Step ${STEP}: ${BEFORE_SIZE} → ${AFTER_SIZE} (保留 ascend_profiler_output)"
            else
                echo "警告: Step ${STEP} 未找到 ascend_profiler_output, 跳过清理"
            fi
        fi
    done
    
    echo "清理完成: ${TEST_ID}"
}

# =====================================================
# 主程序: 运行所有测试配置
# =====================================================

# 解析命令行参数
RUN_ANALYSIS=false
SKIP_PROFILE=false
TEST_FILTER=""
WITH_STACK=false
NODE_FILTER=""

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
        --nnodes)
            NODE_FILTER="$2"
            shift 2
            ;;
        --with-stack)
            WITH_STACK=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--analyse] [--skip-profile] [--test T-XX] [--nnodes N] [--with-stack]"
            exit 1
            ;;
    esac
done

# =====================================================
# 定义所有测试配置
# 格式: TEST_ID MODEL TRAIN_DATA VAL_DATA BATCH SEQLEN TP MICRO NNODES ROLLOUT_N
# =====================================================

# --- S 系列: 覆盖不同 put 体积 (固定 2 机 16 卡) ---
#
# gen_sequences put 大小 ≈ batch × n × seqlen × 44B
#
# S-01: 0.5B, batch=64,   n=1, seq=1024  → 64×1×1024×44     ≈   2.7MB (tiny)
# S-02: 7B,   batch=256,  n=4, seq=2048  → 256×4×2048×44     ≈  88MB  (small)
# S-03: 7B,   batch=512,  n=4, seq=4096  → 512×4×4096×44     ≈ 370MB  (medium-low)
# S-04: 14B,  batch=1024, n=4, seq=8192  → 1024×4×8192×44    ≈ 1.5GB  (medium)
# S-05: 14B,  batch=2048, n=8, seq=8192  → 2048×8×8192×44    ≈ 5.9GB  (large)

declare -a TESTS=(
    "S-01 ${MODEL_QWEN_0_5B} ${DATASET_GSM8K_TRAIN} ${DATASET_GSM8K_TEST} 64 1024 1 8 2 1"
    "S-02 ${MODEL_QWEN_7B} ${DATASET_GSM8K_TRAIN} ${DATASET_GSM8K_TEST} 256 2048 1 4 2 4"
    "S-03 ${MODEL_QWEN_7B} ${DATASET_COMBINED_TRAIN} ${DATASET_COMBINED_TEST} 512 4096 1 2 2 4"
    "S-04 ${MODEL_QWEN_14B} ${DATASET_COMBINED_TRAIN} ${DATASET_COMBINED_TEST} 1024 8192 2 1 2 4"
    "S-05 ${MODEL_QWEN_14B} ${DATASET_COMBINED_TRAIN} ${DATASET_COMBINED_TEST} 2048 8192 2 1 2 8"
)

# --- N 系列: 跨机规模对照 (固定 put 体积) ---
#
# N-M 系列: medium (~1.5GB), 14B, batch=1024, n=4, seq=8192
# N-L 系列: large  (~5.9GB), 14B, batch=2048, n=8, seq=8192

declare -a TESTS_N=(
    "N-M-02 ${MODEL_QWEN_14B} ${DATASET_COMBINED_TRAIN} ${DATASET_COMBINED_TEST} 1024 8192 2 1 2 4"
    "N-M-04 ${MODEL_QWEN_14B} ${DATASET_COMBINED_TRAIN} ${DATASET_COMBINED_TEST} 1024 8192 2 1 4 4"
    "N-M-08 ${MODEL_QWEN_14B} ${DATASET_COMBINED_TRAIN} ${DATASET_COMBINED_TEST} 1024 8192 2 1 8 4"
    "N-L-04 ${MODEL_QWEN_14B} ${DATASET_COMBINED_TRAIN} ${DATASET_COMBINED_TEST} 2048 8192 2 1 4 8"
    "N-L-08 ${MODEL_QWEN_14B} ${DATASET_COMBINED_TRAIN} ${DATASET_COMBINED_TEST} 2048 8192 2 1 8 8"
)

# 合并所有配置
ALL_TESTS=("${TESTS[@]}" "${TESTS_N[@]}")

# =====================================================
# 运行 profiling
# =====================================================
if [ "${SKIP_PROFILE}" = false ]; then
    for test_config in "${ALL_TESTS[@]}"; do
        read -r TEST_ID MODEL TRAIN VAL BATCH SEQ TP MICRO NODES ROLLN <<< "${test_config}"
        
        # 如果指定了测试过滤器，只运行匹配的测试
        if [ -n "${TEST_FILTER}" ] && [ "${TEST_ID}" != "${TEST_FILTER}" ]; then
            continue
        fi
        
        # 如果指定了节点过滤器，只运行匹配节点数的测试
        if [ -n "${NODE_FILTER}" ] && [ "${NODES}" != "${NODE_FILTER}" ]; then
            continue
        fi
        
        # 每次测试前清理资源
        cleanup_resources
        
        run_profile_test "${TEST_ID}" "${MODEL}" "${TRAIN}" "${VAL}" "${BATCH}" "${SEQ}" "${TP}" "${MICRO}" "${NODES}" "${ROLLN}" "${WITH_STACK}"
    done
fi

# =====================================================
# 运行离线分析
# =====================================================
if [ "${RUN_ANALYSIS}" = true ]; then
    for test_config in "${ALL_TESTS[@]}"; do
        read -r TEST_ID _ <<< "${test_config}"
        
        # 如果指定了测试过滤器，只分析匹配的测试
        if [ -n "${TEST_FILTER}" ] && [ "${TEST_ID}" != "${TEST_FILTER}" ]; then
            continue
        fi
        
        run_offline_analysis "${TEST_ID}"
        
        # 分析完成后清理原始数据
        cleanup_profile_raw_data "${TEST_ID}"
    done
fi

echo ""
echo "=============================================="
echo "所有任务完成！"
echo ""
echo "Profile 输出目录: ${PROFILE_BASE_OUTPUT}"
echo ""
echo "测试配置汇总 (gen_sequences put 大小 ≈ batch × n × seqlen × 44B):"
echo ""
echo "  ── S 系列: 覆盖不同 put 体积 (固定 2 机 16 卡) ──"
echo "  S-01: 0.5B + GSM8K        (Batch=64,   Seq=1024, TP=1, n=1, Micro=8) → ~2.7MB  (tiny)"
echo "  S-02: 7B   + GSM8K        (Batch=256,  Seq=2048, TP=1, n=4, Micro=4) → ~88MB   (small)"
echo "  S-03: 7B   + GSM8K+MATH   (Batch=512,  Seq=4096, TP=1, n=4, Micro=2) → ~370MB  (medium-low)"
echo "  S-04: 14B  + GSM8K+MATH   (Batch=1024, Seq=8192, TP=2, n=4, Micro=1) → ~1.5GB  (medium)"
echo "  S-05: 14B  + GSM8K+MATH   (Batch=2048, Seq=8192, TP=2, n=8, Micro=1) → ~5.9GB  (large)"
echo ""
echo "  ── N-M 系列: 14B medium (~1.5GB) 跨机对照 ──"
echo "  N-M-02: 2 机 16 卡  (Batch=1024, Seq=8192, TP=2, n=4, Micro=1)"
echo "  N-M-04: 4 机 32 卡  (Batch=1024, Seq=8192, TP=2, n=4, Micro=1)"
echo "  N-M-08: 8 机 64 卡  (Batch=1024, Seq=8192, TP=2, n=4, Micro=1)"
echo ""
echo "  ── N-L 系列: 14B large (~5.9GB) 跨机对照 ──"
echo "  N-L-04: 4 机 32 卡  (Batch=2048, Seq=8192, TP=2, n=8, Micro=1)"
echo "  N-L-08: 8 机 64 卡  (Batch=2048, Seq=8192, TP=2, n=8, Micro=1)"
echo ""
echo "用法示例:"
echo "  运行所有测试:           $0"
echo "  运行单个测试:           $0 --test S-01"
echo "  按节点数过滤:           $0 --nnodes 2"
echo "  仅运行离线分析:         $0 --skip-profile --analyse"
echo "  运行测试后进行分析:     $0 --analyse"
echo "  启用调用栈记录:         $0 --with-stack"
echo "=============================================="
