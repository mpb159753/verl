#!/bin/bash
# run_profile_npu.sh — 多配置 NPU Profile 脚本
# 用 --help 查看完整帮助

set -x
set -e

# =====================================================
# 辅助函数
# =====================================================
log(){ echo -e "[RUN] $(date +'%F %T') $*"; }

# =====================================================
# 帮助信息
# =====================================================
show_help() {
    cat <<EOF
用法: $0 [选项]

选项:
  --use-transfer-queue      启用 TransferQueue 模式（默认: 禁用，使用标准 main_ppo）
  --train-url PATH          指定输出根目录；output → PATH/output/，log → PATH/log/
                            默认: 脚本同级目录
  --test ID                 只运行指定测试（如 S-01、N-M-04）
  --nnodes N                只运行节点数为 N 的测试
  --with-stack              开启调用栈采集（profile contents: stack/module/npu/cpu）
  --analyse                 在 profile 结束后执行离线分析并清理原始数据
  --skip-profile            跳过 profile，仅执行离线分析（配合 --analyse 使用）
  --head-ip IP              Ray Head 节点 IP（SSH 模式）
  --worker-ips IP,...       Worker 节点 IP 列表，逗号分隔（SSH 模式）
  --ray-port PORT           Ray 端口（默认: 6766）
  --num-gpus N              每节点 NPU 数（默认: 8）
  --ssh-user USER           SSH 登录用户（默认: 当前用户）
  --help                    显示此帮助并退出

测试矩阵:
  S 系列 (S-01~S-05): 固定 2 机 16 卡，覆盖 tiny→large 不同 put 体积
    S-01: 0.5B + GSM8K        (Batch=64,   Seq=1024, TP=1, n=1, Micro=8) → ~2.7MB  (tiny)
    S-02: 7B   + GSM8K        (Batch=256,  Seq=2048, TP=1, n=4, Micro=4) → ~88MB   (small)
    S-03: 7B   + GSM8K+MATH   (Batch=512,  Seq=4096, TP=1, n=4, Micro=2) → ~370MB  (medium-low)
    S-04: 14B  + GSM8K+MATH   (Batch=1024, Seq=8192, TP=2, n=4, Micro=1) → ~1.5GB  (medium)
    S-05: 14B  + GSM8K+MATH   (Batch=2048, Seq=8192, TP=2, n=8, Micro=1) → ~5.9GB  (large)

  N-M 系列: 14B medium (~1.5GB) 跨机规模对照
    N-M-02: 2 机 16 卡  (Batch=1024, Seq=8192, TP=2, n=4, Micro=1)
    N-M-04: 4 机 32 卡  (Batch=1024, Seq=8192, TP=2, n=4, Micro=1)
    N-M-08: 8 机 64 卡  (Batch=1024, Seq=8192, TP=2, n=4, Micro=1)

  N-L 系列: 14B large (~5.9GB) 跨机规模对照
    N-L-04: 4 机 32 卡  (Batch=2048, Seq=8192, TP=2, n=8, Micro=1)
    N-L-08: 8 机 64 卡  (Batch=2048, Seq=8192, TP=2, n=8, Micro=1)

多节点启动模式:
  模式 A — Volcano Job（华为云 ModelArts / CCE，推荐）:
    每个 Pod 自动注入以下环境变量，脚本根据 VC_TASK_INDEX 判断 Head/Worker 角色：
      VC_TASK_INDEX / MA_VJ_NAME / MA_TASK_NAME / MA_NUM_HOSTS / MA_NUM_GPUS / MA_CURRENT_HOST_IP

  模式 B — SSH 手动多节点（Volcano 变量缺失时自动降级）:
    $0 --head-ip 192.168.1.10 --worker-ips 192.168.1.11,192.168.1.12

示例:
  $0                                          # 标准模式，运行全部测试
  $0 --use-transfer-queue                     # TransferQueue 模式，运行全部测试
  $0 --test S-01                              # 只运行 S-01
  $0 --nnodes 2                               # 只运行 2 机配置
  $0 --analyse                                # 运行测试后执行离线分析
  $0 --skip-profile --analyse                 # 仅执行离线分析
  $0 --with-stack                             # 开启调用栈采集
  $0 --train-url /data/my_run                 # 自定义输出根目录
EOF
}

# =====================================================
# 路径配置
# =====================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_QWEN_0_5B="${SCRIPT_DIR}/model/Qwen2.5-0.5B-Instruct"
MODEL_QWEN_7B="${SCRIPT_DIR}/model/Qwen2.5-7B-Instruct"
MODEL_QWEN_14B="${SCRIPT_DIR}/model/Qwen2.5-14B-Instruct"

DATASET_GSM8K_TRAIN="${SCRIPT_DIR}/dataset/gsm8k/train.parquet"
DATASET_GSM8K_TEST="${SCRIPT_DIR}/dataset/gsm8k/test.parquet"
DATASET_MATH_TRAIN="${SCRIPT_DIR}/dataset/math/train.parquet"
DATASET_MATH_TEST="${SCRIPT_DIR}/dataset/math/test.parquet"
DATASET_COMBINED_TRAIN="[${DATASET_GSM8K_TRAIN},${DATASET_MATH_TRAIN}]"
DATASET_COMBINED_TEST="[${DATASET_GSM8K_TEST},${DATASET_MATH_TEST}]"

# =====================================================
# Profile 配置
# =====================================================
TOTAL_STEPS=6
PROFILE_STEPS='[2,4]'

# =====================================================
# 环境变量
# =====================================================
export TORCHDYNAMO_DISABLE=1
# HCCL 跨节点通信配置：禁用 IP 白名单，指定网络接口
export HCCL_WHITELIST_DISABLE=1
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"

# =====================================================
# 环境准备（借鉴 run_roma.sh）
# =====================================================
# Ray 临时目录：将 /tmp/ray 软链到 /cache 避免容器 rootfs 溢出
# 强制清理残留 Ray 进程和旧 session 数据，防止 GCS session 冲突
ray stop --force 2>/dev/null || true
# 杀掉所有可能残留的 Ray/GCS 进程（ray stop 可能因 session 不匹配而失败）
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "gcs_server" 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true
pkill -9 -f "plasma_store" 2>/dev/null || true
sleep 2
# 清理所有 Ray 存储
rm -rf /tmp/ray /dev/shm/ray_* /dev/shm/plasma_* || true
rm -rf /cache/ray /cache/ray_tmp 2>/dev/null || true
mkdir -p /cache/ray /cache/ray_tmp 2>/dev/null || true
ln -s /cache/ray /tmp/ray 2>/dev/null || true
export TMPDIR="${TMPDIR:-/cache/ray_tmp}"

# Ascend toolkit 权限修复（容器内非 root 用户可能缺写权限）
ASCEND_DIR="/usr/local/Ascend/ascend-toolkit/latest"
chmod -R a+wr "${ASCEND_DIR}/opp/vendors" 2>/dev/null || true
chmod a+wr "${ASCEND_DIR}/opp/"*.info 2>/dev/null || true
chmod -R 777 "${ASCEND_DIR}" 2>/dev/null || true
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh || true

log "Python: $(which python) ($(python -V 2>&1))"

# 安装/校验差异依赖
if [ -f "${SCRIPT_DIR}/diff_requirements.txt" ]; then
    pip install -r "${SCRIPT_DIR}/diff_requirements.txt"
fi

# NPU 设备检查
if command -v ascend-dmi >/dev/null 2>&1; then
    ascend-dmi -c || true
else
    log "ascend-dmi 不存在，跳过 NPU 设备检查"
fi

# Ray object store：min(35% Mem, 80% /dev/shm)，且 ≥512MB
if [ -f /proc/meminfo ]; then
    _mem_bytes=$(awk '/MemTotal/{printf "%.0f", $2*1024}' /proc/meminfo)
    _shm_bytes=$(df -B1 /dev/shm 2>/dev/null | awk 'NR==2{print $2}' || echo 0)
    _target_mem=$(( _mem_bytes * 35 / 100 ))
    _target_shm=$(( _shm_bytes * 80 / 100 ))
    RAY_OBJECT_STORE_MEM=$(( _target_mem < _target_shm ? _target_mem : _target_shm ))
    [ "$RAY_OBJECT_STORE_MEM" -lt $((512*1024*1024)) ] && RAY_OBJECT_STORE_MEM=$((512*1024*1024))
    log "ray_object_store_mem(bytes): ${RAY_OBJECT_STORE_MEM}  (/dev/shm=${_shm_bytes})"
fi

# Ray 统一 Python + 资源限制
export RAY_PYTHON_WORKER_COMMAND="$(command -v python) -u"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-64}"
export RAY_USAGE_STATS_ENABLED=0

# =====================================================
# 节点身份：最先推导并打印，确认平台变量已注入
# =====================================================
# NODE_IDX: 0=Head，其余=Worker（对应 VC_TASK_INDEX）
NODE_IDX="${VC_TASK_INDEX:-0}"
# NODE_IP: 容器内 Pod IP，用于 Ray 监听绑定
NODE_IP="${MA_CURRENT_IP:-$(hostname -I | awk '{print $1}')}"
# MASTER_IP: Head 节点 IP（Volcano 模式下通过 DNS 解析，SSH 模式下由参数给出）
if [ -n "${VC_TASK_INDEX}" ]; then
    _HEAD_DOMAIN="${MA_VJ_NAME:-verl-job}-${MA_TASK_NAME:-worker}-0.${MA_VJ_NAME:-verl-job}"
    MASTER_IP="$(getent hosts "${_HEAD_DOMAIN}" 2>/dev/null | awk '{print $1}' | head -n1)"
    # Fallback: 从 VC_WORKER_HOSTS 取第一个 hostname 解析
    if [ -z "${MASTER_IP}" ] && [ -n "${VC_WORKER_HOSTS}" ]; then
        MASTER_IP="$(echo "${VC_WORKER_HOSTS}" | cut -d',' -f1 | \
            xargs -I{} getent hosts {} 2>/dev/null | awk '{print $1}' | head -n1)"
    fi
else
    MASTER_IP=""  # SSH 模式下参数解析后由 HEAD_IP 填充
fi
echo "==========================================="
echo " [节点身份确认]"
echo "   NODE_IDX  (VC_TASK_INDEX) = ${NODE_IDX}"
echo "   MASTER_IP                 = ${MASTER_IP:-(SSH 模式，待参数解析)}"
echo "   NODE_IP   (MA_CURRENT_IP) = ${NODE_IP}"
echo "   MA_NUM_HOSTS              = ${MA_NUM_HOSTS:-<未设置>}"
echo "==========================================="

# =====================================================
# 默认参数值
# =====================================================
RUN_ANALYSIS=false
SKIP_PROFILE=false
TEST_FILTER=""
WITH_STACK=false
NODE_FILTER=""
TRAIN_URL=""
USE_TRANSFER_QUEUE=false
HEAD_IP=""
WORKER_IPS=""
RAY_PORT=6766
NUM_GPUS_PER_NODE=8
SSH_USER="${USER}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

# =====================================================
# 解析命令行参数
# =====================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --use-transfer-queue)
            USE_TRANSFER_QUEUE=true
            shift
            ;;
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
        --train-url)
            TRAIN_URL="$2"
            shift 2
            ;;
        --head-ip)
            HEAD_IP="$2"
            shift 2
            ;;
        --worker-ips)
            WORKER_IPS="$2"
            shift 2
            ;;
        --ray-port)
            RAY_PORT="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS_PER_NODE="$2"
            shift 2
            ;;
        --ssh-user)
            SSH_USER="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "运行 $0 --help 查看帮助"
            exit 1
            ;;
    esac
done

# =====================================================
# 根据模式设置差异化配置
# =====================================================
if [ "${USE_TRANSFER_QUEUE}" = true ]; then
    export TRANSFER_QUEUE_ENABLE=1
    export TQ_PROFILER_ENABLED=1
    export TQ_TRACE_ENABLED=1
    PYTHON_MODULE="recipe.transfer_queue.main_ppo"
    PYTHON_EXTRA_ARGS="--config-name='transfer_queue_ppo_trainer'"
    TQ_ENABLE_FLAG="true"
    PROJECT_NAME="multi_config_profiler"
    PROCESS_PATTERN="recipe.transfer_queue"
    OUTPUT_SUBDIR="multi_config_profiler_npu"
else
    PYTHON_MODULE="verl.trainer.main_ppo"
    PYTHON_EXTRA_ARGS=""
    TQ_ENABLE_FLAG="false"
    PROJECT_NAME="multi_config_profiler_main"
    PROCESS_PATTERN="verl.trainer.main_ppo"
    OUTPUT_SUBDIR="multi_config_main_ppo_npu"
fi

# =====================================================
# 根据 train_url 确定输出和日志目录
# =====================================================
if [ -n "${TRAIN_URL}" ]; then
    PROFILE_BASE_OUTPUT="${TRAIN_URL}/output/${OUTPUT_SUBDIR}"
    LOG_DIR="${TRAIN_URL}/log/${OUTPUT_SUBDIR}"
else
    PROFILE_BASE_OUTPUT="${SCRIPT_DIR}/output/${OUTPUT_SUBDIR}"
    LOG_DIR="${SCRIPT_DIR}/log/${OUTPUT_SUBDIR}"
fi

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
echo "日志将保存到: ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

# =====================================================
# 停止 Worker 节点上的 Ray 进程
# =====================================================
stop_ray_workers() {
    if [ -n "${VC_TASK_INDEX}" ]; then
        echo "[Volcano] 停止本地 Ray 进程 (Pod ${VC_TASK_INDEX})..."
        return 0
    fi
    if [ -z "${WORKER_IPS}" ]; then
        return 0
    fi
    echo "[SSH] 停止 Worker 节点 Ray 进程..."
    IFS=',' read -ra _WORKERS <<< "${WORKER_IPS}"
    for _wp in "${_WORKERS[@]}"; do
        echo "  → 停止 Worker: ${_wp}"
        ssh ${SSH_OPTS} "${SSH_USER}@${_wp}" \
            "ray stop --force 2>/dev/null || true; \
             pkill -9 -f '${PROCESS_PATTERN}' 2>/dev/null || true; \
             rm -rf /dev/shm/ray_* /dev/shm/plasma_* /tmp/ray 2>/dev/null || true" \
            2>/dev/null || echo "  ⚠ Worker ${_wp} SSH 停止失败（可忽略）"
    done
}

# =====================================================
# 等待期望节点数加入集群（通过 NPU 总数判断，与脚本1保持一致）
# =====================================================
_wait_all_nodes() {
    local expected_nodes="$1"
    local npus_per_node="$2"
    local expected_npus=$(( expected_nodes * npus_per_node ))
    echo "等待集群 NPU 就绪（目标: ${expected_nodes} 节点 × ${npus_per_node} NPU = ${expected_npus} NPU 总量）..."
    while true; do
        local status_out
        status_out=$(ray status 2>/dev/null || true)
        local npu_avail
        npu_avail=$(echo "${status_out}" | grep -oP '(?<=/)[0-9]+(?=\.[0-9]+\s*NPU)' 2>/dev/null | head -n1 || echo 0)
        if [ "${npu_avail:-0}" -ge "${expected_npus}" ] 2>/dev/null; then
            echo "Ray 集群就绪: ${npu_avail} NPU 可用（${expected_nodes} 节点）"
            ray status
            break
        fi
        echo "  等待中: 当前 ${npu_avail:-0}/${expected_npus} NPU 就绪..."
        sleep 5
    done
}

# =====================================================
# 启动多节点 Ray 集群
#   使用 NODE_IDX（0=Head，其余=Worker）和 MASTER_IP（IP 直连）
#   Volcano 模式：NODE_IDX=VC_TASK_INDEX，MASTER_IP 已在脚本开头解析
#   SSH 模式：NODE_IDX=0（始终为 Head），MASTER_IP=HEAD_IP
# =====================================================
start_ray_cluster() {
    # SSH 模式下此时参数已解析，补全 MASTER_IP 和 NNODES_TOTAL
    if [ -z "${VC_TASK_INDEX}" ]; then
        MASTER_IP="${HEAD_IP:-${NODE_IP}}"
        if [ -n "${WORKER_IPS}" ]; then
            IFS=',' read -ra _W <<< "${WORKER_IPS}"
            NNODES_TOTAL=$(( 1 + ${#_W[@]} ))
        else
            NNODES_TOTAL=1
        fi
    else
        NNODES_TOTAL="${MA_NUM_HOSTS:-2}"
    fi

    echo "======================================"
    echo "初始化 Ray 集群..."
    echo "  NODE_IDX    : ${NODE_IDX}"
    echo "  MASTER_IP   : ${MASTER_IP}"
    echo "  NODE_IP     : ${NODE_IP}"
    echo "  NNODES_TOTAL: ${NNODES_TOTAL}"
    echo "  RAY 端口    : ${RAY_PORT}"
    echo "  NPU/节点    : ${NUM_GPUS_PER_NODE}"
    echo "======================================"

    if [[ "${NODE_IDX}" == "0" ]]; then
        echo "[Head] 启动 Ray Head 节点..."
        # 检查 Ray GCS 端口是否已被占用
        if ss -tlnp 2>/dev/null | grep -q ":${RAY_PORT} " || \
           lsof -iTCP:${RAY_PORT} -sTCP:LISTEN 2>/dev/null | grep -q .; then
            echo "⚠ 警告: 端口 ${RAY_PORT} 已被占用，占用进程如下:"
            ss -tlnp 2>/dev/null | grep ":${RAY_PORT} " || true
            lsof -iTCP:${RAY_PORT} -sTCP:LISTEN 2>/dev/null || true
            echo "⚠ 请先执行 'ray stop --force' 或手动 kill 上述进程后重试"
        fi
        ray start --head \
            --node-ip-address="${NODE_IP}" \
            --port=${RAY_PORT} \
            --resources='{"NPU": '"${NUM_GPUS_PER_NODE}"'}' \
            --num-cpus="$(nproc)" \
            --include-dashboard=false

        # SSH 模式：手动 SSH 启动各 Worker
        if [ -n "${WORKER_IPS}" ]; then
            IFS=',' read -ra _WORKERS <<< "${WORKER_IPS}"
            for _wp in "${_WORKERS[@]}"; do
                echo "SSH 启动 Worker: ${_wp}（带重试）"
                ssh ${SSH_OPTS} "${SSH_USER}@${_wp}" \
                    "while true; do \
                        ray start --address='${MASTER_IP}:${RAY_PORT}' \
                            --resources='{\"NPU\": ${NUM_GPUS_PER_NODE}}' \
                            --num-cpus=\$(nproc) && break; \
                        echo 'Worker ${_wp} 连接失败，5s 后重试...'; sleep 5; \
                    done" \
                    2>&1 | sed "s/^/  [${_wp}] /" &
            done
        fi

        _wait_all_nodes "${NNODES_TOTAL}" "${NUM_GPUS_PER_NODE}"
    else
        echo "[Worker] NODE_IDX=${NODE_IDX}，连接 Ray Head: ${MASTER_IP}:${RAY_PORT}..."
        while true; do
            ray start \
                --address="${MASTER_IP}:${RAY_PORT}" \
                --node-ip-address="${NODE_IP}" \
                --resources='{"NPU": '"${NUM_GPUS_PER_NODE}"'}' \
                --num-cpus="$(nproc)" || true
            if ray status >/dev/null 2>&1; then
                echo "[Worker] NODE_IDX=${NODE_IDX} 成功加入 Ray 集群！"
                break
            fi
            echo "[Worker] 连接失败，5s 后重试..."
            sleep 5
        done
        echo "[Worker] 保持容器存活等待任务..."
        tail -f /dev/null
        exit 0
    fi
}

cleanup_resources() {
    echo "======================================"
    echo "清理残留资源..."
    echo "======================================"
    stop_ray_workers
    echo "停止本地 Ray Head..."
    ray stop --force 2>/dev/null || true
    echo "终止残留 Python 进程..."
    pkill -9 -f "${PROCESS_PATTERN}" 2>/dev/null || true
    pkill -9 -f "python.*main_ppo" 2>/dev/null || true
    pkill -9 -f "WorkerDict" 2>/dev/null || true
    pkill -9 -f "TaskRunner" 2>/dev/null || true
    echo "等待端口释放..."
    sleep 5
    echo "清理共享内存及 Ray 临时目录..."
    rm -rf /dev/shm/ray_* 2>/dev/null || true
    rm -rf /dev/shm/plasma_* 2>/dev/null || true
    rm -rf /tmp/ray 2>/dev/null || true
    rm -rf /cache/ray/* /cache/ray_tmp/* 2>/dev/null || true
    echo "资源清理完成"
}

cleanup_on_exit() {
    echo ""
    echo "======================================"
    echo "脚本退出，执行清理..."
    echo "======================================"
    ray stop --force 2>/dev/null || true
    echo "清理完成"
}
trap cleanup_on_exit EXIT

# =====================================================
# 运行单个测试配置
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

    local PROFILE_CONTENTS='[]'
    if [ "${WITH_STACK_FLAG}" = "true" ]; then
        PROFILE_CONTENTS='["stack", "module", "npu", "cpu"]'
    fi

    local PROFILE_OUTPUT="${PROFILE_BASE_OUTPUT}/${TEST_ID}"
    local HALF_SEQ_LENGTH=$((MAX_SEQ_LENGTH / 2))
    local EFFECTIVE_BATCH=$((GLOBAL_BATCH_SIZE * ROLLOUT_N))
    local PUT_MB=$(( EFFECTIVE_BATCH * MAX_SEQ_LENGTH * 44 / 1024 / 1024 ))

    echo "======================================"
    echo "开始运行测试: ${TEST_ID}"
    echo "模式: $([ "${USE_TRANSFER_QUEUE}" = true ] && echo 'TransferQueue' || echo '标准 main_ppo')"
    echo "模型: ${MODEL_PATH}"
    echo "数据集: ${TRAIN_DATA}"
    echo "Global Batch Size: ${GLOBAL_BATCH_SIZE}"
    echo "Max Seq Length: ${MAX_SEQ_LENGTH}"
    echo "TP Size: ${TP_SIZE} | Micro Batch: ${MICRO_BATCH}"
    echo "Nodes: ${NNODES} | Rollout N: ${ROLLOUT_N} | With Stack: ${WITH_STACK_FLAG}"
    echo "Effective Batch (batch×n): ${EFFECTIVE_BATCH} | 估算 put 大小: ~${PUT_MB}MB"
    echo "Profile 输出: ${PROFILE_OUTPUT}"
    echo "======================================"

    mkdir -p "${PROFILE_OUTPUT}"

    # shellcheck disable=SC2086
    python3 -m ${PYTHON_MODULE} \
        ${PYTHON_EXTRA_ARGS} \
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
        trainer.project_name="${PROJECT_NAME}" \
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
        transfer_queue.enable=${TQ_ENABLE_FLAG} \
        '+ray_kwargs.ray_init.address=auto'

    echo "======================================"
    echo "测试 ${TEST_ID} 完成！Profile 输出: ${PROFILE_OUTPUT}"
    echo "======================================"
}

# =====================================================
# 离线分析
# =====================================================
run_offline_analysis() {
    local TEST_ID=$1
    local PROFILE_OUTPUT="${PROFILE_BASE_OUTPUT}/${TEST_ID}"
    echo "======================================"
    echo "开始离线分析: ${TEST_ID}"
    echo "======================================"
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
# 清理原始 profile 数据（保留 ascend_profiler_output）
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
            if [ -d "${STEP_DIR}/ascend_profiler_output" ]; then
                local BEFORE_SIZE
                BEFORE_SIZE=$(du -sh "${STEP_DIR}" 2>/dev/null | cut -f1)
                find "${STEP_DIR}" -mindepth 1 -maxdepth 1 \
                    ! -name 'ascend_profiler_output' \
                    -exec rm -rf {} +
                local AFTER_SIZE
                AFTER_SIZE=$(du -sh "${STEP_DIR}" 2>/dev/null | cut -f1)
                echo "Step ${STEP}: ${BEFORE_SIZE} → ${AFTER_SIZE} (保留 ascend_profiler_output)"
            else
                echo "警告: Step ${STEP} 未找到 ascend_profiler_output, 跳过清理"
            fi
        fi
    done
    echo "清理完成: ${TEST_ID}"
}

# =====================================================
# 测试配置定义
# gen_sequences put ≈ batch × n × seqlen × 44B
# =====================================================
declare -a TESTS=(
    "S-00 ${MODEL_QWEN_0_5B} ${DATASET_GSM8K_TRAIN} ${DATASET_GSM8K_TEST} 64 1024 1 8 1 1"  # 单机8卡: normalized_mini(64/8=8) 被 micro=8 整除
    "S-01 ${MODEL_QWEN_0_5B} ${DATASET_GSM8K_TRAIN} ${DATASET_GSM8K_TEST} 64 1024 1 4 2 1"  # 2节点: normalized_mini(64/16=4) 需被 micro=4 整除
    "S-02 ${MODEL_QWEN_7B} ${DATASET_GSM8K_TRAIN} ${DATASET_GSM8K_TEST} 256 2048 1 4 2 4"
    "S-03 ${MODEL_QWEN_7B} ${DATASET_COMBINED_TRAIN} ${DATASET_COMBINED_TEST} 512 4096 1 2 2 4"
    "S-04 ${MODEL_QWEN_14B} ${DATASET_COMBINED_TRAIN} ${DATASET_COMBINED_TEST} 1024 8192 2 1 2 4"
    "S-05 ${MODEL_QWEN_14B} ${DATASET_COMBINED_TRAIN} ${DATASET_COMBINED_TEST} 2048 8192 2 1 2 8"
)
declare -a TESTS_N=(
    "N-M-02 ${MODEL_QWEN_14B} ${DATASET_COMBINED_TRAIN} ${DATASET_COMBINED_TEST} 1024 8192 2 1 2 4"
    "N-M-04 ${MODEL_QWEN_14B} ${DATASET_COMBINED_TRAIN} ${DATASET_COMBINED_TEST} 1024 8192 2 1 4 4"
    "N-M-08 ${MODEL_QWEN_14B} ${DATASET_COMBINED_TRAIN} ${DATASET_COMBINED_TEST} 1024 8192 2 1 8 4"
    "N-L-04 ${MODEL_QWEN_14B} ${DATASET_COMBINED_TRAIN} ${DATASET_COMBINED_TEST} 2048 8192 2 1 4 8"
    "N-L-08 ${MODEL_QWEN_14B} ${DATASET_COMBINED_TRAIN} ${DATASET_COMBINED_TEST} 2048 8192 2 1 8 8"
)
ALL_TESTS=("${TESTS[@]}" "${TESTS_N[@]}")

# =====================================================
# 启动 Ray 集群
# =====================================================
if [ -n "${VC_TASK_INDEX}" ] || [ "${SKIP_PROFILE}" = false ] || [ -n "${WORKER_IPS}" ]; then
    cleanup_resources
    start_ray_cluster
fi

# =====================================================
# 运行 profiling
# =====================================================
if [ "${SKIP_PROFILE}" = false ]; then
    for test_config in "${ALL_TESTS[@]}"; do
        read -r TEST_ID MODEL TRAIN VAL BATCH SEQ TP MICRO NODES ROLLN <<< "${test_config}"

        if [ -n "${TEST_FILTER}" ] && [ "${TEST_ID}" != "${TEST_FILTER}" ]; then
            continue
        fi
        if [ -n "${NODE_FILTER}" ] && [ "${NODES}" != "${NODE_FILTER}" ]; then
            continue
        fi
        if [ -n "${VC_TASK_INDEX}" ] && [ -n "${MA_NUM_HOSTS}" ]; then
            NODES="${MA_NUM_HOSTS}"
            echo "ℹ Volcano 模式：强制将训练节点数设为 ${NODES}"
        fi

        echo "终止上一轮残留 Python 进程..."
        pkill -9 -f "${PROCESS_PATTERN}" 2>/dev/null || true
        pkill -9 -f "python.*main_ppo" 2>/dev/null || true
        pkill -9 -f "WorkerDict" 2>/dev/null || true
        pkill -9 -f "TaskRunner" 2>/dev/null || true
        sleep 3

        run_profile_test "${TEST_ID}" "${MODEL}" "${TRAIN}" "${VAL}" \
            "${BATCH}" "${SEQ}" "${TP}" "${MICRO}" "${NODES}" "${ROLLN}" "${WITH_STACK}"
    done
fi

# =====================================================
# 运行离线分析
# =====================================================
if [ "${RUN_ANALYSIS}" = true ]; then
    for test_config in "${ALL_TESTS[@]}"; do
        read -r TEST_ID _ <<< "${test_config}"
        if [ -n "${TEST_FILTER}" ] && [ "${TEST_ID}" != "${TEST_FILTER}" ]; then
            continue
        fi
        run_offline_analysis "${TEST_ID}"
        cleanup_profile_raw_data "${TEST_ID}"
    done
fi

echo ""
echo "=============================================="
echo "所有任务完成！输出目录: ${PROFILE_BASE_OUTPUT}"
echo "运行 $0 --help 查看完整帮助"
echo "=============================================="
