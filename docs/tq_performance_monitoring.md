# TQ性能监控使用指南

本文档介绍如何使用TransferQueue性能监控功能。

## 一、启用日志追踪

### 1.1 环境变量配置

```bash
# 启用函数级追踪日志（推荐）
export TQ_TRACE_ENABLED=1

# 启用TQ操作详细日志（可选，用于深入分析）
export TQ_TRACE_DETAIL_ENABLED=1

# 设置VERL日志级别
export VERL_LOGGING_LEVEL=INFO  # 或DEBUG以查看详细日志
```

### 1.2 运行训练

启用环境变量后，正常运行训练任务：

```bash
export TQ_TRACE_ENABLED=1
export VERL_LOGGING_LEVEL=INFO
python your_training_script.py
```

日志将自动输出到标准输出或日志文件中。

## 二、日志格式说明

### 2.1 TQ-TRACE日志（函数级）

记录每个函数调用的完整生命周期：

```
[TQ-TRACE] trace_id=<uuid> | func=<function_name> | stage=<stage> | <metrics>
```

**阶段（stage）：**
- `FUNCTION_START` - 函数开始
- `GET_START/GET_END` - 数据获取阶段
- `COMPUTE_START/COMPUTE_END` - 业务逻辑执行阶段
- `PUT_START/PUT_END` - 数据存储阶段
- `FUNCTION_END` - 函数结束（包含总结信息）

**示例：**
```
[TQ-TRACE] trace_id=a1b2c3d4 | func=rollout | stage=FUNCTION_END | total_duration=0.914024s | get_ratio=5.69% | compute_ratio=87.52% | put_ratio=6.78%
```

### 2.2 TQ-DETAIL日志（TQ操作细节）

记录TransferQueue Client的具体操作：

```
[TQ-DETAIL] trace_id=<uuid> | stage=<stage> | <metrics>
```

**阶段：**
- `TQ_CLIENT_GET_START/END` - TQ Get操作
- `TQ_CLIENT_PUT_START/END` - TQ Put操作

**示例：**
```
[TQ-DETAIL] trace_id=a1b2c3d4 | stage=TQ_CLIENT_PUT_END | duration=0.060000s | throughput_mbps=253.33
```

## 三、日志分析

### 3.1 使用分析脚本

我们提供了日志分析脚本 `scripts/analyze_tq_logs.py`：

```bash
# 分析日志文件
python scripts/analyze_tq_logs.py /path/to/training.log
```

**输出示例：**
```
================================================================================
TransferQueue Performance Analysis Report
================================================================================

Total Traces: 150

────────────────────────────────────────────────────────────────────────────────
Function: rollout
────────────────────────────────────────────────────────────────────────────────
  调用次数:           50
  平均总耗时:         0.914024s
  平均GET耗时:        0.052000s (5.69%)
  平均COMPUTE耗时:    0.800000s (87.52%)
  平均PUT耗时:        0.062024s (6.78%)
  TQ总开销:           12.47%

================================================================================
全局汇总
================================================================================
  总执行时间:         45.701200s
  总TQ时间:           5.700100s
  全局TQ开销占比:     12.47%
================================================================================
```

### 3.2 手动分析

也可以使用grep命令过滤日志：

```bash
# 查看所有函数结束日志
grep "FUNCTION_END" training.log

# 查看特定函数的追踪
grep "func=rollout" training.log | grep "TQ-TRACE"

# 查看TQ操作详情
grep "TQ-DETAIL" training.log
```

## 四、关键指标解读

### 4.1 耗时占比

- **get_ratio** - GET数据耗时占比（从TQ读取数据）
- **compute_ratio** - 业务逻辑耗时占比（实际计算）
- **put_ratio** - PUT数据耗时占比（向TQ写入数据）

**理想分布：**
- compute_ratio应占大部分（>80%）
- TQ总开销（get_ratio + put_ratio）应尽量小（<20%）

### 4.2 数据吞吐量

**throughput_mbps** - TQ数据传输速度（MB/s）

- 高吞吐量（>500 MB/s）- 良好
- 中等吞吐量（100-500 MB/s）- 一般
- 低吞吐量（<100 MB/s）- 可能有性能问题

### 4.3 异常情况

如果发现以下情况，可能需要优化：

1. **TQ总开销 > 30%** - TQ操作占用过多时间
2. **GET/PUT耗时波动大** - 网络或存储不稳定
3. **吞吐量持续偏低** - 可能是存储后端瓶颈

## 五、性能优化建议

### 5.1 如果TQ开销过高

1. 检查存储后端配置（Redis/分布式存储）
2. 优化批次大小（batch_size），减少传输次数
3. 考虑数据压缩

### 5.2 如果吞吐量偏低

1. 检查网络带宽
2. 检查存储IOPS
3. 优化数据序列化方式

### 5.3 日志性能影响

- TQ-TRACE日志开销 < 1%（推荐常开）
- TQ-DETAIL日志开销 < 2%（按需开启）
- 大规模训练建议采样记录（每N次记录一次）

## 六、TraceID机制

### 6.1 自动生成TraceID

在创建DataProto时自动生成：

```python
import uuid
from verl.protocol import DataProto

# 在业务逻辑中
data_proto = DataProto(...)
data_proto.meta_info['trace_id'] = str(uuid.uuid4())
```

### 6.2 TraceID传递

TraceID会自动通过BatchMeta在整个数据流中传递，无需手动处理。

## 七、常见问题

### Q1: 日志文件过大怎么办？

A: 可以：
1. 定期轮转日志文件
2. 只在需要时启用TQ追踪
3. 使用采样记录（修改代码添加采样逻辑）

### Q2: 没有看到TQ-TRACE日志？

A: 检查：
1. `TQ_TRACE_ENABLED=1` 是否正确设置
2. `VERL_LOGGING_LEVEL` 是否设置为INFO或DEBUG
3. TransferQueue是否启用（`TRANSFER_QUEUE_ENABLE=1`）

### Q3: trace_id显示为N/A？

A: 表示BatchMeta中没有trace_id字段，这不影响日志功能，但无法关联同一数据的完整流转路径。建议在数据源头添加trace_id。

## 八、高级用法

### 8.1 自定义采样率

修改`transferqueue_utils.py`，添加采样逻辑：

```python
import random

# 在async_inner中
if TQ_TRACE_ENABLED and random.random() < 0.1:  # 10%采样率
    # 记录日志
    ...
```

### 8.2 集成到监控系统

可以将日志导出到：
- Prometheus + Grafana
- ELK Stack (Elasticsearch + Logstash + Kibana)
- 分布式追踪系统（Jaeger、Zipkin）

---

更多问题请参考 `tq_logging_plan.md` 或联系开发团队。
