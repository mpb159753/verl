# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import functools
import inspect
import logging
import os
import threading
import time
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from verl.single_controller.base.decorator import Dispatch

from tensordict import TensorDict

try:
    from transfer_queue import (
        AsyncTransferQueueClient,
        BatchMeta,
        TransferQueueClient,
    )

except ImportError:
    # TODO: Use a hacky workaround for ImportError since
    # transfer_queue isn't a default verl dependency.
    class BatchMeta:
        pass


from verl.protocol import DataProto

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TRANSFER_QUEUE_CLIENT = None

is_transferqueue_enabled = os.environ.get("TRANSFER_QUEUE_ENABLE", False)

# TQ Performance Logging Configuration
TQ_TRACE_ENABLED = os.getenv("TQ_TRACE_ENABLED", "0") == "1"
TQ_TRACE_DETAIL_ENABLED = os.getenv("TQ_TRACE_DETAIL_ENABLED", "0") == "1"
TQ_PROFILER_ENABLED = os.getenv("TQ_PROFILER_ENABLED", "0") == "1"


class TQMetricsCollector:
    """收集一个 step 内所有 TQ 操作的 timing 数据（单例模式）"""

    _instance = None

    def __init__(self):
        self.reset()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self):
        """重置所有收集的数据"""
        self.get_durations = []
        self.put_durations = []
        self.compute_durations = []
        self.total_durations = []
        self.data_sizes_mb = []

    def record_get(self, duration: float):
        self.get_durations.append(duration)

    def record_put(self, duration: float):
        self.put_durations.append(duration)

    def record_compute(self, duration: float):
        self.compute_durations.append(duration)

    def record_total(self, duration: float):
        self.total_durations.append(duration)

    def record_data_size(self, size_mb: float):
        self.data_sizes_mb.append(size_mb)

    def get_metrics(self) -> dict:
        """获取汇总的 metrics"""
        metrics = {}
        if self.get_durations:
            metrics["tq/get_time_total"] = sum(self.get_durations)
            metrics["tq/get_time_mean"] = sum(self.get_durations) / len(self.get_durations)
            metrics["tq/get_count"] = len(self.get_durations)
        if self.put_durations:
            metrics["tq/put_time_total"] = sum(self.put_durations)
            metrics["tq/put_time_mean"] = sum(self.put_durations) / len(self.put_durations)
            metrics["tq/put_count"] = len(self.put_durations)
        if self.compute_durations:
            metrics["tq/compute_time_total"] = sum(self.compute_durations)
            metrics["tq/compute_time_mean"] = sum(self.compute_durations) / len(self.compute_durations)
        if self.total_durations:
            metrics["tq/total_time"] = sum(self.total_durations)
        if self.data_sizes_mb:
            metrics["tq/data_size_mb_total"] = sum(self.data_sizes_mb)
        return metrics


def get_tq_metrics() -> dict:
    """获取当前 step 的 TQ metrics"""
    if not is_transferqueue_enabled:
        return {}
    return TQMetricsCollector.get_instance().get_metrics()


def reset_tq_metrics():
    """重置 TQ metrics 收集器（每个 step 开始时调用）"""
    if not is_transferqueue_enabled:
        return
    TQMetricsCollector.get_instance().reset()


# Lazy import profiler functions to avoid circular imports
_mark_start_range = None
_mark_end_range = None


def _get_profiler_functions():
    """Lazy load profiler functions when TQ_PROFILER_ENABLED is set."""
    global _mark_start_range, _mark_end_range
    if _mark_start_range is None:
        from verl.utils.profiler import mark_start_range, mark_end_range
        _mark_start_range = mark_start_range
        _mark_end_range = mark_end_range
    return _mark_start_range, _mark_end_range


def create_transferqueue_client(
    client_id: str,
    config,
    sync: bool = False,
) -> "AsyncTransferQueueClient | TransferQueueClient":
    global _TRANSFER_QUEUE_CLIENT
    if _TRANSFER_QUEUE_CLIENT is None:
        if sync:
            _TRANSFER_QUEUE_CLIENT = TransferQueueClient(client_id, config.controller_info)
        else:
            _TRANSFER_QUEUE_CLIENT = AsyncTransferQueueClient(client_id, config.controller_info)
        _TRANSFER_QUEUE_CLIENT.initialize_storage_manager(manager_type=config.storage_backend, config=config)

    return _TRANSFER_QUEUE_CLIENT


def get_transferqueue_client() -> "AsyncTransferQueueClient | TransferQueueClient":
    return _TRANSFER_QUEUE_CLIENT


# TODO (TQ): verl will make all actor async, so this can be cleanup later.
def _run_async_in_temp_loop(async_func: Callable[..., Any], *args, **kwargs) -> Any:
    # Use a temporary event loop in a new thread because event
    # loop may already exist in server mode
    tmp_event_loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=tmp_event_loop.run_forever,
        name="batchmeta dataproto converter",
        daemon=True,
    )

    def run_coroutine(coroutine):
        if not thread.is_alive():
            thread.start()
        future = asyncio.run_coroutine_threadsafe(coroutine, tmp_event_loop)
        return future.result()

    async def stop_loop():
        tmp_event_loop.stop()

    try:
        return run_coroutine(async_func(*args, **kwargs))
    finally:
        if thread.is_alive():
            asyncio.run_coroutine_threadsafe(stop_loop(), tmp_event_loop)
            thread.join()


def _find_batchmeta(*args, **kwargs):
    for arg in args:
        if isinstance(arg, BatchMeta):
            return arg
    for v in kwargs.values():
        if isinstance(v, BatchMeta):
            return v
    return None


def _estimate_tensordict_size(tensordict: TensorDict) -> float:
    """估算TensorDict的内存大小(MB)
    
    Args:
        tensordict: 需要估算大小的TensorDict对象
        
    Returns:
        float: 估算的内存大小(MB)
    """
    total_bytes = 0
    try:
        for key, tensor in tensordict.items():
            if hasattr(tensor, 'element_size') and hasattr(tensor, 'nelement'):
                total_bytes += tensor.nelement() * tensor.element_size()
    except Exception as e:
        logger.debug(f"Failed to estimate tensordict size: {e}")
        return 0.0
    return total_bytes / (1024 * 1024)  # 转换为MB


def _get_trace_id(batchmeta: "BatchMeta") -> str:
    """从BatchMeta获取trace_id，如果不存在则返回N/A
    
    Args:
        batchmeta: BatchMeta对象
        
    Returns:
        str: trace_id或"N/A"
    """
    try:
        return batchmeta.extra_info.get('trace_id', 'N/A')
    except Exception:
        return 'N/A'


async def _async_batchmeta_to_dataproto(batchmeta: "BatchMeta") -> DataProto:
    if batchmeta.samples == [] or batchmeta.samples is None:
        return DataProto(
            batch=TensorDict({}, batch_size=(0,)),
            non_tensor_batch={},
            meta_info=batchmeta.extra_info.copy(),
        )

    # ========== 日志点: TQ Client Get调用 ==========
    if TQ_TRACE_DETAIL_ENABLED:
        get_client_start = time.perf_counter()
        trace_id = _get_trace_id(batchmeta)
        logger.debug(
            f"[TQ-DETAIL] trace_id={trace_id} | "
            f"stage=TQ_CLIENT_GET_START | "
            f"batch_size={batchmeta.size} | timestamp={get_client_start:.6f}"
        )

    tensordict = await _TRANSFER_QUEUE_CLIENT.async_get_data(batchmeta)

    if TQ_TRACE_DETAIL_ENABLED:
        get_client_end = time.perf_counter()
        data_size = _estimate_tensordict_size(tensordict)
        logger.debug(
            f"[TQ-DETAIL] trace_id={trace_id} | "
            f"stage=TQ_CLIENT_GET_END | "
            f"duration={get_client_end - get_client_start:.6f}s | "
            f"data_size_mb={data_size:.2f} | "
            f"timestamp={get_client_end:.6f}"
        )

    return DataProto.from_tensordict(tensordict, meta_info=batchmeta.extra_info.copy())


def _batchmeta_to_dataproto(batchmeta: "BatchMeta") -> DataProto:
    return _run_async_in_temp_loop(_async_batchmeta_to_dataproto, batchmeta)


async def _async_update_batchmeta_with_output(output: DataProto, batchmeta: "BatchMeta", func_name=None) -> "BatchMeta":
    pid = os.getpid()

    for k, v in output.meta_info.items():
        batchmeta.set_extra_info(k, v)

    if len(output) > 0:
        tensordict = output.to_tensordict()
        # pop meta_info
        for key in output.meta_info.keys():
            tensordict.pop(key)

        logger.info(
            f"Task {func_name} (pid={pid}) putting output data to TransferQueue with "
            f"batch_size={tensordict.batch_size},\n"
            f"tensordict keys={list(tensordict.keys())}"
        )

        # ========== 日志点: TQ Client Put调用 ==========
        if TQ_TRACE_DETAIL_ENABLED:
            put_client_start = time.perf_counter()
            trace_id = _get_trace_id(batchmeta)
            data_size = _estimate_tensordict_size(tensordict)
            logger.debug(
                f"[TQ-DETAIL] trace_id={trace_id} | "
                f"func={func_name} | pid={pid} | "
                f"stage=TQ_CLIENT_PUT_START | "
                f"batch_size={tensordict.batch_size} | "
                f"data_size_mb={data_size:.2f} | "
                f"timestamp={put_client_start:.6f}"
            )

        updated_batch_meta = await _TRANSFER_QUEUE_CLIENT.async_put(data=tensordict, metadata=batchmeta)

        if TQ_TRACE_DETAIL_ENABLED:
            put_client_end = time.perf_counter()
            duration = put_client_end - put_client_start
            throughput = data_size / duration if duration > 0 else 0
            logger.debug(
                f"[TQ-DETAIL] trace_id={trace_id} | "
                f"func={func_name} | pid={pid} | "
                f"stage=TQ_CLIENT_PUT_END | "
                f"duration={duration:.6f}s | "
                f"throughput_mbps={throughput:.2f} | "
                f"timestamp={put_client_end:.6f}"
            )

        return updated_batch_meta
    else:
        return batchmeta


def _update_batchmeta_with_output(output: DataProto, batchmeta: "BatchMeta", func_name=None) -> "BatchMeta":
    updated_batch_meta = _run_async_in_temp_loop(_async_update_batchmeta_with_output, output, batchmeta, func_name)
    return updated_batch_meta


def _compute_need_collect(dispatch_mode: "dict | Dispatch", args: list) -> bool:
    """Compute whether data collection is needed for the current worker.

    This function determines whether the current worker should collect data based on
    the dispatch mode configuration and worker parameters. It's used to optimize
    distributed data collection by ensuring only the appropriate rank collects data.

    Args:
        dispatch_mode: Controls data collection logic for the current worker. Can be None,
                      a Dispatch instance, or a dict with 'collect_fn' key. If None or Dispatch,
                      always returns True (current worker should collect). If dict, checks
                      collect_fn for lazy compute optimization.
        args: List of arguments passed to the function. Should contain a Worker instance
             as the first argument when using lazy compute mode.

    Returns:
        bool: True if data collection is needed, False otherwise.

    Note:
        Only checks worker attributes when dispatch_mode is a dict with 'collect_fn',
        the collect_fn is 'collect_lazy_compute_data_proto', and args[0] is a Worker.
        Otherwise, returns True. For the lazy compute case, checks the worker's
        data parallel rank for the mesh specified in collect_fn.args[0] to determine
        if this worker should collect data.
    """
    from verl.single_controller.base.decorator import Dispatch
    from verl.single_controller.base.worker import Worker

    if dispatch_mode is None or isinstance(dispatch_mode, Dispatch):
        return True

    assert "collect_fn" in dispatch_mode.keys(), "collect_fn should be in dispatch_mode."

    collect_fn = dispatch_mode["collect_fn"]

    # Check if collect_fn is a functools.partial and handle gracefully
    if isinstance(collect_fn, functools.partial):
        collect_fn_name = collect_fn.func.__name__
        if collect_fn_name != "collect_lazy_compute_data_proto" or len(args) < 1 or not isinstance(args[0], Worker):
            return True

        collect_mesh_name = collect_fn.args[0] if collect_fn.args else None
        if collect_mesh_name is None:
            return True

        return args[0].query_collect_info(collect_mesh_name)
    else:
        # If collect_fn is not a partial, we can't extract mesh_name information
        # Fall back to default behavior (collect data)
        return True


def _postprocess_common(output, put_data, need_collect):
    """Common post-processing logic for function outputs in TransferQueue bridge.

    This function handles the final return value based on whether data should be
    put into storage (put_data) and whether collection is needed (need_collect).
    It ensures proper return types based on the execution context.

    Args:
        output: The original output from the decorated function. Can be any type.
        put_data: bool, indicating whether the output should be put into TransferQueue.
                 If True, output will be put to TQ and return the corresponding BatchMeta;
                 if False, output will not be put into TQ.
        need_collect: bool, indicating whether this process needs to collect data.
                     If False, the output will be replaced by an empty BatchMeta or DataProto
                     to avoid redundant communication.

    Returns:
        - BatchMeta.empty(): When put_data=True but need_collect=False, indicating
          no data should be stored but BatchMeta structure is expected.
        - DataProto(): When put_data=False, need_collect=False, and output is DataProto,
          returning an empty DataProto.
        - output: In all other cases, returns the original output unchanged.

    Note:
        This function is used in the tqbridge decorator to normalize return values
        across different execution paths and avoid redundant data operations in
        distributed scenarios.
    """
    if put_data and not need_collect:
        return BatchMeta.empty()
    elif not put_data and not need_collect and isinstance(output, DataProto):
        return DataProto()
    else:
        return output


def tqbridge(dispatch_mode: "dict | Dispatch" = None, put_data: bool = True):
    """Creates a decorator for bridging BatchMeta and DataProto.

    This decorator automatically handles conversions between `BatchMeta` and
    `DataProto` in function parameters, and decides whether to sync function
    output back to `BatchMeta` based on configuration(`put_data`). It supports
    both synchronous and asynchronous functions (async def), and can control
    whether to enable enhanced logic via the global `HAS_TQ` variable (when disabled,
    simply calls the original function as-is).

    Args:
        dispatch_mode: Controls data collection behavior for the current worker. Passed to
                      _compute_need_collect to determine if current worker should collect data.
                      If None, _compute_need_collect will return True to fallback default logics.
        put_data: Whether put the DataProto into Storage after func return.
                  If True, after function execution, the output result will be
                  updated to `BatchMeta` and `BatchMeta` will be returned;
                  If False, the function output result will be returned directly.
                  Defaults to True.

    Returns:
        A decorator function used to decorate target functions (synchronous or asynchronous).
    """

    def decorator(func):
        pid = os.getpid()

        @wraps(func)
        def inner(*args, **kwargs):
            batchmeta = _find_batchmeta(*args, **kwargs)
            if batchmeta is None:
                return func(*args, **kwargs)
            else:
                logger.info(
                    f"Task {func.__name__} (pid={pid}) is getting len_samples={batchmeta.size}, "
                    f"global_idx={batchmeta.global_indexes}"
                )
                
                # Profiler: TQ_GET phase
                get_range_id = None
                if TQ_PROFILER_ENABLED:
                    mark_start_range, mark_end_range = _get_profiler_functions()
                    get_range_id = mark_start_range(message=f"TQ_GET:{func.__name__}", color="blue")
                
                args = [_batchmeta_to_dataproto(arg) if isinstance(arg, BatchMeta) else arg for arg in args]
                kwargs = {k: _batchmeta_to_dataproto(v) if isinstance(v, BatchMeta) else v for k, v in kwargs.items()}
                
                if TQ_PROFILER_ENABLED and get_range_id is not None:
                    mark_end_range(get_range_id)
                
                output = func(*args, **kwargs)
                need_collect = _compute_need_collect(dispatch_mode, args)
                if put_data and need_collect:
                    # Profiler: TQ_PUT phase
                    put_range_id = None
                    if TQ_PROFILER_ENABLED:
                        mark_start_range, mark_end_range = _get_profiler_functions()
                        put_range_id = mark_start_range(message=f"TQ_PUT:{func.__name__}", color="green")
                    
                    updated_batch_meta = _update_batchmeta_with_output(output, batchmeta, func.__name__)
                    
                    if TQ_PROFILER_ENABLED and put_range_id is not None:
                        mark_end_range(put_range_id)
                    
                    return updated_batch_meta
                return _postprocess_common(output, put_data, need_collect)

        @wraps(func)
        async def async_inner(*args, **kwargs):
            batchmeta = _find_batchmeta(*args, **kwargs)
            if batchmeta is None:
                return await func(*args, **kwargs)
            else:
                # 始终记录计时数据用于 metrics 收集
                start_time = time.perf_counter()
                collector = TQMetricsCollector.get_instance()
                
                # ========== 日志点1: 函数开始 ==========
                if TQ_TRACE_ENABLED:
                    trace_id = _get_trace_id(batchmeta)
                    logger.info(
                        f"[TQ-TRACE] trace_id={trace_id} | "
                        f"func={func.__name__} | pid={pid} | "
                        f"stage=FUNCTION_START | timestamp={start_time:.6f}"
                    )
                
                logger.info(
                    f"Task {func.__name__} (pid={pid}) is getting len_samples={batchmeta.size}, "
                    f"global_idx={batchmeta.global_indexes}"
                )
                
                # ========== 日志点2: Get数据开始 ==========
                get_start = time.perf_counter()
                if TQ_TRACE_ENABLED:
                    logger.info(
                        f"[TQ-TRACE] trace_id={trace_id} | "
                        f"func={func.__name__} | stage=GET_START | "
                        f"batch_size={batchmeta.size} | timestamp={get_start:.6f}"
                    )
                
                # Profiler: TQ_GET phase
                get_range_id = None
                if TQ_PROFILER_ENABLED:
                    mark_start_range, mark_end_range = _get_profiler_functions()
                    get_range_id = mark_start_range(message=f"TQ_GET:{func.__name__}", color="blue")
                
                args = [await _async_batchmeta_to_dataproto(arg) if isinstance(arg, BatchMeta) else arg for arg in args]
                kwargs = {
                    k: await _async_batchmeta_to_dataproto(v) if isinstance(v, BatchMeta) else v
                    for k, v in kwargs.items()
                }
                
                if TQ_PROFILER_ENABLED and get_range_id is not None:
                    mark_end_range(get_range_id)
                
                # ========== 日志点3: Get数据完成 ==========
                get_end = time.perf_counter()
                get_duration = get_end - get_start
                collector.record_get(get_duration)
                
                if TQ_TRACE_ENABLED:
                    logger.info(
                        f"[TQ-TRACE] trace_id={trace_id} | "
                        f"func={func.__name__} | stage=GET_END | "
                        f"duration={get_duration:.6f}s | timestamp={get_end:.6f}"
                    )
                
                # ========== 日志点4: 业务逻辑开始 ==========
                compute_start = time.perf_counter()
                if TQ_TRACE_ENABLED:
                    logger.info(
                        f"[TQ-TRACE] trace_id={trace_id} | "
                        f"func={func.__name__} | stage=COMPUTE_START | timestamp={compute_start:.6f}"
                    )
                
                output = await func(*args, **kwargs)
                
                # ========== 日志点5: 业务逻辑完成 ==========
                compute_end = time.perf_counter()
                compute_duration = compute_end - compute_start
                collector.record_compute(compute_duration)
                
                if TQ_TRACE_ENABLED:
                    logger.info(
                        f"[TQ-TRACE] trace_id={trace_id} | "
                        f"func={func.__name__} | stage=COMPUTE_END | "
                        f"duration={compute_duration:.6f}s | timestamp={compute_end:.6f}"
                    )
                
                need_collect = _compute_need_collect(dispatch_mode, args)
                if put_data and need_collect:
                    # ========== 日志点6: Put数据开始 ==========
                    put_start = time.perf_counter()
                    if TQ_TRACE_ENABLED:
                        logger.info(
                            f"[TQ-TRACE] trace_id={trace_id} | "
                            f"func={func.__name__} | stage=PUT_START | timestamp={put_start:.6f}"
                        )
                    
                    # Profiler: TQ_PUT phase
                    put_range_id = None
                    if TQ_PROFILER_ENABLED:
                        mark_start_range, mark_end_range = _get_profiler_functions()
                        put_range_id = mark_start_range(message=f"TQ_PUT:{func.__name__}", color="green")
                    
                    updated_batchmeta = await _async_update_batchmeta_with_output(output, batchmeta, func.__name__)
                    
                    if TQ_PROFILER_ENABLED and put_range_id is not None:
                        mark_end_range(put_range_id)
                    
                    # ========== 日志点7: Put数据完成 ==========
                    put_end = time.perf_counter()
                    put_duration = put_end - put_start
                    collector.record_put(put_duration)
                    
                    if TQ_TRACE_ENABLED:
                        logger.info(
                            f"[TQ-TRACE] trace_id={trace_id} | "
                            f"func={func.__name__} | stage=PUT_END | "
                            f"duration={put_duration:.6f}s | timestamp={put_end:.6f}"
                        )
                    
                    # ========== 日志点8: 函数结束 ==========
                    total_duration = put_end - start_time
                    collector.record_total(total_duration)
                    
                    if TQ_TRACE_ENABLED:
                        get_ratio = (get_duration / total_duration * 100) if total_duration > 0 else 0
                        compute_ratio = (compute_duration / total_duration * 100) if total_duration > 0 else 0
                        put_ratio = (put_duration / total_duration * 100) if total_duration > 0 else 0
                        
                        logger.info(
                            f"[TQ-TRACE] trace_id={trace_id} | "
                            f"func={func.__name__} | stage=FUNCTION_END | "
                            f"total_duration={total_duration:.6f}s | "
                            f"get_ratio={get_ratio:.2f}% | "
                            f"compute_ratio={compute_ratio:.2f}% | "
                            f"put_ratio={put_ratio:.2f}% | "
                            f"timestamp={put_end:.6f}"
                        )
                    
                    return updated_batchmeta
                return _postprocess_common(output, put_data, need_collect)

        @wraps(func)
        def dummy_inner(*args, **kwargs):
            output = func(*args, **kwargs)
            return output

        @wraps(func)
        async def dummy_async_inner(*args, **kwargs):
            output = await func(*args, **kwargs)
            return output

        wrapper_inner = inner if is_transferqueue_enabled else dummy_inner
        wrapper_async_inner = async_inner if is_transferqueue_enabled else dummy_async_inner

        wrapper = wrapper_async_inner if inspect.iscoroutinefunction(func) else wrapper_inner
        return wrapper

    return decorator
