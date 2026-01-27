#!/usr/bin/env python3
"""TransferQueue Performance Log Analyzer

该脚本解析TQ追踪日志并生成统计报告，评估TQ数据传递的性能开销。

使用方法:
    python analyze_tq_logs.py <log_file_path>

环境变量:
    TQ_TRACE_ENABLED=1          # 启用函数级追踪日志
    TQ_TRACE_DETAIL_ENABLED=1   # 启用TQ操作详细日志
"""

import re
import sys
from collections import defaultdict
from typing import Dict, Any


def parse_tq_logs(log_file: str) -> Dict[str, Dict[str, Dict[str, Dict[str, str]]]]:
    """解析TQ追踪日志
    
    Args:
        log_file: 日志文件路径
        
    Returns:
        traces: 嵌套字典 {trace_id: {func_name: {stage: metrics_dict}}}
    """
    pattern = r'\[TQ-TRACE\] trace_id=(\S+) \| func=(\S+) \| (.+)'
    
    traces = defaultdict(lambda: defaultdict(dict))
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    trace_id, func, metrics_str = match.groups()
                    
                    # 解析metrics
                    metrics_dict = {}
                    for item in metrics_str.split(' | '):
                        if '=' in item:
                            k, v = item.split('=', 1)
                            metrics_dict[k] = v
                    
                    stage = metrics_dict.get('stage', 'UNKNOWN')
                    traces[trace_id][func][stage] = metrics_dict
    except FileNotFoundError:
        print(f"错误: 找不到日志文件 '{log_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 解析日志文件时出错 - {e}")
        sys.exit(1)
    
    return traces


def analyze_traces(traces: Dict[str, Dict[str, Dict[str, Dict[str, str]]]]) -> Dict[str, Any]:
    """分析追踪数据并生成统计信息
    
    Args:
        traces: 解析后的追踪数据
        
    Returns:
        stats: 统计信息字典
    """
    stats = {
        'total_traces': len(traces),
        'functions': defaultdict(lambda: {
            'count': 0,
            'total_duration': 0.0,
            'get_duration': 0.0,
            'compute_duration': 0.0,
            'put_duration': 0.0,
        })
    }
    
    for trace_id, funcs in traces.items():
        for func, stages in funcs.items():
            if 'FUNCTION_END' in stages:
                # 提取总耗时
                total_dur_str = stages['FUNCTION_END'].get('total_duration', '0s')
                total_dur = float(total_dur_str.rstrip('s'))
                
                stats['functions'][func]['count'] += 1
                stats['functions'][func]['total_duration'] += total_dur
                
                # 计算各阶段耗时
                if 'GET_END' in stages:
                    get_dur_str = stages['GET_END'].get('duration', '0s')
                    get_dur = float(get_dur_str.rstrip('s'))
                    stats['functions'][func]['get_duration'] += get_dur
                
                if 'COMPUTE_END' in stages:
                    compute_dur_str = stages['COMPUTE_END'].get('duration', '0s')
                    compute_dur = float(compute_dur_str.rstrip('s'))
                    stats['functions'][func]['compute_duration'] += compute_dur
                
                if 'PUT_END' in stages:
                    put_dur_str = stages['PUT_END'].get('duration', '0s')
                    put_dur = float(put_dur_str.rstrip('s'))
                    stats['functions'][func]['put_duration'] += put_dur
    
    return stats


def print_report(stats: Dict[str, Any]):
    """打印分析报告
    
    Args:
        stats: 统计信息
    """
    print("=" * 80)
    print("TransferQueue Performance Analysis Report")
    print("=" * 80)
    print(f"\nTotal Traces: {stats['total_traces']}\n")
    
    if not stats['functions']:
        print("没有找到完整的追踪数据。")
        return
    
    # 计算全局TQ开销
    global_total = 0.0
    global_tq = 0.0
    
    for func, metrics in stats['functions'].items():
        count = metrics['count']
        total = metrics['total_duration']
        get = metrics['get_duration']
        compute = metrics['compute_duration']
        put = metrics['put_duration']
        
        global_total += total
        global_tq += (get + put)
        
        print(f"\n{'─' * 80}")
        print(f"Function: {func}")
        print(f"{'─' * 80}")
        print(f"  调用次数:           {count}")
        print(f"  平均总耗时:         {total/count:.6f}s")
        print(f"  平均GET耗时:        {get/count:.6f}s ({get/total*100:.2f}%)" if total > 0 else "  平均GET耗时:        0s")
        print(f"  平均COMPUTE耗时:    {compute/count:.6f}s ({compute/total*100:.2f}%)" if total > 0 else "  平均COMPUTE耗时:    0s")
        print(f"  平均PUT耗时:        {put/count:.6f}s ({put/total*100:.2f}%)" if total > 0 else "  平均PUT耗时:        0s")
        
        tq_overhead = (get + put) / total * 100 if total > 0 else 0
        print(f"  TQ总开销:           {tq_overhead:.2f}%")
    
    print(f"\n{'=' * 80}")
    print("全局汇总")
    print(f"{'=' * 80}")
    global_tq_ratio = (global_tq / global_total * 100) if global_total > 0 else 0
    print(f"  总执行时间:         {global_total:.6f}s")
    print(f"  总TQ时间:           {global_tq:.6f}s")
    print(f"  全局TQ开销占比:     {global_tq_ratio:.2f}%")
    print("=" * 80)


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法: python analyze_tq_logs.py <log_file_path>")
        print("\n示例:")
        print("  python analyze_tq_logs.py /path/to/training.log")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    print(f"正在解析日志文件: {log_file}")
    traces = parse_tq_logs(log_file)
    
    print(f"正在分析 {len(traces)} 条追踪记录...")
    stats = analyze_traces(traces)
    
    print_report(stats)


if __name__ == '__main__':
    main()
