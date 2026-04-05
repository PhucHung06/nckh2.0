# benchmark/metrics.py
"""Tính toán và lưu trữ kết quả benchmark. Độc lập với hardware và RL framework."""
import csv, os, statistics
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class TrialResult:
    trial_id:        int
    method:          str    # 'GA', 'PPO', 'DQN', 'Fixed'
    fitness:         float
    chromosome:      list
    time_s:          float
    # Chỉ số chi tiết từ SUMO output (bổ sung sau nếu cần)
    avg_timeLoss:    float = 0.0
    avg_waitingTime: float = 0.0
    avg_density:     float = 0.0
    avg_speed:       float = 0.0


def save_results(results: List[TrialResult], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'comparison_results.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"Ket qua luu: {path}")
    return path


def print_summary(results: List[TrialResult]):
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[r.method].append(r.fitness)

    print("\n=== Thong ke tom tat ===")
    print(f"{'Method':<12} {'Mean Fitness':>14} {'Std':>10} {'Trials':>8}")
    print("-" * 46)
    for method, scores in grouped.items():
        mean = statistics.mean(scores)
        std  = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f"{method:<12} {mean:>14.4f} {std:>10.4f} {len(scores):>8}")
