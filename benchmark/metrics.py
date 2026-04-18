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
    # 1. Save CSV
    csv_path = os.path.join(out_dir, 'comparison_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    
    # 2. Save Markdown Report (Dùng cho bài báo)
    report_path = os.path.join(out_dir, 'report.md')
    
    # Tính toán thống kê
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results: grouped[r.method].append(r)
    stats = {}
    for m, rs in grouped.items():
        stats[m] = {
            'tl': statistics.mean([x.avg_timeLoss for x in rs]),
            'wt': statistics.mean([x.avg_waitingTime for x in rs]),
            'den': statistics.mean([x.avg_density for x in rs]),
            'spd': statistics.mean([x.avg_speed for x in rs]),
            'time': statistics.mean([x.time_s for x in rs])
        }

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Bảng So sánh Hiệu năng Giao thông (Cho Bài báo)\n\n")
        f.write("| Metric | Fixed (30/30) | GA | PPO |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        
        metrics = [
            ('Avg timeLoss (s)', 'tl'),
            ('Avg waitingTime (s)', 'wt'),
            ('Avg density (xe/km)', 'den'),
            ('Avg speed (m/s)', 'spd'),
            ('Thời gian tính toán (s)', 'time')
        ]
        
        for label, key in metrics:
            line = f"| {label} "
            for m in ['Fixed', 'GA', 'PPO']:
                val = stats[m][key] if m in stats else 0
                line += f"| {val:.2f} "
            f.write(line + "|\n")
            
        # Dòng cải thiện
        imp_line = "| **Cải thiện so với Fixed** | baseline "
        if 'Fixed' in stats:
            f_tl = stats['Fixed']['tl']
            for m in ['GA', 'PPO']:
                if m in stats:
                    imp = ((f_tl - stats[m]['tl']) / f_tl) * 100
                    imp_line += f"| **{imp:+.1f}%** "
                else: imp_line += "| - "
        f.write(imp_line + "|\n")

    print(f"Ket qua CSV: {csv_path}")
    print(f"Bao cao Markdown (Paper format): {report_path}")
    return csv_path


def print_summary(results: List[TrialResult]):
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[r.method].append({
            'tl': r.avg_timeLoss,
            'wt': r.avg_waitingTime,
            'den': r.avg_density,
            'spd': r.avg_speed,
            'time': r.time_s
        })

    # Tính toán mean cho từng method
    stats = {}
    for method, runs in grouped.items():
        stats[method] = {k: statistics.mean([r[k] for r in runs]) for k in ['tl', 'wt', 'den', 'spd', 'time']}

    if 'Fixed' not in stats:
        print("Error: Missing 'Fixed' method for comparison.")
        return

    # In bảng ra Terminal theo yêu cầu của USER (Dòng là Metric, Cột là Method)
    methods = ['Fixed', 'GA', 'PPO']
    active_methods = [m for m in methods if m in stats]
    
    print("\n" + "="*70)
    print(f"{'Metric':<25} | {'Fixed (30/30)':>12} | {'GA':>12} | {'PPO':>12}")
    print("-" * 70)
    
    metrics_map = {
        'Avg timeLoss (s)': 'tl',
        'Avg waitingTime (s)': 'wt',
        'Avg density (xe/km)': 'den',
        'Avg speed (m/s)': 'spd',
        'Computing Time (s)': 'time'
    }

    for label, key in metrics_map.items():
        row = f"{label:<25} | "
        for m in active_methods:
            val = stats[m][key]
            row += f"{val:>12.2f} | "
        print(row)
    
    print("-" * 70)
    # Dòng cải thiện (Dựa trên timeLoss - càng thấp càng tốt)
    imp_row = f"{'Improvement vs Fixed':<25} | "
    fixed_tl = stats['Fixed']['tl']
    for m in active_methods:
        if m == 'Fixed':
            imp_row += f"{'baseline':>12} | "
        else:
            imp = ((fixed_tl - stats[m]['tl']) / fixed_tl) * 100
            imp_row += f"{imp:>+11.1f}% | "
    print(imp_row)
    print("="*70 + "\n")
