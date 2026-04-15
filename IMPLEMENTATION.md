# IMPLEMENTATION GUIDE
## Tối ưu đèn giao thông: GA → Deep RL → (Phần cứng sau)

> **Chiến lược hiện tại:** Tập trung hoàn thiện **Hướng 2 (Deep RL + Benchmark)** trên PC/laptop trước. Toàn bộ code được thiết kế theo kiến trúc **interface-based** — khi có phần cứng, chỉ cần cắm thêm module `hardware/` vào mà không sửa gì ở phần lõi.

---

## Cấu trúc cây thư mục

```
traffic-light-optimization/
│
├── README.md
├── IMPLEMENTATION.md                    # File này
├── requirements.txt                     # PC/laptop (đang dùng)
├── requirements_rpi.txt                 # Raspberry Pi 5 (dùng sau)
│
├── data/                                # Dữ liệu SUMO — giữ nguyên từ dự án cũ
│   ├── run1.sumocfg
│   ├── ngatu_net.xml
│   ├── ngatu_net_rou.xml
│   ├── time_light.xml
│   ├── dulieu_matdo.xml
│   └── thongke.xml
│
├── simulation/                          # Lõi mô phỏng SUMO — dùng chung cả 2 hướng
│   ├── __init__.py
│   ├── sumo_env.py                      # Giữ nguyên từ dự án cũ
│   ├── ga.py                            # Giữ nguyên từ dự án cũ
│   ├── sumo_gym_env.py                  # [MỚI] Gymnasium wrapper cho RL
│   └── main_ga.py                       # Entry point GA (cập nhật import path)
│
├── rl/                                  # ★ ĐANG LÀM — Deep Reinforcement Learning
│   ├── __init__.py
│   ├── train_ppo.py                     # Huấn luyện PPO agent
│   ├── train_dqn.py                     # Huấn luyện DQN agent (tùy chọn)
│   ├── evaluate_rl.py                   # Đánh giá model đã train
│   └── models/                          # Checkpoint model lưu tại đây
│       ├── ppo_traffic/
│       └── dqn_traffic/
│
├── benchmark/                           # ★ ĐANG LÀM — So sánh GA vs RL
│   ├── __init__.py
│   ├── run_comparison.py                # Chạy benchmark song song
│   ├── metrics.py                       # Tính các chỉ số đánh giá
│   ├── visualize_results.py             # Vẽ biểu đồ so sánh
│   └── results/                         # Output CSV + ảnh biểu đồ
│       ├── ga_results.csv
│       ├── rl_results.csv
│       └── comparison_plots/
│
├── hardware/                            # ⏳ DÙNG SAU — Raspberry Pi 5 + Arduino
│   ├── __init__.py
│   ├── base_controller.py               # [THIẾT KẾ SẴN] Abstract interface
│   ├── mock_controller.py               # [THIẾT KẾ SẴN] Giả lập Pi5 để test trên PC
│   ├── pi_controller.py                 # [THIẾT KẾ SẴN] Controller thật trên Pi5
│   ├── arduino_serial.py                # [THIẾT KẾ SẴN] Giao tiếp Serial
│   ├── sensor_reader.py                 # [THIẾT KẾ SẴN] Camera/cảm biến
│   └── arduino/
│       ├── traffic_light.ino            # [THIẾT KẾ SẴN] Firmware Arduino
│       └── serial_protocol.md           # Mô tả giao thức SET/ACK/STATUS
│
├── tests/
│   ├── test_ga.py
│   ├── test_sumo_env.py
│   ├── test_rl_env.py
│   └── test_hardware_mock.py            # Test hardware KHÔNG cần Pi thật
│
├── notebooks/
│   ├── 01_ga_analysis.ipynb
│   ├── 02_rl_training_curves.ipynb
│   └── 03_comparison_report.ipynb
│
└── docs/
    ├── architecture.md
    ├── hardware_setup.md
    └── paper_notes.md
```

> **Ghi chú thư mục `hardware/`:** Toàn bộ folder này được **thiết kế sẵn kiến trúc** ngay từ bây giờ (abstract interface + mock controller), nhưng **chưa cần chạy**. Khi có Pi5 và Arduino, chỉ cần implement `pi_controller.py` và `arduino_serial.py` thật — toàn bộ luồng còn lại hoạt động ngay.

---

## Lộ trình triển khai

```
HIỆN TẠI                                         SAU NÀY
──────────────────────────────────────           ───────────────────────────
Phase 1 (Tuần 1)    → Baseline GA           →   Phase 4 (Sau khi có HW)
Phase 2A (Tuần 2-3) → RL Agent              →   Cắm Pi5 + Arduino vào
Phase 2B (Tuần 4)   → Benchmark             →   Chạy hardware/pi_controller.py
Phase 3 (Tuần 5)    → Bài báo / Báo cáo    →   Demo phần cứng hoàn chỉnh
```

---

## PHASE 1 — Tái cấu trúc & Baseline GA (Tuần 1)

**Mục tiêu:** Đóng gói code cũ thành module chuẩn, ghi nhận kết quả GA làm baseline để so sánh.

### Bước 1.1 — Cài đặt môi trường

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

**`requirements.txt`:**
```
#Yolo
opencv-python>=4.8.0
ultralytics>=8.0.0
supervision>=0.18.0

# SUMO Python API
traci>=1.18.0
sumolib>=1.18.0

# Deep RL
stable-baselines3>=2.3.0
gymnasium>=0.29.0
torch>=2.2.0

# Phân tích & Benchmark
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
seaborn>=0.13.0
scipy>=1.12.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.0.0

# Hardware (cài sẵn, dùng sau — không ảnh hưởng phần đang làm)
paho-mqtt>=2.0.0
pyserial>=3.5
pyyaml>=6.0
```

**`requirements_rpi.txt`** *(dùng khi có Pi5)*:
```
# Subset nhẹ hơn cho Pi5, không cần torch đầy đủ
traci>=1.18.0
paho-mqtt>=2.0.0
pyserial>=3.5
pyyaml>=6.0
opencv-python-headless>=4.8.0
numpy>=1.26.0
```

### Bước 1.2 — Tái cấu trúc code cũ

```bash
# Tạo cấu trúc thư mục
mkdir -p simulation rl benchmark
mkdir -p hardware/arduino hardware/config
mkdir -p tests notebooks docs
mkdir -p rl/models/ppo_traffic rl/models/dqn_traffic
mkdir -p benchmark/results/comparison_plots

# Di chuyển file cũ vào đúng vị trí
cp ga.py simulation/ga.py
cp sumo_env.py simulation/sumo_env.py
cp main.py simulation/main_ga.py

# Tạo __init__.py
touch simulation/__init__.py rl/__init__.py benchmark/__init__.py hardware/__init__.py
```

Cập nhật phần đầu `simulation/main_ga.py` (logic bên trong giữ nguyên):

```python
# simulation/main_ga.py — chỉ sửa import paths, không đổi logic
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulation.sumo_env import SumoEnvironment
from simulation.ga import GeneticAlgorithm

DATA_DIR       = os.path.join(os.path.dirname(__file__), '..', 'data')
SUMO_CFG       = os.path.join(DATA_DIR, 'run1.sumocfg')
TIME_LIGHT_XML = os.path.join(DATA_DIR, 'time_light.xml')
OUTPUT_XML     = os.path.join(DATA_DIR, 'dulieu_matdo.xml')

# Phần còn lại KHÔNG thay đổi
```

### Bước 1.3 — Chạy GA và lưu baseline

```bash
python simulation/main_ga.py
```

Ghi lại fitness tốt nhất và bộ gen kết quả — đây là **baseline** cho bài báo.

**Checklist Phase 1:**
- [x] GA chạy không lỗi, SUMO mô phỏng thành công
- [x] `dulieu_matdo.xml` sinh ra sau mỗi lần chạy
- [x] Fitness cải thiện qua các thế hệ (ghi lại số liệu)
- [x] Module imports hoạt động đúng từ thư mục gốc

---

## PHASE 2A — Deep Reinforcement Learning (Tuần 2–3)

**Mục tiêu:** Xây dựng PPO agent sử dụng cùng môi trường SUMO, huấn luyện và lưu model.

### Bước 2.1 — Gymnasium Environment Wrapper

Tạo `simulation/sumo_gym_env.py`:

```python
# simulation/sumo_gym_env.py
"""
Biến SumoEnvironment thành Gymnasium-compatible environment cho RL.

Thiết kế:
- Observation: [g_ns, y_ns, g_ew, y_ew] chuẩn hóa về [0, 1]
- Action:      Discrete(9) — 9 preset thời gian đèn định sẵn
- Reward:      Fitness score từ hàm evaluate() của SumoEnvironment
- Terminated:  True sau mỗi bước (1 episode = 1 lần mô phỏng SUMO ~10s)

Lý do dùng action preset thay vì continuous:
  Không gian [5,90]x[3,5]x[5,90]x[3,5] quá lớn cho DQN.
  9 preset bao phủ các cấu hình thực tế phổ biến.
  Mở rộng thêm preset sau mà không sửa interface.
"""
import gymnasium as gym
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from simulation.sumo_env import SumoEnvironment


class SumoGymEnv(gym.Env):
    metadata = {'render_modes': []}

    # 9 preset (GreenNS, GreenEW) — Yellow cố định 4s
    # Thêm preset tại đây nếu muốn mở rộng không gian hành động
    ACTION_PRESETS = [
        (10, 10), (20, 20), (30, 30),
        (40, 40), (50, 50), (60, 60),
        (30, 60), (60, 30), (45, 45),
    ]
    YELLOW = 4

    def __init__(self, sumocfg: str, time_light_xml: str, output_xml: str):
        super().__init__()
        self._sumo = SumoEnvironment(sumocfg, time_light_xml, output_xml)
        self._current_chromosome = [30, self.YELLOW, 30, self.YELLOW]

        # Bounds khớp với ga.py để chuẩn hóa nhất quán
        self._bounds = [(5, 90), (3, 5), (5, 90), (3, 5)]

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(self.ACTION_PRESETS))

    def _to_obs(self, chromosome) -> np.ndarray:
        return np.array([
            (chromosome[i] - self._bounds[i][0]) /
            (self._bounds[i][1] - self._bounds[i][0])
            for i in range(4)
        ], dtype=np.float32)

    def _action_to_chromosome(self, action: int) -> list:
        g_ns, g_ew = self.ACTION_PRESETS[action]
        return [g_ns, self.YELLOW, g_ew, self.YELLOW]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_chromosome = [30, self.YELLOW, 30, self.YELLOW]
        return self._to_obs(self._current_chromosome), {}

    def step(self, action: int):
        chromosome = self._action_to_chromosome(action)
        reward = self._sumo.evaluate(chromosome)
        self._current_chromosome = chromosome
        obs = self._to_obs(chromosome)
        return obs, reward, True, False, {'chromosome': chromosome}

    def render(self):
        pass

    def close(self):
        pass

    def get_best_chromosome(self) -> list:
        """Trả về chromosome hiện tại — dùng để export sang Pi5 sau này."""
        return self._current_chromosome.copy()
```

### Bước 2.2 — Huấn luyện PPO Agent

Tạo `rl/train_ppo.py`:

```python
# rl/train_ppo.py
"""
Huấn luyện PPO agent tối ưu thời gian đèn giao thông.

Cấu hình mặc định phù hợp với SUMO ~10s/step:
  - n_steps=64, batch_size=16: mini-batch nhỏ để không tốn RAM
  - total_timesteps=500: khoảng 500 lần chạy SUMO (~1-2h)
  Tăng lên 2000+ khi có thời gian để model hội tụ tốt hơn.

Output:
  rl/models/ppo_traffic/final_model.zip     -- dùng cho benchmark
  rl/logs/                                  -- TensorBoard logs
  hardware/config/best_chromosome_rl.json   -- export cho Pi5 sau này
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from simulation.sumo_gym_env import SumoGymEnv

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'ppo_traffic')
LOG_DIR   = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def make_env():
    return Monitor(SumoGymEnv(
        sumocfg        = os.path.join(DATA_DIR, 'run1.sumocfg'),
        time_light_xml = os.path.join(DATA_DIR, 'time_light.xml'),
        output_xml     = os.path.join(DATA_DIR, 'dulieu_matdo.xml'),
    ))


def main():
    env = make_env()

    model = PPO(
        policy          = 'MlpPolicy',
        env             = env,
        learning_rate   = 3e-4,
        n_steps         = 64,    # Nhỏ vì mỗi step tốn ~10s
        batch_size      = 16,
        n_epochs        = 5,
        gamma           = 0.99,
        ent_coef        = 0.01,  # Khuyến khích khám phá
        verbose         = 1,
        tensorboard_log = LOG_DIR,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = 50,
        save_path   = MODEL_DIR,
        name_prefix = 'ppo_traffic',
    )

    print("Bat dau huan luyen PPO...")
    print("Theo doi: tensorboard --logdir rl/logs")
    model.learn(
        total_timesteps = 500,   # Tang len 2000+ de model hoi tu tot hon
        callback        = checkpoint_cb,
        progress_bar    = True,
    )

    final_path = os.path.join(MODEL_DIR, 'final_model')
    model.save(final_path)
    print(f"Model da luu: {final_path}.zip")

    # Tự động export chromosome tốt nhất cho hardware sau này
    _export_best_chromosome(model, env)


def _export_best_chromosome(model, env):
    """
    Chạy model inference, tìm chromosome cho fitness cao nhất,
    lưu ra JSON để hardware/pi_controller.py đọc khi có Pi5.
    """
    import json
    obs, _ = env.reset()
    best_fitness, best_chromosome = -float('inf'), None

    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, info = env.step(int(action))
        if reward > best_fitness:
            best_fitness = reward
            best_chromosome = info['chromosome']

    export = {
        'chromosome':  best_chromosome,
        'fitness':     best_fitness,
        'method':      'PPO',
        'description': '[GreenNS, YellowNS, GreenEW, YellowEW] in seconds'
    }
    out_path = os.path.join(os.path.dirname(__file__), '..', 'hardware',
                            'config', 'best_chromosome_rl.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(export, f, indent=2)
    print(f"Best RL chromosome: {best_chromosome} (fitness={best_fitness:.4f})")
    print(f"Da luu cho hardware: {out_path}")


if __name__ == '__main__':
    main()
```

### Bước 2.3 — Theo dõi training

```bash
# Terminal 1 — Mở TensorBoard
tensorboard --logdir rl/logs
# Truy cập: http://localhost:6006

# Terminal 2 — Bắt đầu train
python rl/train_ppo.py
```

**Các metric cần theo dõi:**

| Metric | Mong đợi | Hành động nếu lệch |
|--------|----------|---------------------|
| `ep_rew_mean` | Tăng dần, ổn định | Giảm `learning_rate` nếu dao động |
| `policy_loss` | Giảm ổn định | Tăng `n_epochs` |
| `entropy_loss` | Giảm chậm | Tăng `ent_coef` nếu giảm quá nhanh |
| `explained_variance` | Tiến gần 1.0 | Tăng `n_steps` nếu thấp |

**Checklist Phase 2A:**
- [x] `SumoGymEnv` import và chạy không lỗi
- [x] PPO train được ít nhất 100 timesteps
- [x] `rl/models/ppo_traffic/final_model.zip` tồn tại
- [x] `hardware/config/best_chromosome_rl.json` được tạo ra tự động
- [x] TensorBoard hiển thị `ep_rew_mean` tăng theo timesteps

---

## PHASE 2B — Benchmark GA vs RL (Tuần 4)

**Mục tiêu:** So sánh khách quan 3 phương án (Fixed / GA / PPO) để lấy số liệu bài báo.

### Bước 2.4 — Module metrics

Tạo `benchmark/metrics.py`:

```python
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
```

### Bước 2.5 — Script benchmark chính

Tạo `benchmark/run_comparison.py`:

```python
# benchmark/run_comparison.py
"""
Chạy benchmark so sánh Fixed-timing vs GA vs PPO.
Fixed (30s/30s cố định) là baseline tham chiếu cho bài báo.

Cách dùng:
  python benchmark/run_comparison.py
  python benchmark/run_comparison.py --trials 20 --ga-gens 15
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulation.sumo_env import SumoEnvironment
from simulation.ga import GeneticAlgorithm
from simulation.sumo_gym_env import SumoGymEnv
from stable_baselines3 import PPO
from benchmark.metrics import TrialResult, save_results, print_summary

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'rl',
                          'models', 'ppo_traffic', 'final_model')


def run_fixed_trial(env: SumoEnvironment, trial_id: int) -> TrialResult:
    """Baseline cố định: 30s NS / 4s Yellow / 30s EW / 4s Yellow."""
    chromosome = [30, 4, 30, 4]
    start = time.time()
    fitness = env.evaluate(chromosome)
    return TrialResult(trial_id=trial_id, method='Fixed',
                       fitness=fitness, chromosome=chromosome,
                       time_s=time.time() - start)


def run_ga_trial(env: SumoEnvironment, trial_id: int,
                 n_gens: int = 10) -> TrialResult:
    ga = GeneticAlgorithm(pop_size=6, mutation_rate=0.1)
    ga.init_population()
    best_fitness, best_chromosome = -float('inf'), None
    start = time.time()

    for _ in range(n_gens):
        pop_fit = [(c, env.evaluate(c)) for c in ga.population]
        pop_fit.sort(key=lambda x: x[1], reverse=True)
        if pop_fit[0][1] > best_fitness:
            best_fitness, best_chromosome = pop_fit[0][1], pop_fit[0][0]
        new_pop = [pop_fit[0][0]]
        while len(new_pop) < 6:
            p1 = ga.selection(pop_fit)
            p2 = ga.selection(pop_fit)
            c1, c2 = ga.crossover(p1, p2)
            new_pop.extend([ga.mutate(c1), ga.mutate(c2)])
        ga.population = new_pop[:6]

    return TrialResult(trial_id=trial_id, method='GA',
                       fitness=best_fitness, chromosome=best_chromosome,
                       time_s=time.time() - start)


def run_rl_trial(model: PPO, gym_env: SumoGymEnv,
                 trial_id: int) -> TrialResult:
    start = time.time()
    obs, _ = gym_env.reset()
    best_fitness, best_chromosome = -float('inf'), None

    for _ in range(20):    # 20 inference steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, info = gym_env.step(int(action))
        if reward > best_fitness:
            best_fitness = reward
            best_chromosome = info['chromosome']

    return TrialResult(trial_id=trial_id, method='PPO',
                       fitness=best_fitness, chromosome=best_chromosome,
                       time_s=time.time() - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials',  type=int, default=10)
    parser.add_argument('--ga-gens', type=int, default=10)
    args = parser.parse_args()

    env = SumoEnvironment(
        os.path.join(DATA_DIR, 'run1.sumocfg'),
        os.path.join(DATA_DIR, 'time_light.xml'),
        os.path.join(DATA_DIR, 'dulieu_matdo.xml'),
    )
    gym_env = SumoGymEnv(
        os.path.join(DATA_DIR, 'run1.sumocfg'),
        os.path.join(DATA_DIR, 'time_light.xml'),
        os.path.join(DATA_DIR, 'dulieu_matdo.xml'),
    )
    model = PPO.load(MODEL_PATH)

    results = []
    for i in range(1, args.trials + 1):
        print(f"\n[Trial {i}/{args.trials}]")
        results.append(run_fixed_trial(env, i))
        results.append(run_ga_trial(env, i, n_gens=args.ga_gens))
        results.append(run_rl_trial(model, gym_env, i))

    save_results(results, RESULT_DIR)
    print_summary(results)


if __name__ == '__main__':
    main()
```

### Bước 2.6 — Vẽ biểu đồ so sánh

Tạo `benchmark/visualize_results.py`:

```python
# benchmark/visualize_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
PLOT_DIR   = os.path.join(RESULT_DIR, 'comparison_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

df      = pd.read_csv(os.path.join(RESULT_DIR, 'comparison_results.csv'))
ORDER   = ['Fixed', 'GA', 'PPO']
PALETTE = {'Fixed': '#9E9E9E', 'GA': '#2196F3', 'PPO': '#FF9800'}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Fixed vs GA vs PPO — Toi uu den giao thong',
             fontsize=13, fontweight='bold')

sns.boxplot(data=df, x='method', y='fitness', order=ORDER,
            palette=PALETTE, ax=axes[0])
axes[0].set_title('Fitness Score (cao hon = tot hon)')
axes[0].set_xlabel('Phuong phap')
axes[0].set_ylabel('Fitness Score')

sns.boxplot(data=df, x='method', y='time_s', order=ORDER,
            palette=PALETTE, ax=axes[1])
axes[1].set_title('Thoi gian tinh toan (giay)')
axes[1].set_xlabel('Phuong phap')
axes[1].set_ylabel('Thoi gian (s)')

plt.tight_layout()
out = os.path.join(PLOT_DIR, 'comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Bieu do luu: {out}")
plt.show()

print("\n=== Thong ke ===")
print(df.groupby('method')[['fitness', 'time_s']].agg(['mean', 'std']).round(4))
```

**Chạy benchmark và sinh số liệu báo cáo:**

```bash
# 1. Chay benchmark so sanh 3 phuong phap
python benchmark/run_comparison.py --trials 5             # Nhanh để kiểm thử
python benchmark/run_comparison.py --trials 10 --ga-gens 15  # Đầy đủ cho bài báo
python benchmark/visualize_results.py

# 2. Ve bieu do hoi tu cua Thuật toán di truyền (GA)
python benchmark/plot_ga_curve.py

# 3. Ve bieu do hoc cua Deep RL PPO (Learning curve tu TensorBoard)
python benchmark/plot_ppo_curve.py
```

**Checklist Phase 2B:**
- [x] `benchmark/results/comparison_results.csv` có đủ 3 method
- [x] Biểu đồ boxplot render được, lưu vào `comparison_plots/`
- [x] GA và PPO đều tốt hơn Fixed (fitness cao hơn)
- [x] Có mean ± std để điền vào bảng bài báo

---

## PHASE 3 — Bài báo & Báo cáo NCKH (Tuần 5)

### Bảng số liệu cần thu thập

| Thông số | Fixed (30s/30s) | GA | PPO |
|----------|----------------|----|-----|
| Avg fitness | ? | ? | ? |
| Avg timeLoss (s) | ? | ? | ? |
| Avg waitingTime (s) | ? | ? | ? |
| Avg density | ? | ? | ? |
| Avg speed (m/s) | ? | ? | ? |
| Thời gian tính toán (s) | — | ? | ? |
| Bộ gen tốt nhất | [30,4,30,4] | ? | ? |

> **Tip:** Bổ sung parse chi tiết `dulieu_matdo.xml` trong `TrialResult` (timeLoss, waitingTime, density, speed riêng biệt) để có bảng số liệu đầy đủ hơn cho bài báo.

### Cấu trúc bài báo đề xuất

1. **Abstract** — Kết quả: GA cải thiện X% / PPO cải thiện Y% so với fixed timing
2. **Introduction** — Bài toán tắc nghẽn, lý do cần tối ưu thích nghi
3. **Related Work** — SUMO + GA, SUMO + RL trong nghiên cứu trước
4. **Methodology** — Hàm Fitness, kiến trúc GA, kiến trúc PPO (MlpPolicy)
5. **Experiments** — Cấu hình SUMO: 1800 xe/h, 3600s, 1 ngã tư
6. **Results** — Bảng số liệu + biểu đồ từ `benchmark/results/`
7. **Hardware Demo** *(thêm sau khi có Pi5)* — Pi5 + Arduino
8. **Conclusion** — Kết luận, hướng mở rộng (multi-intersection, MARL)

---

## PHASE 4 — Hardware Deployment *(Làm sau khi có Pi5 + Arduino)*

> **Kiến trúc đã được thiết kế sẵn.** Khi có phần cứng, chỉ cần implement 2 file và code GA/RL ở trên hoạt động ngay — không cần sửa gì.

### Tại sao thiết kế theo interface?

```python
# Hiện tại — test trên PC:
from hardware.mock_controller import MockController
ctrl = MockController()

# Sau này — deploy lên Pi5 (chỉ đổi 1 dòng này):
from hardware.pi_controller import PiController
ctrl = PiController()

# Toàn bộ code GA/RL gọi cùng interface — KHÔNG đổi gì khác
ctrl.apply_chromosome(best_chromosome)
```

### Luồng dữ liệu tổng thể

```
[PC -- Đang làm]                          [Pi5 -- Dùng sau]
  SUMO simulation                           Đọc best_chromosome_rl.json
       |                                           |
  GA / PPO train                           PiController.apply_chromosome()
       |                                           |
  best_chromosome_rl.json  ---------->     Arduino SET:gns:yns:gew:yew
  final_model.zip                                  |
       |                                     LED ngã tư mô hình
  benchmark → bài báo
```

### Abstract interface (tạo ngay bây giờ)

Tạo `hardware/base_controller.py`:

```python
# hardware/base_controller.py
from abc import ABC, abstractmethod


class BaseController(ABC):
    """Interface chung — mọi controller đều implement."""

    @abstractmethod
    def send_timing(self, green_ns: int, yellow_ns: int,
                    green_ew: int, yellow_ew: int) -> bool:
        """Gửi bộ thời gian đèn. Trả về True nếu thành công."""

    @abstractmethod
    def get_status(self) -> dict:
        """Lấy trạng thái hiện tại của hệ thống đèn."""

    def apply_chromosome(self, chromosome: list) -> bool:
        """Tiện ích: áp dụng trực tiếp từ chromosome [g_ns,y_ns,g_ew,y_ew]."""
        g_ns, y_ns, g_ew, y_ew = chromosome
        return self.send_timing(g_ns, y_ns, g_ew, y_ew)
```

Tạo `hardware/mock_controller.py` — test trên PC không cần Pi5:

```python
# hardware/mock_controller.py
"""Giả lập Pi5 Controller — chạy hoàn toàn trên PC, in ra console."""
from hardware.base_controller import BaseController


class MockController(BaseController):

    def __init__(self):
        self._current = {'green_ns': 30, 'yellow_ns': 4,
                         'green_ew': 30, 'yellow_ew': 4}
        print("[MockController] Khoi tao OK — chay tren PC (khong can Pi5)")

    def send_timing(self, green_ns, yellow_ns, green_ew, yellow_ew) -> bool:
        self._current = dict(green_ns=green_ns, yellow_ns=yellow_ns,
                             green_ew=green_ew, yellow_ew=yellow_ew)
        print(f"[MockController] SET: NS={green_ns}s/{yellow_ns}s | "
              f"EW={green_ew}s/{yellow_ew}s  -->  ACK:OK (simulated)")
        return True

    def get_status(self) -> dict:
        return {'status': 'mock_running', **self._current}
```

### Test mock ngay bây giờ

Tạo `tests/test_hardware_mock.py`:

```python
# tests/test_hardware_mock.py
"""Test luồng GA/RL -> Hardware với MockController trên PC."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from hardware.mock_controller import MockController


def test_mock_controller():
    ctrl = MockController()
    assert ctrl.send_timing(30, 4, 45, 4) == True
    assert ctrl.apply_chromosome([45, 4, 35, 4]) == True
    status = ctrl.get_status()
    assert 'green_ns' in status
    print("PASSED — MockController hoat dong dung")


if __name__ == '__main__':
    test_mock_controller()
```

```bash
# Chạy ngay bây giờ (không cần Pi5)
python tests/test_hardware_mock.py
```

### Pi5 Controller *(implement khi có phần cứng)*

Tạo `hardware/pi_controller.py`:

```python
# hardware/pi_controller.py
"""Controller thật trên Raspberry Pi 5. Drop-in thay thế MockController."""
import serial, time, json, os
from hardware.base_controller import BaseController

SERIAL_PORT = '/dev/ttyUSB0'   # Kiểm tra: ls /dev/tty* | grep -E 'USB|ACM'
BAUD_RATE   = 9600


class PiController(BaseController):

    def __init__(self, port=SERIAL_PORT, baud=BAUD_RATE):
        self.ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)   # Đợi Arduino khởi động lại
        print(f"[PiController] Ket noi Arduino tai {port} -- OK")

    def send_timing(self, green_ns, yellow_ns, green_ew, yellow_ew) -> bool:
        cmd = f"SET:{green_ns}:{yellow_ns}:{green_ew}:{yellow_ew}\n"
        self.ser.write(cmd.encode())
        resp = self.ser.readline().decode().strip()
        ok = (resp == "ACK:OK")
        print(f"[Pi] {cmd.strip()}  -->  {'OK' if ok else 'LOI: ' + resp}")
        return ok

    def get_status(self) -> dict:
        self.ser.write(b"STATUS\n")
        raw = self.ser.readline().decode().strip()
        parts = raw.split(':')
        return {'raw': raw,
                'phase':      parts[1] if len(parts) > 1 else '?',
                'elapsed_ms': parts[2] if len(parts) > 2 else '?'}


def main():
    BEST_PATH = os.path.join(os.path.dirname(__file__),
                             'config', 'best_chromosome_rl.json')
    ctrl = PiController()

    if os.path.exists(BEST_PATH):
        with open(BEST_PATH) as f:
            data = json.load(f)
        print(f"[Pi] Nap: {data['chromosome']} (method={data['method']})")
        ctrl.apply_chromosome(data['chromosome'])
    else:
        print("[Pi] Chua co best_chromosome_rl.json -- dung mac dinh 30/4/30/4")
        ctrl.send_timing(30, 4, 30, 4)


if __name__ == '__main__':
    main()
```

### Firmware Arduino

Tạo `hardware/arduino/traffic_light.ino`:

```cpp
// hardware/arduino/traffic_light.ino
// Giao thuc: "SET:gns:yns:gew:yew\n"  -->  "ACK:OK\n"
//            "STATUS\n"               -->  "STATUS:<phase>:<elapsed_ms>\n"

const int NS_GREEN=3, NS_YELLOW=4, NS_RED=5;
const int EW_GREEN=9, EW_YELLOW=10, EW_RED=11;

enum Phase { NS_GO, NS_YELLOW_PH, EW_GO, EW_YELLOW_PH };
Phase currentPhase = NS_GO;
int greenNS=30, yellowNS=4, greenEW=30, yellowEW=4;
unsigned long phaseStart = 0;

void setup() {
  Serial.begin(9600);
  int pins[] = {NS_GREEN, NS_YELLOW, NS_RED,
                EW_GREEN, EW_YELLOW, EW_RED};
  for (int p : pins) pinMode(p, OUTPUT);
  setPhase(NS_GO);
  phaseStart = millis();
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.startsWith("SET:")) {
      int vals[4]; int idx = 0;
      String s = cmd.substring(4);
      while (s.length() > 0 && idx < 4) {
        int sep = s.indexOf(':');
        vals[idx++] = (sep < 0 ? s : s.substring(0, sep)).toInt();
        if (sep < 0) break;
        s = s.substring(sep + 1);
      }
      greenNS=vals[0]; yellowNS=vals[1]; greenEW=vals[2]; yellowEW=vals[3];
      Serial.println("ACK:OK");
    }
    else if (cmd == "STATUS") {
      Serial.print("STATUS:"); Serial.print(currentPhase);
      Serial.print(":"); Serial.println(millis() - phaseStart);
    }
  }

  unsigned long elapsed = (millis() - phaseStart) / 1000UL;
  int durations[] = {greenNS, yellowNS, greenEW, yellowEW};
  if (elapsed >= (unsigned long)durations[currentPhase]) {
    currentPhase = (Phase)((currentPhase + 1) % 4);
    setPhase(currentPhase);
    phaseStart = millis();
  }
}

void setPhase(Phase p) {
  digitalWrite(NS_GREEN,0); digitalWrite(NS_YELLOW,0); digitalWrite(NS_RED,0);
  digitalWrite(EW_GREEN,0); digitalWrite(EW_YELLOW,0); digitalWrite(EW_RED,0);
  switch (p) {
    case NS_GO:        digitalWrite(NS_GREEN,1);  digitalWrite(EW_RED,1);  break;
    case NS_YELLOW_PH: digitalWrite(NS_YELLOW,1); digitalWrite(EW_RED,1);  break;
    case EW_GO:        digitalWrite(EW_GREEN,1);  digitalWrite(NS_RED,1);  break;
    case EW_YELLOW_PH: digitalWrite(EW_YELLOW,1); digitalWrite(NS_RED,1);  break;
  }
}
```

### Sơ đồ kết nối Arduino *(chuẩn bị sẵn)*

```
Raspberry Pi 5 ──── USB Cable ────► Arduino Uno
                                         |
                         Pin 3  ── [220Ω] ── LED Xanh  (Bắc-Nam)  ── GND
                         Pin 4  ── [220Ω] ── LED Vàng  (Bắc-Nam)  ── GND
                         Pin 5  ── [220Ω] ── LED Đỏ    (Bắc-Nam)  ── GND
                         Pin 9  ── [220Ω] ── LED Xanh  (Đông-Tây) ── GND
                         Pin 10 ── [220Ω] ── LED Vàng  (Đông-Tây) ── GND
                         Pin 11 ── [220Ω] ── LED Đỏ    (Đông-Tây) ── GND
```

### Copy kết quả sang Pi5 *(khi có phần cứng)*

```bash
# Trên PC: copy chromosome tốt nhất và model sang Pi5
scp hardware/config/best_chromosome_rl.json  pi@<PI_IP>:~/traffic-light/hardware/config/
scp -r rl/models/ppo_traffic/final_model.zip pi@<PI_IP>:~/traffic-light/rl/models/ppo_traffic/

# Trên Pi5: chạy controller
ssh pi@<PI_IP>
cd ~/traffic-light
python hardware/pi_controller.py
```

### Checklist Phase 4

- [ ] Arduino nhận `SET:30:4:45:4` → phản hồi `ACK:OK`
- [ ] Đèn LED chuyển pha đúng: NS_Green → NS_Yellow → EW_Green → EW_Yellow
- [ ] `PiController.apply_chromosome()` hoạt động với chromosome từ GA/RL
- [ ] Demo hoàn chỉnh: chạy RL trên PC → copy JSON → Pi5 → Arduino → Đèn

---

## Tài nguyên tham khảo

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium API](https://gymnasium.farama.org/)
- [SUMO-RL Library](https://github.com/LucasAlegre/sumo-rl)
- [Arduino Serial Reference](https://www.arduino.cc/reference/en/language/functions/communication/serial/)
- [Raspberry Pi GPIO Docs](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)
