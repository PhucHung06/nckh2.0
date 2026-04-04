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
