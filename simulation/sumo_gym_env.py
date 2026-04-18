import gymnasium as gym
import numpy as np
import os
import sys
import traci
import sumolib

class SumoGymEnv(gym.Env):
    """
    Môi trường Gymnasium điều khiển đèn giao thông theo thời gian thực sử dụng TraCI.
    """
    def __init__(self, sumocfg, tl_id="Center", delta_time=5, yellow_time=4, min_green=10, use_gui=False):
        super().__init__()
        self.sumocfg = sumocfg
        self.tl_id = tl_id
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.use_gui = use_gui
        
        # Đường dẫn SUMO (Sẽ tự tìm nếu biến môi trường SUMO_HOME tồn tại)
        if 'SUMO_HOME' in os.environ:
            sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
        
        # Chọn binary: sumo (CMD) hoặc sumo-gui (Giao diện)
        binary_name = 'sumo-gui' if self.use_gui else 'sumo'
        self.sumo_binary = sumolib.checkBinary(binary_name)

        # Định nghĩa Action Space: 0: Giữ nguyên pha, 1: Chuyển pha
        self.action_space = gym.spaces.Discrete(2)

        # Định nghĩa Observation Space (8 chiều): 
        # [Số xe chờ ở 4 lanes ngõ vào, Tổng thời gian chờ ở 4 lanes ngõ vào]
        # Giả định ngã tư có 4 hướng vào chính.
        self.observation_space = gym.spaces.Box(
            low=0, high=100, shape=(8,), dtype=np.float32
        )

        self.is_closed = True
        self.current_step = 0
        self.max_steps = 360  # Mô phỏng khoảng 30 phút (360 * 5s)

    def _get_obs(self):
        # Lấy danh sách các làn được điều khiển bởi cột đèn này
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        # Lọc ra các làn duy nhất (Traci có thể trả về trùng lặp cho mỗi hướng rẽ)
        unique_lanes = list(dict.fromkeys(controlled_lanes))
        
        # Chọn ra 4 làn chính vào (tùy thuộc vào cấu trúc network của bạn)
        # Ở đây ta lấy 4 làn đầu tiên làm ví dụ, hoặc bạn có thể fix cứng ID
        lanes = unique_lanes[:4] 
        
        queues = [traci.lane.getLastStepHaltingNumber(l) for l in lanes]
        wait_times = [traci.lane.getWaitingTime(l) / 100.0 for l in lanes] # Chuẩn hóa nhẹ
        
        # Nếu thiếu làn thì fill 0
        while len(queues) < 4: queues.append(0)
        while len(wait_times) < 4: wait_times.append(0)
            
        return np.array(queues + wait_times, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if not self.is_closed:
            traci.close()
            
        traci.start([self.sumo_binary, "-c", self.sumocfg, "--no-warnings", "true"])
        self.is_closed = False
        self.current_step = 0
        self.time_since_last_phase_change = 0
        
        return self._get_obs(), {}

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        
        # Lấy pha hiện tại và tổng số pha có sẵn
        current_phase = traci.trafficlight.getPhase(self.tl_id)
        all_phases = traci.trafficlight.getAllProgramLogics(self.tl_id)[0].phases
        num_phases = len(all_phases)
        
        # CHỈ cho phép AI đổi đèn nếu:
        # 1. Action = 1
        # 2. Đèn hiện tại là đèn XANH (thường là pha 0 hoặc 2)
        # 3. Đã xanh đủ thời gian tối thiểu (min_green)
        
        is_green_phase = current_phase % 2 == 0 # Giả định 0, 2 là xanh; 1, 3 là vàng
        
        if action == 1 and is_green_phase and self.time_since_last_phase_change >= self.min_green:
            # Chuyển sang đèn vàng kế tiếp (dùng % để tránh tràn index)
            yellow_phase = (current_phase + 1) % num_phases
            traci.trafficlight.setPhase(self.tl_id, yellow_phase)
            
            # Chạy mô phỏng hết thời gian đèn vàng
            for _ in range(self.yellow_time):
                traci.simulationStep()
            
            # Chuyển sang pha XANH tiếp theo sau đèn vàng
            next_green_phase = (yellow_phase + 1) % num_phases
            traci.trafficlight.setPhase(self.tl_id, next_green_phase)
            self.time_since_last_phase_change = 0
        else:
            self.time_since_last_phase_change += self.delta_time
        # Chạy mô phỏng 5 giây tiếp theo
        for _ in range(self.delta_time):
            traci.simulationStep()
            
        self.current_step += 1
        obs = self._get_obs()
        
        # Reward = Âm của tổng số xe đang phải dừng (càng ít xe dừng càng tốt)
        reward = -np.sum(obs[:4]) 
        
        if self.current_step >= self.max_steps:
            terminated = True
            
        return obs, reward, terminated, truncated, {}

    def close(self):
        if not self.is_closed:
            traci.close()
            self.is_closed = True
