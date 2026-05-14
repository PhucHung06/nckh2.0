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
    def __init__(self, sumocfg, tl_id="Center", delta_time=5, yellow_time=4, min_green=10, max_steps=360, use_gui=False, label="default"):
        super().__init__()
        self.sumocfg = sumocfg
        self.tl_id = tl_id
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.use_gui = use_gui
        self.label = label
        self.conn = None
        
        # Đường dẫn SUMO (Sẽ tự tìm nếu biến môi trường SUMO_HOME tồn tại)
        if 'SUMO_HOME' in os.environ:
            sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
        
        # Chọn binary: sumo (CMD) hoặc sumo-gui (Giao diện)
        binary_name = 'sumo-gui' if self.use_gui else 'sumo'
        self.sumo_binary = sumolib.checkBinary(binary_name)

        # Định nghĩa Action Space: 0: Giữ nguyên pha, 1: Chuyển pha
        self.action_space = gym.spaces.Discrete(2)

        # Định nghĩa Observation Space (9 chiều): 
        # [Số xe chờ ở 4 lanes ngõ vào, Tổng thời gian chờ ở 4 lanes ngõ vào, Trạng thái Pha NS]
        # Giả định ngã tư có 4 hướng vào chính.
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(9,), dtype=np.float32
        )

        self.is_closed = True
        self.current_step = 0
        self.max_steps = max_steps

        # Khởi tạo các biến lưu trữ metrics để tính trung bình tích lũy (CMA)
        self.cumulative_metrics = {
            "timeLoss": 0.0,
            "waitingTime": 0.0,
            "density": 0.0,
            "speed": 0.0,
            "step_count": 0
        }

    def _get_obs(self):
        # Lấy danh sách các làn được điều khiển bởi cột đèn này
        controlled_lanes = self.conn.trafficlight.getControlledLanes(self.tl_id)
        unique_lanes = list(dict.fromkeys(controlled_lanes))
        # 🚨 FIX CỨNG TÊN 4 CẠNH VÀO NGÃ TƯ (North, East, South, West)
        # Đảm bảo mảng Obs luôn có thứ tự cố định để mô hình Neural Network học tập ổn định
        edges = ['N2C', 'E2C', 'S2C', 'W2C']
        
        queues = []
        wait_times = []
        
        for edge in edges:
            edge_lanes = [l for l in unique_lanes if self.conn.lane.getEdgeID(l) == edge]
            q = sum([self.conn.lane.getLastStepHaltingNumber(l) for l in edge_lanes])
            w = sum([self.conn.lane.getWaitingTime(l) / 100.0 for l in edge_lanes])
            queues.append(q)
            wait_times.append(w)
        
        # Nếu thiếu nhánh thì fill 0
        while len(queues) < 4: queues.append(0)
        while len(wait_times) < 4: wait_times.append(0)
            
        # Thêm trạng thái pha hiện tại vào State để PPO không bị "Mù"
        # 1.0 = Đang xanh hướng Bắc Nam (Pha 0)
        # 0.0 = Đang xanh hướng Đông Tây (Pha 2)
        try:
            current_phase = self.conn.trafficlight.getPhase(self.tl_id)
            is_ns_green = 1.0 if current_phase == 0 else 0.0
        except:
            is_ns_green = 1.0
            
        return np.array(queues + wait_times + [is_ns_green], dtype=np.float32)

    def get_live_metrics(self):
        """
        Trích xuất các chỉ số hiệu suất thời gian thực từ TraCI.
        """
        if self.conn is None:
            return None

        # Danh sách các cạnh cần theo dõi (tương ứng với Benchmark)
        edges = ['N2C', 'E2C', 'S2C', 'W2C']
        
        step_timeLoss = 0.0
        step_waitingTime = 0.0
        step_density = 0.0
        step_speed = 0.0
        edge_count = len(edges)

        for edge in edges:
            # Lấy thông số từ TraCI (giống cách SUMO tính trong XML)
            # timeLoss: s, waitingTime: s, density: xe/km, speed: m/s
            step_timeLoss += self.conn.edge.getWaitingTime(edge) # TraCI trả về tổng waiting time của các xe trên cạnh
            # Lưu ý: TraCI edge.getWaitingTime trả về tổng waiting time tích lũy của xe hiện tại. 
            # Để khớp với benchmark XML (thường là trung bình), ta cần xử lý cẩn thận.
            
            # Sử dụng cách tính tương đương Benchmark:
            v_ids = self.conn.edge.getLastStepVehicleIDs(edge)
            num_vehicles = len(v_ids)
            
            if num_vehicles > 0:
                s_speed = sum([self.conn.vehicle.getSpeed(v) for v in v_ids]) / num_vehicles
                s_wait = sum([self.conn.vehicle.getWaitingTime(v) for v in v_ids]) / num_vehicles
                s_loss = sum([self.conn.vehicle.getTimeLoss(v) for v in v_ids]) / num_vehicles
            else:
                s_speed = self.conn.edge.getLastStepMeanSpeed(edge)
                s_wait = 0.0
                s_loss = 0.0
            
            # Density (xe/km) = n / (L / 1000)
            edge_length = sum([self.conn.lane.getLength(l) for l in self.conn.trafficlight.getControlledLanes(self.tl_id) if self.conn.lane.getEdgeID(l) == edge])
            if edge_length == 0: edge_length = 100 # fallback
            s_density = num_vehicles / (edge_length / 1000.0)

            step_timeLoss += s_loss
            step_waitingTime += s_wait
            step_density += s_density
            step_speed += s_speed

        # Trung bình của các cạnh trong bước này
        metrics = {
            "timeLoss": step_timeLoss / edge_count,
            "waitingTime": step_waitingTime / edge_count,
            "density": step_density / edge_count,
            "speed": step_speed / edge_count
        }

        # Cập nhật tích lũy
        self.cumulative_metrics["timeLoss"] += metrics["timeLoss"]
        self.cumulative_metrics["waitingTime"] += metrics["waitingTime"]
        self.cumulative_metrics["density"] += metrics["density"]
        self.cumulative_metrics["speed"] += metrics["speed"]
        self.cumulative_metrics["step_count"] += 1

        # Tính CMA
        count = self.cumulative_metrics["step_count"]
        cma = {
            "avg_timeLoss": self.cumulative_metrics["timeLoss"] / count,
            "avg_waitingTime": self.cumulative_metrics["waitingTime"] / count,
            "avg_density": self.cumulative_metrics["density"] / count,
            "avg_speed": self.cumulative_metrics["speed"] / count
        }
        
        return metrics, cma

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if not self.is_closed:
            try:
                self.conn.close()
            except traci.exceptions.FatalTraCIError:
                pass
            
        traci.start([self.sumo_binary, "-c", self.sumocfg, "--no-warnings", "true"], label=self.label)
        self.conn = traci.getConnection(self.label)
        
        # [CỰC KỲ QUAN TRỌNG] Khóa bộ đếm tự động của SUMO để trao toàn quyền cho AI
        self.conn.trafficlight.setPhase(self.tl_id, 0)
        self.conn.trafficlight.setPhaseDuration(self.tl_id, 10000)
        
        self.is_closed = False
        self.current_step = 0
        self.time_since_last_phase_change = 0
        
        # Reset metrics tích lũy khi bắt đầu simulation mới
        for key in self.cumulative_metrics:
            self.cumulative_metrics[key] = 0.0
        
        return self._get_obs(), {}

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        
        # Lấy pha hiện tại và tổng số pha có sẵn
        current_phase = self.conn.trafficlight.getPhase(self.tl_id)
        all_phases = self.conn.trafficlight.getAllProgramLogics(self.tl_id)[0].phases
        num_phases = len(all_phases)
        
        is_green_phase = current_phase % 2 == 0 # Giả định 0, 2 là xanh; 1, 3 là vàng
        
        # 1. Khởi tạo cờ kiểm tra xem có chuyển pha hay không
        phase_changed = False
        
        if action == 1 and is_green_phase and self.time_since_last_phase_change >= self.min_green:
            phase_changed = True # Đánh dấu đã chuyển pha
            
            # Chuyển sang đèn vàng kế tiếp
            yellow_phase = (current_phase + 1) % num_phases
            self.conn.trafficlight.setPhase(self.tl_id, yellow_phase)
            
            # Chạy 4 giây đèn vàng
            for _ in range(self.yellow_time):
                self.conn.simulationStep()
            
            # Chuyển sang pha XANH tiếp theo sau đèn vàng
            next_green_phase = (yellow_phase + 1) % num_phases
            self.conn.trafficlight.setPhase(self.tl_id, next_green_phase)
            # Khóa bộ đếm của đèn Xanh mới
            self.conn.trafficlight.setPhaseDuration(self.tl_id, 10000)
            self.time_since_last_phase_change = 0
            
            # Chạy bù phần thời gian còn lại của delta_time (Ví dụ: 5 - 4 = 1 giây)
            # Để tống thời gian bước này đúng bằng 5 giây (Chuẩn hoá MDP cho PPO)
            remaining_time = max(0, self.delta_time - self.yellow_time)
            for _ in range(remaining_time):
                self.conn.simulationStep()
                
            self.time_since_last_phase_change += remaining_time

        else:
            # Liên tục reset timer để đảm bảo thời gian SUMO không tự nhảy làm hỏng Action của AI
            self.conn.trafficlight.setPhaseDuration(self.tl_id, 10000)
            
            # Nếu không đổi đèn, chạy đủ 5s
            self.time_since_last_phase_change += self.delta_time
            for _ in range(self.delta_time):
                self.conn.simulationStep()
        
        # Cập nhật metrics sau mỗi bước delta_time
        self.get_live_metrics()

        self.current_step += 1
        obs = self._get_obs()
        
        # --- LOGIC REWARD "CẢI TIẾN" ---
        # 1. Hình phạt cho ùn tắc (Tăng trọng số cho WaitingTime để AI không để xe chờ quá lâu)
        # obs[:4]: Halting count, obs[4:8]: WaitingTime/100
        reward = -(np.sum(obs[:4]) * 1.0 + np.sum(obs[4:8]) * 0.5)
        
        # 2. Phạt nhẹ khi đòi đổi đèn để tránh spam (Gỡ bỏ kỷ luật thép, để AI linh hoạt hơn)
        if action == 1:
            reward -= 2 
            
        # 3. Phạt hành vi gây gián đoạn dòng chảy (Thực sự đổi đèn)
        if phase_changed:
            reward -= 10 
        # ----------------------------------------

        
        if self.current_step >= self.max_steps:
            terminated = True
            
        return obs, reward, terminated, truncated, {}

    def close(self):
        if not self.is_closed:
            try:
                self.conn.close()
            except traci.exceptions.FatalTraCIError:
                pass
            self.is_closed = True
