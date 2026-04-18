# demo_live_twin.py
import os
import sys
import time
import numpy as np

# Thêm path
sys.path.insert(0, os.path.dirname(__file__))

import traci
from stable_baselines3 import PPO
from simulation.sumo_gym_env import SumoGymEnv
from hardware.pi_controller import PiController

def set_arduino_phase(ctrl, phase_num):
    """Gửi lệnh ép phase xuống mạch thật"""
    if ctrl and ctrl.ser:
        cmd = f"FORCE:{phase_num}\n"
        ctrl.ser.write(cmd.encode())
        print(f"[Hardware] Synced Arduino Phase: {phase_num}")

def main():
    print("🚦 Bắt đầu chế độ Digital Twin (SUMO GUI + PPO + ARDUINO) 🚦")
    
    # 1. Kết nối mạch
    ctrl = PiController()
    if not ctrl.ser:
        print("[Demo] Không tìm thấy mạch Arduino. Sẽ chỉ chạy trên PC.")
        
    # 2. Nạp Model
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'rl', 'models', 'ppo_dynamic', 'ppo_traffic_dynamic.zip')
    if not os.path.exists(MODEL_PATH):
        print("Không tìm thấy model PPO. Vui lòng train trước.")
        return
    model = PPO.load(MODEL_PATH, device='cpu')
    
    # 3. Chạy Môi trường SUMO (Bật Cửa Sổ Màn hình - GUI)
    SUMO_CFG = os.path.join(os.path.dirname(__file__), 'data', 'run1.sumocfg')
    env = SumoGymEnv(SUMO_CFG, use_gui=True, delta_time=5)
    
    try:
        obs, _ = env.reset() # Khởi tạo trạng thái ban đầu
        step_count = 0
        while True:
            step_count += 1
            # 1. AI quan sát trạng thái ngã tư ảo
            action, _ = model.predict(obs, deterministic=True)
            
            # 2. Lấy dữ liệu hàng chờ hiện tại để in ra (lấy từ obs đã chuẩn hóa)
            # Obs: [Q_N, Q_S, Q_E, Q_W, W_N, W_S, W_E, W_W]
            queues = obs[:4].astype(int)
            
            # 3. Ra quyết định và in Dashboard
            decision_text = "🔄 CHUYỂN PHA (Switch)" if action == 1 else "🟢 GIỮ NGUYÊN (Stay)"
            
            print(f"\n{'='*50}")
            print(f"🚀 [Digital Twin Step {step_count}]")
            print(f"📍 Hướng Bắc: {queues[0]} xe | Hướng Nam: {queues[1]} xe")
            print(f"📍 Hướng Đông: {queues[2]} xe | Hướng Tây: {queues[3]} xe")
            print(f"🤖 AI PPO Quyết định: {decision_text}")
            
            # 4. Thực thi trong SUMO
            try:
                obs, reward, terminated, truncated, _ = env.step(action)
                current_phase = traci.trafficlight.getPhase(env.tl_id)
                # Chuyển đổi phase số sang tên pha cho dễ hiểu
                phase_name = ["NORTH-SOUTH GREEN", "YELLOW", "EAST-WEST GREEN", "YELLOW"][current_phase]
                print(f"🚥 Đèn SUMO hiện tại: {phase_name}")
                
                # 5. Đồng bộ Arduino thật
                set_arduino_phase(ctrl, current_phase)
            except traci.exceptions.FatalTraCIError:
                print("⚠️ Đã đóng cửa sổ SUMO GUI. Kết thúc Demo.")
                break
            
            time.sleep(1) # Giãn cách để dễ nhìn
            
            if terminated or truncated:
                print("🔄 Reset mô phỏng (Hết thời gian)...")
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\n[Demo] Đã dừng theo yêu cầu người dùng.")
    finally:
        try:
            env.close()
        except:
            pass

if __name__ == "__main__":
    main()
