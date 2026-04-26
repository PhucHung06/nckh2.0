# demo_live_twin.py
import os
import sys
import time
import tkinter as tk
import numpy as np

# Thêm path
sys.path.insert(0, os.path.dirname(__file__))

import traci
from PIL import Image, ImageTk
from stable_baselines3 import PPO
from simulation.sumo_gym_env import SumoGymEnv
from hardware.pi_controller import PiController


PROJECT_ROOT = os.path.dirname(__file__)
GREEN_LIGHT_PATH = os.path.join(PROJECT_ROOT, "data", "image", "Đèn xanh.jpg")
RED_LIGHT_PATH = os.path.join(PROJECT_ROOT, "data", "image", "Đèn đỏ.jpg")
YELLOW_LIGHT_PATH = os.path.join(PROJECT_ROOT, "data", "image", "Đèn vàng.jpg")


def get_direction_images(phase_num):
    """Map SUMO phase to image states for North/East/South/West."""
    states = {
        "north": "red",
        "east": "red",
        "south": "red",
        "west": "red",
    }

    if phase_num == 0:
        states["north"] = "green"
        states["south"] = "green"
    elif phase_num == 1:
        states["north"] = "yellow"
        states["south"] = "yellow"
    elif phase_num == 2:
        states["east"] = "green"
        states["west"] = "green"
    elif phase_num == 3:
        states["east"] = "yellow"
        states["west"] = "yellow"

    return states


class LiveTwinLightPanel:
    """A dark traffic-light panel that shows STAY/SWITCH in the center."""

    def __init__(self):
        self.closed = False
        self.root = tk.Tk()
        self.root.title("Live Twin - Traffic Light State")
        self.root.geometry("620x620")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.canvas = tk.Canvas(self.root, width=620, height=620, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.images = {
            "green": self._load_square_image(GREEN_LIGHT_PATH),
            "red": self._load_square_image(RED_LIGHT_PATH),
            "yellow": self._load_square_image(YELLOW_LIGHT_PATH),
        }
        self.light_items = {}
        self._draw_layout()
        self.root.update()

    def _load_square_image(self, path):
        image = Image.open(path).convert("RGBA")
        image.thumbnail((120, 120), Image.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _draw_layout(self):
        positions = {
            "north": (310, 115),
            "west": (115, 310),
            "east": (505, 310),
            "south": (310, 505),
        }

        for direction, (x, y) in positions.items():
            self.light_items[direction] = self.canvas.create_image(
                x,
                y,
                image=self.images["red"],
            )

        self.decision_item = self.canvas.create_text(
            310,
            310,
            text="WAIT",
            fill="#10f5a8",
            font=("Consolas", 48, "bold"),
        )

    def update(self, step_count, action, phase_num):
        if self.closed:
            return

        decision = "SWITCH" if int(action) == 1 else "STAY"
        self.canvas.itemconfig(self.decision_item, text=decision)

        for direction, color in get_direction_images(phase_num).items():
            self.canvas.itemconfig(self.light_items[direction], image=self.images[color])

        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

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
    light_panel = LiveTwinLightPanel()
    
    try:
        obs, _ = env.reset() # Khởi tạo trạng thái ban đầu
        step_count = 0
        while True:
            step_count += 1
            # 1. AI quan sát trạng thái ngã tư ảo
            action, _ = model.predict(obs, deterministic=True)
            action_value = int(np.asarray(action).item())
            
            # Lấy dữ liệu hàng chờ hiện tại để in ra (lấy từ obs đã chuẩn hóa)
            # Theo danh sách Edge thực tế từ SUMO: [Bắc (0), Đông (1), Nam (2), Tây (3)]
            q_n, q_e, q_s, q_w = obs[0:4].astype(int)
            
            # 3. Ra quyết định và in Dashboard
            decision_text = "🔄 CHUYỂN PHA (Switch)" if action_value == 1 else "🟢 GIỮ NGUYÊN (Stay)"
            
            print(f"\n{'='*50}")
            print(f"🚀 [Digital Twin Step {step_count}]")
            print(f"📍 Hướng Bắc: {q_n:2d} xe | Hướng Nam: {q_s:2d} xe")
            print(f"📍 Hướng Đông: {q_e:2d} xe | Hướng Tây: {q_w:2d} xe")
            print(f"🤖 AI PPO Quyết định: {decision_text}")
            
            # 4. Thực thi trong SUMO
            try:
                current_phase_before = env.conn.trafficlight.getPhase(env.tl_id)
                can_switch_now = (
                    action_value == 1
                    and current_phase_before % 2 == 0
                    and env.time_since_last_phase_change >= env.min_green
                )
                if can_switch_now:
                    yellow_phase = (current_phase_before + 1) % 4
                    light_panel.update(step_count, action_value, yellow_phase)
                    time.sleep(1)

                obs, reward, terminated, truncated, _ = env.step(action_value)
                current_phase = env.conn.trafficlight.getPhase(env.tl_id)
                # Chuyển đổi phase số sang tên pha cho dễ hiểu
                phase_name = ["NORTH-SOUTH GREEN", "YELLOW", "EAST-WEST GREEN", "YELLOW"][current_phase]
                light_panel.update(step_count, action_value, current_phase)
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
        light_panel.close()
        try:
            env.close()
        except:
            pass

if __name__ == "__main__":
    main()
