# demo_live_twin_v2.py
import os
import sys
import time
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk

# Thêm path
sys.path.insert(0, os.path.dirname(__file__))

import traci
from stable_baselines3 import PPO
from simulation.sumo_gym_env import SumoGymEnv
from hardware.pi_controller import PiController

PROJECT_ROOT = os.path.dirname(__file__)
GREEN_LIGHT_PATH = os.path.join(PROJECT_ROOT, "data", "image", "Green.jpg")
RED_LIGHT_PATH = os.path.join(PROJECT_ROOT, "data", "image", "Red.jpg")
YELLOW_LIGHT_PATH = os.path.join(PROJECT_ROOT, "data", "image", "Yellow.jpg")

def get_direction_images(phase_num):
    states = {"north": "red", "east": "red", "south": "red", "west": "red"}
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

class AdvancedDigitalTwinDashboard:
    def __init__(self):
        self.closed = False
        self.root = tk.Tk()
        self.root.title("Advanced Digital Twin Dashboard - NCKH 2.0")
        self.root.geometry("1100x700")
        self.root.configure(bg="#1a1a1a")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        # Layout Main
        self.left_frame = tk.Frame(self.root, bg="#1a1a1a", width=650)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.right_frame = tk.Frame(self.root, bg="#2d2d2d", width=400)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

        self.current_mode = "PPO"
        self.all_cmas = {
            "Fixed": {"avg_timeLoss": 0, "avg_waitingTime": 0, "avg_density": 0, "avg_speed": 0},
            "GA": {"avg_timeLoss": 0, "avg_waitingTime": 0, "avg_density": 0, "avg_speed": 0},
            "PPO": {"avg_timeLoss": 0, "avg_waitingTime": 0, "avg_density": 0, "avg_speed": 0},
            "STOP": {"avg_timeLoss": 0, "avg_waitingTime": 0, "avg_density": 0, "avg_speed": 0}
        }
        self.flow_mode = "AI Vision" # Mặc định

        # Left: Visualizer
        self.canvas = tk.Canvas(self.left_frame, width=620, height=620, bg="#111111", highlightthickness=0)
        self.canvas.pack(pady=20)

        self.images = {
            "green": self._load_square_image(GREEN_LIGHT_PATH),
            "red": self._load_square_image(RED_LIGHT_PATH),
            "yellow": self._load_square_image(YELLOW_LIGHT_PATH),
        }
        self.light_items = {}
        self.lane_count_items = {}
        self._draw_base_layout()

        # Right: Metrics
        self._setup_metrics_ui()

        self.root.update()

    def _load_square_image(self, path):
        image = Image.open(path).convert("RGBA")
        image.thumbnail((100, 100), Image.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _draw_base_layout(self):
        # Draw roads (simplified)
        self.canvas.create_rectangle(260, 0, 360, 620, fill="#333333", outline="") # NS Road
        self.canvas.create_rectangle(0, 260, 620, 360, fill="#333333", outline="") # EW Road
        
        positions = {
            "north": (310, 100),
            "west": (100, 310),
            "east": (520, 310),
            "south": (310, 520),
        }
        for direction, (x, y) in positions.items():
            self.light_items[direction] = self.canvas.create_image(x, y, image=self.images["red"])

        # Lane count labels
        self.lane_count_items = {
            "N": self.canvas.create_text(310, 200, text="N: 0", fill="white", font=("Arial", 14, "bold")),
            "S": self.canvas.create_text(310, 420, text="S: 0", fill="white", font=("Arial", 14, "bold")),
            "E": self.canvas.create_text(420, 310, text="E: 0", fill="white", font=("Arial", 14, "bold")),
            "W": self.canvas.create_text(200, 310, text="W: 0", fill="white", font=("Arial", 14, "bold")),
        }

        self.decision_item = self.canvas.create_text(310, 310, text="WAIT", fill="#10f5a8", font=("Consolas", 32, "bold"))

    def _setup_metrics_ui(self):
        # Title
        tk.Label(self.right_frame, text="HỆ THỐNG GIÁM SÁT", fg="#10f5a8", bg="#2d2d2d", font=("Arial", 18, "bold")).pack(pady=15)

        # Mode Selection
        mode_btn_frame = tk.Frame(self.right_frame, bg="#2d2d2d")
        mode_btn_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self.mode_btns = {}
        for mode in ["Fixed", "GA", "PPO", "STOP"]:
            color = "#444444"
            if mode == "STOP": color = "#cc0000"
            btn = tk.Button(mode_btn_frame, text=mode, command=lambda m=mode: self.set_mode(m), 
                            bg=color, fg="white", font=("Arial", 9, "bold"), width=7)
            btn.pack(side=tk.LEFT, padx=3, expand=True)
            self.mode_btns[mode] = btn

        # Mode Indicator
        self.mode_label = tk.Label(self.right_frame, text="MODE: PPO", fg="white", bg="#10f5a8", font=("Arial", 12, "bold"), width=30, pady=5)
        self.mode_label.pack(pady=10)
        self.set_mode("PPO") # Initial state

        # Traffic Flow Selection
        flow_container = tk.LabelFrame(self.right_frame, text=" LUỒNG GIAO THÔNG ", fg="white", bg="#2d2d2d", font=("Arial", 10, "bold"), padx=10, pady=10)
        flow_container.pack(fill=tk.X, padx=15, pady=5)

        self.flow_btns = {}
        for fmode in ["AI Vision", "Random"]:
            btn = tk.Button(flow_container, text=fmode, command=lambda f=fmode: self.set_flow_mode(f), 
                            bg="#444444", fg="white", font=("Arial", 9))
            btn.pack(side=tk.LEFT, padx=5, expand=True)
            self.flow_btns[fmode] = btn
        self.set_flow_mode("AI Vision")

        self.set_flow_mode("AI Vision")

        # Step Counter
        self.step_label = tk.Label(self.right_frame, text="Step: 0", fg="#cccccc", bg="#2d2d2d", font=("Arial", 11))
        self.step_label.pack()

        # Current Metrics Table
        metrics_container = tk.LabelFrame(self.right_frame, text=" THÔNG SỐ HIỆN TẠI ", fg="white", bg="#2d2d2d", font=("Arial", 10, "bold"), padx=10, pady=10)
        metrics_container.pack(fill=tk.X, padx=15, pady=10)

        self.live_vars = {
            "TimeLoss": tk.StringVar(value="0.00 s"),
            "WaitTime": tk.StringVar(value="0.00 s"),
            "Density": tk.StringVar(value="0.00 xe/km"),
            "Speed": tk.StringVar(value="0.00 m/s")
        }

        for label, var in self.live_vars.items():
            row = tk.Frame(metrics_container, bg="#2d2d2d")
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"{label}:", fg="#aaaaaa", bg="#2d2d2d", font=("Arial", 10)).pack(side=tk.LEFT)
            tk.Label(row, textvariable=var, fg="white", bg="#2d2d2d", font=("Arial", 10, "bold")).pack(side=tk.RIGHT)

        # Comparison Table
        comp_container = tk.LabelFrame(self.right_frame, text=" SO SÁNH HIỆU SUẤT (Avg) ", fg="#10f5a8", bg="#2d2d2d", font=("Arial", 10, "bold"), padx=10, pady=10)
        comp_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Header
        header = tk.Frame(comp_container, bg="#3d3d3d")
        header.pack(fill=tk.X)
        tk.Label(header, text="Metric", width=15, fg="#10f5a8", bg="#3d3d3d", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(header, text="Fixed", width=8, fg="white", bg="#3d3d3d", font=("Arial", 9)).pack(side=tk.LEFT)
        tk.Label(header, text="GA", width=8, fg="white", bg="#3d3d3d", font=("Arial", 9)).pack(side=tk.LEFT)
        tk.Label(header, text="PPO", width=8, fg="#10f5a8", bg="#3d3d3d", font=("Arial", 9, "bold")).pack(side=tk.LEFT)

        self.comp_rows = {}
        metrics_with_units = [("TimeLoss", "s"), ("WaitTime", "s"), ("Density", "xe/km"), ("Speed", "m/s")]
        for m, unit in metrics_with_units:
            row = tk.Frame(comp_container, bg="#2d2d2d")
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"{m} ({unit})", width=15, fg="#aaaaaa", bg="#2d2d2d", font=("Arial", 9)).pack(side=tk.LEFT)
            self.comp_rows[m] = {
                "Fixed": tk.Label(row, text="-", width=8, fg="white", bg="#2d2d2d", font=("Arial", 9)),
                "GA": tk.Label(row, text="-", width=8, fg="white", bg="#2d2d2d", font=("Arial", 9)),
                "PPO": tk.Label(row, text="-", width=8, fg="#10f5a8", bg="#2d2d2d", font=("Arial", 9, "bold"))
            }
            for btn in self.comp_rows[m].values(): btn.pack(side=tk.LEFT)

    def set_mode(self, mode):
        self.current_mode = mode
        for m, btn in self.mode_btns.items():
            if m == mode:
                btn.config(bg="#10f5a8", fg="black")
            else:
                color = "#444444"
                if m == "STOP": color = "#cc0000"
                btn.config(bg=color, fg="white")
        
        color = "#10f5a8" if mode == "PPO" else "#f5a810" if mode == "GA" else "#a810f5"
        if mode == "STOP": color = "#ff4444"
        self.mode_label.config(text=f"MODE: {mode}", bg=color)

    def set_flow_mode(self, fmode):
        self.flow_mode = fmode
        for f, btn in self.flow_btns.items():
            if f == fmode:
                btn.config(bg="#3498db", fg="white") # Blue for flow
            else:
                btn.config(bg="#444444", fg="white")
        print(f"🌊 Đổi luồng giao thông sang: {fmode}")

    def flash_decision(self, action):
        """Tạo hiệu ứng nhấp nháy khi AI ra quyết định Switch."""
        if action == 1:
            # Màu vàng rực cho Switch
            self.canvas.itemconfig(self.decision_item, fill="#ffff00", font=("Consolas", 42, "bold"))
            self.root.after(300, lambda: self.canvas.itemconfig(self.decision_item, fill="#ffcc00", font=("Consolas", 36, "bold")))
        else:
            # Xanh dịu cho Stay
            self.canvas.itemconfig(self.decision_item, fill="#00ff00", font=("Consolas", 32, "bold"))
            self.root.after(300, lambda: self.canvas.itemconfig(self.decision_item, fill="#10f5a8", font=("Consolas", 32, "bold")))

    def update_ui(self, step, action, phase, queues, metrics, cma, mode="PPO"):
        if self.closed: return
        
        # Update CMA for current mode
        self.all_cmas[mode] = cma
        
        # Update lights
        decision = "SWITCH" if int(action) == 1 else "STAY"
        if mode == "Fixed": decision = "FIXED"
        if mode == "GA": decision = "GA CTRL"
        
        self.canvas.itemconfig(self.decision_item, text=decision)
        self.flash_decision(action)

        for direction, color in get_direction_images(phase).items():
            self.canvas.itemconfig(self.light_items[direction], image=self.images[color])

        # Update lane counts
        self.canvas.itemconfig(self.lane_count_items["N"], text=f"N: {queues[0]} xe")
        self.canvas.itemconfig(self.lane_count_items["E"], text=f"E: {queues[1]} xe")
        self.canvas.itemconfig(self.lane_count_items["S"], text=f"S: {queues[2]} xe")
        self.canvas.itemconfig(self.lane_count_items["W"], text=f"W: {queues[3]} xe")

        # Update Live Metrics
        self.step_label.config(text=f"Simulation Step: {step}")
        self.live_vars["TimeLoss"].set(f"{metrics['timeLoss']:.2f} s")
        self.live_vars["WaitTime"].set(f"{metrics['waitingTime']:.2f} s")
        self.live_vars["Density"].set(f"{metrics['density']:.2f} xe/km")
        self.live_vars["Speed"].set(f"{metrics['speed']:.2f} m/s")

        # Update Comparison (Display all stored CMAs)
        for m_key, m_label in [("timeLoss", "TimeLoss"), ("waitingTime", "WaitTime"), ("density", "Density"), ("speed", "Speed")]:
            for m_type in ["Fixed", "GA", "PPO"]:
                val = self.all_cmas[m_type][f"avg_{m_key}"]
                text = f"{val:.1f}" if val > 0 else "-"
                self.comp_rows[m_label][m_type].config(text=text)

        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def close(self):
        if self.closed: return
        self.closed = True
        try: self.root.destroy()
        except: pass

def main():
    print("🚦 Bắt đầu Advanced Digital Twin Dashboard v2 🚦")
    
    # Kết nối phần cứng
    ctrl = PiController()
    
    # Nạp Model PPO v2.0 (Bản tốt nhất)
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'rl', 'models', 'ppo_random', 'ppo_traffic_random_v2.zip')
    if not os.path.exists(MODEL_PATH):
        # Fallback về best_model nếu không thấy v2.0
        MODEL_PATH = os.path.join(PROJECT_ROOT, 'rl', 'models', 'ppo_dynamic', 'best_model.zip')
        
    print(f"🧠 Đang nạp Model PPO: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, device='cpu')
    
    # Môi trường
    SUMO_CFG = os.path.join(PROJECT_ROOT, 'data', 'run1.sumocfg')
    env = SumoGymEnv(SUMO_CFG, use_gui=True, delta_time=5)
    dashboard = AdvancedDigitalTwinDashboard()
    
    try:
        obs, _ = env.reset()
        step_count = 0
        
        # GA Best Chromosome from Benchmark: [24, 3, 19, 3]
        GA_CHROMOSOME = [15, 3, 17, 3]
        FIXED_CHROMOSOME = [30, 4, 30, 4]
        
        last_mode = dashboard.current_mode
        
        while not dashboard.closed:
            mode = dashboard.current_mode

            # Nếu người dùng đổi Mode trên UI -> Reset lại toàn bộ simulation để so sánh công bằng
            if mode != last_mode:
                if mode == "STOP":
                    print("⏸️ Simulation PAUSED.")
                    last_mode = mode
                    continue
                
                print(f"🔄 Đổi chế độ sang {mode}. Đang reset dòng xe...")
                obs, _ = env.reset()
                step_count = 0
                last_mode = mode
                
            if mode == "STOP":
                dashboard.update_ui(step_count, 0, current_phase, queues, metrics, cma, mode="STOP")
                time.sleep(0.5)
                continue

            step_count += 1
            
            # Ra quyết định dựa trên Mode
            action_value = 0
            if mode == "PPO":
                action, _ = model.predict(obs, deterministic=True)
                action_value = int(np.asarray(action).item())
            elif mode == "GA":
                # GA logic: Đợi hết green_time thì switch
                current_phase = env.conn.trafficlight.getPhase(env.tl_id)
                g_ns, y_ns, g_ew, y_ew = GA_CHROMOSOME
                
                # Tính toán xem có nên switch không (đơn giản hóa cho demo)
                time_in_phase = env.time_since_last_phase_change
                if current_phase == 0 and time_in_phase >= g_ns: action_value = 1
                elif current_phase == 2 and time_in_phase >= g_ew: action_value = 1
            else: # Fixed
                current_phase = env.conn.trafficlight.getPhase(env.tl_id)
                g_ns, y_ns, g_ew, y_ew = FIXED_CHROMOSOME
                time_in_phase = env.time_since_last_phase_change
                if current_phase == 0 and time_in_phase >= g_ns: action_value = 1
                elif current_phase == 2 and time_in_phase >= g_ew: action_value = 1

            # Sinh xe ngẫu nhiên nếu ở chế độ Random
            if dashboard.flow_mode == "Random":
                import random
                routes = [
                    "route_Zone_AL", "route_Zone_AR", "route_Zone_AM",
                    "route_Zone_BL", "route_Zone_BR", "route_Zone_BM",
                    "route_Zone_CL", "route_Zone_CR", "route_Zone_CM",
                    "route_Zone_DL", "route_Zone_DR", "route_Zone_DM"
                ]
                # Sinh 1-3 xe ngẫu nhiên mỗi bước 5s
                for _ in range(random.randint(1, 3)):
                    r = random.choice(routes)
                    v_id = f"rand_{step_count}_{_}_{random.randint(0,100)}"
                    try:
                        env.conn.vehicle.add(v_id, r, typeID="yolo_car")
                    except:
                        pass

            # Step SUMO
            try:
                obs, reward, terminated, truncated, _ = env.step(action_value)
                metrics, cma = env.get_live_metrics()
                current_phase = env.conn.trafficlight.getPhase(env.tl_id)
                queues = obs[0:4].astype(int)
                
                dashboard.update_ui(step_count, action_value, current_phase, queues, metrics, cma, mode=mode)
                
                if ctrl and ctrl.ser:
                    cmd = f"FORCE:{current_phase}\n"
                    ctrl.ser.write(cmd.encode())
                
                if terminated or truncated:
                    print(f"🔄 Reset mô phỏng {mode}...")
                    obs, _ = env.reset()
                    step_count = 0
                
                time.sleep(0.5)
                
            except traci.exceptions.FatalTraCIError:
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.close()
        env.close()

if __name__ == "__main__":
    main()
