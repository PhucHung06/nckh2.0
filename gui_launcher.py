import os
import subprocess
import tkinter as tk
from functools import partial
from tkinter import messagebox, ttk

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")

if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = "python"

def run_command_in_new_terminal(command_str, title="Terminal"):
    cmd = f'start "{title}" cmd.exe /k "{command_str}"'
    try:
        subprocess.Popen(cmd, shell=True, cwd=PROJECT_ROOT)
    except Exception as exc:
        messagebox.showerror("Loi", f"Khong the chay lenh.\nChi tiet: {exc}")

def launch_benchmark():
    cmd = f'"{VENV_PYTHON}" benchmark/run_comparison.py --trials 10 --ga-gens 15'
    run_command_in_new_terminal(cmd, "Chạy Benchmark so sánh")

def launch_ga_training():
    cmd = f'"{VENV_PYTHON}" simulation/main_ga.py'
    run_command_in_new_terminal(cmd, "Huấn luyện Thuật toán di truyền (GA)")

def launch_ppo_training():
    cmd = f'"{VENV_PYTHON}" rl/train_ppo2.0.py'
    run_command_in_new_terminal(cmd, "Huấn luyện PPO(Random Flow)")

def launch_plot_ga():
    cmd = f'"{VENV_PYTHON}" benchmark/plot_ga_curve.py'
    run_command_in_new_terminal(cmd, "Trích xuất Fitness Curve (GA)")

def launch_plot_ppo():
    cmd = f'"{VENV_PYTHON}" benchmark/plot_ppo_curve.py'
    run_command_in_new_terminal(cmd, "Trích xuất Learning Curve (PPO)")

def launch_draw_roi(group_name):
    cmd = f'"{VENV_PYTHON}" vision/draw_roi.py {group_name}'
    run_command_in_new_terminal(cmd, f"Vẽ ROI {group_name}")

def launch_sumo_simulation():
    cmd = f'"{VENV_PYTHON}" demo_live_twin_v2.py'
    run_command_in_new_terminal(cmd, "Mô phỏng SUMO - Digital Twin")

# GUI Setup
root = tk.Tk()
root.title("Traffic Light AI - Control Panel v2")
root.geometry("640x650")
root.minsize(640, 650)
root.configure(padx=20, pady=20)

try:
    style = ttk.Style()
    style.theme_use("clam")
except Exception:
    pass

header = tk.Label(root, text="Hệ thống Quản lý Đèn giao thông AI", font=("Arial", 16, "bold"))
header.pack(pady=(0, 5))

sub_header = tk.Label(root, text="Bảng điều khiển trung tâm dành cho người vận hành", font=("Arial", 10), fg="gray")
sub_header.pack(pady=(0, 20))

frame_main = ttk.Frame(root)
frame_main.pack(fill=tk.BOTH, expand=True)

# 1. Giai đoạn thiết lập Vision & Simulation
frame_vision = ttk.LabelFrame(frame_main, text="1. Thiết lập Vision & Mô phỏng")
frame_vision.pack(fill=tk.X, pady=10, ipadx=10, ipady=10)

frame_roi = ttk.Frame(frame_vision)
frame_roi.pack(fill=tk.X, padx=10, pady=(5, 8))
for column_index in range(4): frame_roi.columnconfigure(column_index, weight=1)

roi_groups = ("A", "B", "C", "D")
for index, group_name in enumerate(roi_groups):
    ttk.Button(frame_roi, text=f"ROI {group_name}", command=partial(launch_draw_roi, group_name)).grid(row=0, column=index, sticky="ew", padx=2)

btn_vision_main = ttk.Button(frame_vision, text="🚀 MÔ PHỎNG SUMO (Digital Twin Dashboard)", command=launch_sumo_simulation)
btn_vision_main.pack(fill=tk.X, pady=(10, 0), padx=10)

# 2. Giai đoạn Huấn luyện (Training)
frame_train = ttk.LabelFrame(frame_main, text="2. Huấn luyện (Training)")
frame_train.pack(fill=tk.X, pady=10, ipadx=10, ipady=10)

btn_ga_train = ttk.Button(frame_train, text="🧬 Huấn luyện GA (Tối ưu chu kỳ cố định)", command=launch_ga_training)
btn_ga_train.pack(fill=tk.X, pady=3, padx=10)

btn_ppo_train = ttk.Button(frame_train, text="🧠 Huấn luyện PPO 2.0 (Học tăng cường sâu)", command=launch_ppo_training)
btn_ppo_train.pack(fill=tk.X, pady=3, padx=10)

# 3. Giai đoạn Đánh giá & Trích xuất số liệu
frame_bench = ttk.LabelFrame(frame_main, text="3. So sánh & Trích xuất kết quả")
frame_bench.pack(fill=tk.X, pady=10, ipadx=10, ipady=10)

btn_run_bench = ttk.Button(frame_bench, text="📊 Chạy Benchmark (Fixed vs GA vs PPO)", command=launch_benchmark)
btn_run_bench.pack(fill=tk.X, pady=3, padx=10)

frame_curves = ttk.Frame(frame_bench)
frame_curves.pack(fill=tk.X, padx=10, pady=5)

btn_curve_ga = ttk.Button(frame_curves, text="Fitness Curve (GA)", command=launch_plot_ga)
btn_curve_ga.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

btn_curve_ppo = ttk.Button(frame_curves, text="Learning Curve (PPO)", command=launch_plot_ppo)
btn_curve_ppo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

footer = tk.Label(root, text="Developed for Traffic Light Optimization System", font=("Arial", 8), fg="gray")
footer.pack(side=tk.BOTTOM, pady=10)

if __name__ == "__main__":
    root.mainloop()
