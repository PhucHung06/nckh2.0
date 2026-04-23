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


def launch_ga_training():
    cmd = f'"{VENV_PYTHON}" simulation/main_ga.py'
    run_command_in_new_terminal(cmd, "Huan luyen Thuat toan di truyen (GA)")


def launch_ppo_training():
    cmd = f'"{VENV_PYTHON}" rl/train_ppo.py'
    run_command_in_new_terminal(cmd, "Huan luyen Deep RL (PPO)")


def launch_tensorboard():
    tb_path = os.path.join(PROJECT_ROOT, "venv", "Scripts", "tensorboard.exe")
    if not os.path.exists(tb_path):
        tb_path = "tensorboard"

    cmd = f'"{tb_path}" --logdir rl/logs'
    run_command_in_new_terminal(cmd, "TensorBoard Monitor")
    messagebox.showinfo(
        "TensorBoard",
        "TensorBoard dang duoc khoi dong tren cua so moi.\nHay mo trinh duyet tai: http://localhost:6006",
    )


def launch_benchmark():
    cmd = f'"{VENV_PYTHON}" benchmark/run_comparison.py --trials 10 --ga-gens 15'
    run_command_in_new_terminal(cmd, "Chay Benchmark Thi dau (10 Trials)")


def launch_plot_benchmark():
    cmd = f'"{VENV_PYTHON}" benchmark/visualize_results.py'
    run_command_in_new_terminal(cmd, "Ve Bieu do Boxplot Benchmark")


def launch_plot_ga():
    cmd = f'"{VENV_PYTHON}" benchmark/plot_ga_curve.py'
    run_command_in_new_terminal(cmd, "Ve Bieu do Hoi tu GA")


def launch_plot_ppo():
    cmd = f'"{VENV_PYTHON}" benchmark/plot_ppo_curve.py'
    run_command_in_new_terminal(cmd, "Ve Bieu do PPO tu TensorBoard")


def launch_draw_roi(group_name):
    cmd = f'"{VENV_PYTHON}" vision/draw_roi.py {group_name}'
    run_command_in_new_terminal(cmd, f"Vẽ ROI {group_name}")


def launch_vision_sumo():
    cmd = f'"{VENV_PYTHON}" vision/main.py'
    run_command_in_new_terminal(cmd, "Mô phong SUMO")


root = tk.Tk()
root.title("Traffic Light AI - Control Panel")
root.geometry("640x650")
root.minsize(640, 650)
root.configure(padx=20, pady=20)

try:
    style = ttk.Style()
    style.theme_use("clam")
except Exception:
    pass


header = tk.Label(
    root,
    text="Hệ thống Quản lý Đèn giao thông AI",
    font=("Arial", 16, "bold"),
)
header.pack(pady=(0, 5))

sub_header = tk.Label(
    root,
    text="Nhấn các nút bên dưới để khởi chạy tự động thay vì gõ lệnh",
    font=("Arial", 10),
    fg="gray",
)
sub_header.pack(pady=(0, 20))


frame_main = ttk.Frame(root)
frame_main.pack(fill=tk.BOTH, expand=True)


frame_vision = ttk.LabelFrame(frame_main, text="0. Vision + SUMO")
frame_vision.pack(fill=tk.X, pady=10, ipadx=10, ipady=10)

frame_roi = ttk.Frame(frame_vision)
frame_roi.pack(fill=tk.X, padx=10, pady=(5, 8))

for column_index in range(2):
    frame_roi.columnconfigure(column_index, weight=1)

roi_groups = ("A", "B", "C", "D")
for index, group_name in enumerate(roi_groups):
    row_index = index // 2
    column_index = index % 2
    ttk.Button(
        frame_roi,
        text=f"Vẽ ROI {group_name}",
        command=partial(launch_draw_roi, group_name),
    ).grid(row=row_index, column=column_index, sticky="ew", padx=5, pady=5)

btn_vision_main = ttk.Button(frame_vision, text="Mô phỏng SUMO", command=launch_vision_sumo)
btn_vision_main.pack(fill=tk.X, pady=(0, 5), padx=10)


frame_train = ttk.LabelFrame(frame_main, text="1. Giai đoạn Huấn luyện (Training)")
frame_train.pack(fill=tk.X, pady=10, ipadx=10, ipady=10)

btn_ga = ttk.Button(
    frame_train,
    text="Huấn luyện Thuật toán di truyền (GA)",
    command=launch_ga_training,
)
btn_ga.pack(fill=tk.X, pady=3, padx=10)

btn_ppo = ttk.Button(
    frame_train,
    text="Huấn luyện Deep RL (PPO)",
    command=launch_ppo_training,
)
btn_ppo.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=3, padx=(10, 5))

btn_tb = ttk.Button(frame_train, text="Mở TensorBoard", command=launch_tensorboard)
btn_tb.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=3, padx=(5, 10))


frame_bench = ttk.LabelFrame(frame_main, text="2. Giai đoạn Benchmark & Thống kê")
frame_bench.pack(fill=tk.X, pady=10, ipadx=10, ipady=10)

btn_run_bench = ttk.Button(
    frame_bench,
    text="Chạy Benchmark (Fixed vs GA vs PPO)",
    command=launch_benchmark,
)
btn_run_bench.pack(fill=tk.X, pady=3, padx=10)

btn_plot_bench = ttk.Button(
    frame_bench,
    text="Xuất Biểu đồ Boxplot số liệu tĩnh",
    command=launch_plot_benchmark,
)
btn_plot_bench.pack(fill=tk.X, pady=3, padx=10)


frame_curves = ttk.LabelFrame(frame_main, text="3. Xuất Biểu đồ Hội tụ (Vẽ ảnh PNG)")
frame_curves.pack(fill=tk.X, pady=10, ipadx=10, ipady=10)

btn_curve_ga = ttk.Button(
    frame_curves,
    text="Trích xuất Fitness Curve (GA)",
    command=launch_plot_ga,
)
btn_curve_ga.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=3, padx=(10, 5))

btn_curve_ppo = ttk.Button(
    frame_curves,
    text="Trích xuất Learning Curve (PPO)",
    command=launch_plot_ppo,
)
btn_curve_ppo.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=3, padx=(5, 10))


footer = tk.Label(
    root,
    text="Developed for Traffic Light Optimization System",
    font=("Arial", 8),
    fg="gray",
)
footer.pack(side=tk.BOTTOM, pady=10)


if __name__ == "__main__":
    root.mainloop()
