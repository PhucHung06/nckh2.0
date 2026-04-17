import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

# Lấy đường dẫn tới thư mục gốc của dự án
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Lấy python.exe bên trong môi trường ảo
VENV_PYTHON = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")

if not os.path.exists(VENV_PYTHON):
    # Fallback to system python if venv is missing
    VENV_PYTHON = "python"


def run_command_in_new_terminal(command_str, title="Terminal"):
    """
    Chạy lệnh trong một cửa sổ Command Prompt mới.
    Sử dụng tham số /k để giữ cửa sổ mở sau khi chạy xong, giúp người dùng đọc kết quả.
    """
    # Lệnh cmd hoàn chỉnh: Mở cmd, set tiêu đề, activate venv (tuỳ chọn) và chạy lệnh
    # Vì sử dụng thẳng venv_python nên không cần activate venv nữa.
    cmd = f'start "{title}" cmd.exe /k "{command_str}"'
    try:
        subprocess.Popen(cmd, shell=True, cwd=PROJECT_ROOT)
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể chạy lệnh khởi chạy.\nChi tiết: {str(e)}")


def launch_ga_training():
    cmd = f'"{VENV_PYTHON}" simulation/main_ga.py'
    run_command_in_new_terminal(cmd, "Huan luyen Thuat toan di truyen (GA)")

def launch_ppo_training():
    cmd = f'"{VENV_PYTHON}" rl/train_ppo.py'
    run_command_in_new_terminal(cmd, "Huan luyen Deep RL (PPO)")

def launch_tensorboard():
    # Tensorboard.exe thường nằm cùng thư mục Scripts của venv
    tb_path = os.path.join(PROJECT_ROOT, "venv", "Scripts", "tensorboard.exe")
    if not os.path.exists(tb_path):
        tb_path = "tensorboard"
    cmd = f'"{tb_path}" --logdir rl/logs'
    run_command_in_new_terminal(cmd, "TensorBoard Monitor")
    messagebox.showinfo("TensorBoard", "TensorBoard đang được khởi động trên cửa sổ mới.\nHãy mở trình duyệt và truy cập: http://localhost:6006")

def launch_benchmark():
    cmd = f'"{VENV_PYTHON}" benchmark/run_comparison.py --trials 10 --ga-gens 15'
    run_command_in_new_terminal(cmd, "Chay Benchmark Thi dau (10 Trials)")

def launch_plot_benchmark():
    cmd = f'"{VENV_PYTHON}" benchmark/visualize_results.py'
    run_command_in_new_terminal(cmd, "Ve Bieu Do Boxplot Benchmark")

def launch_plot_ga():
    cmd = f'"{VENV_PYTHON}" benchmark/plot_ga_curve.py'
    run_command_in_new_terminal(cmd, "Ve Bieu Do Hoi Tu GA")

def launch_plot_ppo():
    cmd = f'"{VENV_PYTHON}" benchmark/plot_ppo_curve.py'
    run_command_in_new_terminal(cmd, "Ve Bieu Do PPO tu TensorBoard")

def launch_vision():
    cmd = f'"{VENV_PYTHON}" vision/main.py'
    run_command_in_new_terminal(cmd, "AI Vision - Vehicle Tracking & SUMO Route Generation")


# --- Xây dựng giao diện ---
root = tk.Tk()
root.title("Traffic Light AI - Control Panel")
root.geometry("520x720")
root.configure(padx=20, pady=20)
try:
    # Set phong cách UI cho đẹp mượt hơn
    style = ttk.Style()
    style.theme_use('clam')
except Exception:
    pass

# Header
header = tk.Label(root, text="🚥 Hệ thống Quản lý Đèn giao thông AI", font=("Arial", 16, "bold"))
header.pack(pady=(0, 5))

sub_header = tk.Label(root, text="Nhấn các nút bên dưới để khởi chạy tự động thay vì gõ lệnh", font=("Arial", 10), fg="gray")
sub_header.pack(pady=(0, 20))

# Khung chứa các tính năng nhóm theo từng phần
frame_main = ttk.Frame(root)
frame_main.pack(fill=tk.BOTH, expand=True)


# 0. Nhóm AI Vision
frame_vision = ttk.LabelFrame(frame_main, text="0. Nhận diện xe AI Vision (YOLO + Tracking)")
frame_vision.pack(fill=tk.X, pady=10, ipadx=10, ipady=10)

btn_vision = ttk.Button(frame_vision, text="🔍 Chạy AI Vision & Sinh Route SUMO", command=launch_vision)
btn_vision.pack(fill=tk.X, pady=3, padx=10)

# 1. Nhóm Huấn luyện
frame_train = ttk.LabelFrame(frame_main, text="1. Giai đoạn Huấn luyện (Training)")
frame_train.pack(fill=tk.X, pady=10, ipadx=10, ipady=10)

btn_ga = ttk.Button(frame_train, text="🚀 Huấn luyện GA (Thuật toán Di truyền)", command=launch_ga_training)
btn_ga.pack(fill=tk.X, pady=3, padx=10)

btn_ppo = ttk.Button(frame_train, text="🧠 Huấn luyện Deep RL (PPO)", command=launch_ppo_training)
btn_ppo.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=3, padx=(10, 5))

btn_tb = ttk.Button(frame_train, text="📈 Mở TensorBoard", command=launch_tensorboard)
btn_tb.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=3, padx=(5, 10))


# 2. Nhóm Đánh giá Benchmark
frame_bench = ttk.LabelFrame(frame_main, text="2. Giai đoạn Benchmark & Thông kê")
frame_bench.pack(fill=tk.X, pady=10, ipadx=10, ipady=10)

btn_run_bench = ttk.Button(frame_bench, text="📊 Chạy đọ sức Benchmark (Fixed vs GA vs PPO)", command=launch_benchmark)
btn_run_bench.pack(fill=tk.X, pady=3, padx=10)

btn_plot_bench = ttk.Button(frame_bench, text="📉 Xuất Biểu đồ Boxplot Số liệu tĩnh", command=launch_plot_benchmark)
btn_plot_bench.pack(fill=tk.X, pady=3, padx=10)


# 3. Nhóm Vẽ biểu đồ học (Learning Curves)
frame_curves = ttk.LabelFrame(frame_main, text="3. Xuất Biểu đồ Hội tụ (Vẽ Ảnh PNG)")
frame_curves.pack(fill=tk.X, pady=10, ipadx=10, ipady=10)

btn_curve_ga = ttk.Button(frame_curves, text="Trích xuất Fitness Curve (GA)", command=launch_plot_ga)
btn_curve_ga.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=3, padx=(10, 5))

btn_curve_ppo = ttk.Button(frame_curves, text="Trích xuất Learning Curve (PPO)", command=launch_plot_ppo)
btn_curve_ppo.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=3, padx=(5, 10))




# Footer
footer = tk.Label(root, text="Developed for Traffic Light Optimization System", font=("Arial", 8), fg="gray")
footer.pack(side=tk.BOTTOM, pady=10)

if __name__ == "__main__":
    root.mainloop()
