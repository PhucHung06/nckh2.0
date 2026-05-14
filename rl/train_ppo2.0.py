# rl/train_ppo2.0.py
"""
Huấn luyện PPO agent trên dữ liệu giao thông NGẪU NHIÊN (Random Flow).
Model được lưu vào thư mục riêng logs2.0 để tránh ghi đè kết quả cũ.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from simulation.sumo_gym_env import SumoGymEnv

# Cấu hình đường dẫn mới
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'ppo_random')
LOG_DIR   = os.path.join(os.path.dirname(__file__), 'logs2.0')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def make_env(label="default"):
    env = SumoGymEnv(
        sumocfg     = os.path.join(DATA_DIR, 'run_random.sumocfg'), # Dùng config ngẫu nhiên
        tl_id       = "Center",
        delta_time  = 5,
        yellow_time = 4,
        min_green   = 10,
        max_steps   = 720, # 1 tiếng mô phỏng (720 steps * 5s = 3600s)
        label       = label
    )
    return Monitor(env)

def main():
    # 1. Khởi tạo môi trường huấn luyện
    env = make_env(label="train_random")

    final_path = os.path.join(MODEL_DIR, 'ppo_traffic_random_v2.zip')
    
    # 2. Môi trường đánh giá
    eval_env = make_env(label="eval_random")
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR, 
        eval_freq=5000, # Đánh giá sau mỗi 5000 bước
        deterministic=True, 
        render=False
    )
    
    # 3. Khởi tạo model PPO
    model = PPO(
        policy          = 'MlpPolicy',
        env             = env,
        learning_rate   = 3e-4, 
        n_steps         = 2048, # Tăng n_steps để học tốt hơn trên dữ liệu ngẫu nhiên
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,
        verbose         = 1,
        tensorboard_log = LOG_DIR,
        device          = 'cpu' 
    )

    print("--- Starting PPO 2.0 Training (Random Traffic Flow) ---")
    print(f"Logs will be saved at: {LOG_DIR}")
    print(f"Models will be saved at: {MODEL_DIR}")
    
    try:
        model.learn(
            total_timesteps = 100000, # Training 100k steps
            progress_bar    = True,
            callback        = eval_callback
        )
        
        model.save(final_path)
        print(f"✅ Model 2.0 saved at: {final_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted.")
        model.save(os.path.join(MODEL_DIR, 'ppo_random_interrupted'))
    finally:
        env.close()
        eval_env.close()

if __name__ == '__main__':
    main()
