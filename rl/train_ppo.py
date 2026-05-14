# rl/train_ppo.py
"""
Huấn luyện PPO agent điều khiển đèn giao thông ĐƠN LUỒNG (Single-threaded).
Phù hợp cho máy tính cấu hình trung bình hoặc khi cần debug quan sát dễ dàng.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from simulation.sumo_gym_env import SumoGymEnv

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'ppo_dynamic')
LOG_DIR   = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def make_env(label="default"):
    env = SumoGymEnv(
        sumocfg = os.path.join(DATA_DIR, 'run1.sumocfg'),
        tl_id   = "Center",
        delta_time = 5,
        yellow_time = 4,
        min_green = 10,
        max_steps = 120,  # 1 phút = 60s / 5s = 12 steps
        label = label
    )
    return Monitor(env)

def main():
    # 1. Khởi tạo môi trường đơn
    env = make_env(label="train")

    final_path = os.path.join(MODEL_DIR, 'ppo_traffic_dynamic.zip')
    
    # 2. Môi trường đánh giá
    eval_env = make_env(label="eval")
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR, 
        eval_freq=2000,
        deterministic=True, 
        render=False
    )
    
    # 3. Khởi tạo model PPO (Cấu hình ổn định)
    model = PPO(
        policy          = 'MlpPolicy',
        env             = env,
        learning_rate   = 3e-4, 
        n_steps         = 1024,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.1,
        ent_coef        = 0.01,
        verbose         = 1,
        tensorboard_log = LOG_DIR,
        device          = 'cpu' 
    )

    print("--- Bắt đầu huấn luyện RL Đơn luồng (Traci) ---")
    
    try:
        model.learn(
            total_timesteps = 10000, 
            progress_bar    = True,
            callback        = eval_callback
        )
        
        model.save(final_path)
        print(f"✅ Model đã lưu tại: {final_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Dừng training sớm.")
        model.save(os.path.join(MODEL_DIR, 'ppo_interrupted'))
    finally:
        env.close()
        eval_env.close()

if __name__ == '__main__':
    main()
