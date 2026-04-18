# rl/train_ppo.py
"""
Huan luyen PPO agent dieu khien den giao thong DONG (Real-time).
Phien ban su dung TraCI de tuong tac truc tiep voi SUMO moi 5 giay.

Cấu hình mới:
  - Action Space: Discrete(2) [0: Giữ đèn, 1: Chuyển pha]
  - total_timesteps: 100,000 (Tang manh de AI hoc phan xa)
  - Reward: Penalty dua tren so xe dung cho (Halting vehicles)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from simulation.sumo_gym_env import SumoGymEnv

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'ppo_dynamic')
LOG_DIR   = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def make_env():
    # Su dung cau hinh Traci moi
    env = SumoGymEnv(
        sumocfg = os.path.join(DATA_DIR, 'run1.sumocfg'),
        tl_id   = "Center",
        delta_time = 5,
        yellow_time = 4,
        min_green = 10
    )
    return Monitor(env)

def main():
    env = make_env()

    final_path = os.path.join(MODEL_DIR, 'ppo_traffic_dynamic.zip')
    
    # Khoi tao model PPO
    # Su dung device='cpu' vi mang MLP nhỏ chay tren CPU nhanh hon nạp data vao GPU
    model = PPO(
        policy          = 'MlpPolicy',
        env             = env,
        learning_rate   = 3e-4,
        n_steps         = 2048,
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

    print("--- Bat dau huan luyen RL Dong (Traci) ---")
    print("Theo doi: tensorboard --logdir rl/logs")
    
    try:
        model.learn(
            total_timesteps = 100000, 
            progress_bar    = True
        )
        
        # Luu model
        model.save(final_path)
        print(f"Model da luu tai: {final_path}")
        
    except KeyboardInterrupt:
        print("Dung training som...")
        model.save(os.path.join(MODEL_DIR, 'ppo_interrupted'))
    finally:
        env.close()

if __name__ == '__main__':
    main()
