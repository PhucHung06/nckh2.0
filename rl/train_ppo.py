# rl/train_ppo.py
"""
Huấn luyện PPO agent tối ưu thời gian đèn giao thông.

Cấu hình mặc định phù hợp với SUMO ~10s/step:
  - n_steps=64, batch_size=16: mini-batch nhỏ để không tốn RAM
  - total_timesteps=500: khoảng 500 lần chạy SUMO (~1-2h)
  Tăng lên 2000+ khi có thời gian để model hội tụ tốt hơn.

Output:
  rl/models/ppo_traffic/final_model.zip     -- dùng cho benchmark
  rl/logs/                                  -- TensorBoard logs
  hardware/config/best_chromosome_rl.json   -- export cho Pi5 sau này
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from simulation.sumo_gym_env import SumoGymEnv

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'ppo_traffic')
LOG_DIR   = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def make_env():
    return Monitor(SumoGymEnv(
        sumocfg        = os.path.join(DATA_DIR, 'run1.sumocfg'),
        time_light_xml = os.path.join(DATA_DIR, "xml/time_light.xml"),
        output_xml     = os.path.join(DATA_DIR, "xml/dulieu_matdo.xml"),
    ))


def main():
    env = make_env()

    final_path = os.path.join(MODEL_DIR, 'final_model.zip')
    
    if os.path.exists(final_path):
        print(f"Phat hien checkpoint cu tai {final_path}.")
        print("Tiep tuc Huan luyen (Resume Training)...")
        model = PPO.load(final_path, env=env, tensorboard_log=LOG_DIR)
        reset_timesteps = False
    else:
        print("Bat dau huan luyen PPO tu dau (Training from scratch)...")
        model = PPO(
            policy          = 'MlpPolicy',
            env             = env,
            learning_rate   = 3e-4,
            n_steps         = 64,
            batch_size      = 16,
            n_epochs        = 5,
            gamma           = 0.99,
            ent_coef        = 0.01,
            verbose         = 1,
            tensorboard_log = LOG_DIR,
        )
        reset_timesteps = True

    checkpoint_cb = CheckpointCallback(
        save_freq   = 50,
        save_path   = MODEL_DIR,
        name_prefix = 'ppo_traffic',
    )

    print("Theo doi: tensorboard --logdir rl/logs")
    model.learn(
        total_timesteps = 500,
        callback        = checkpoint_cb,
        progress_bar    = True,
        reset_num_timesteps = reset_timesteps
    )

    final_path = os.path.join(MODEL_DIR, 'final_model')
    model.save(final_path)
    print(f"Model da luu: {final_path}.zip")

    # Tự động export chromosome tốt nhất cho hardware sau này
    _export_best_chromosome(model, env)


def _export_best_chromosome(model, env):
    """
    Chạy model inference, tìm chromosome cho fitness cao nhất,
    lưu ra JSON để hardware/pi_controller.py đọc khi có Pi5.
    """
    import json
    obs, _ = env.reset()
    best_fitness, best_chromosome = -float('inf'), None

    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, info = env.step(int(action))
        if reward > best_fitness:
            best_fitness = reward
            best_chromosome = info['chromosome']

    export = {
        'chromosome':  best_chromosome,
        'fitness':     best_fitness,
        'method':      'PPO',
        'description': '[GreenNS, YellowNS, GreenEW, YellowEW] in seconds'
    }
    out_path = os.path.join(os.path.dirname(__file__), '..', 'hardware',
                            'config', 'best_chromosome_rl.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(export, f, indent=2)
    print(f"Best RL chromosome: {best_chromosome} (fitness={best_fitness:.4f})")
    print(f"Da luu cho hardware: {out_path}")


if __name__ == '__main__':
    main()
