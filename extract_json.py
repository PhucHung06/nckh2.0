import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from simulation.sumo_gym_env import SumoGymEnv

MODEL_DIR = os.path.join("d:\\khongtrongluc", "rl", "models", "ppo_traffic")
step_500_path = os.path.join(MODEL_DIR, "ppo_traffic_500_steps.zip")
final_path = os.path.join(MODEL_DIR, "final_model.zip")

# Tái tạo lại final_model.zip từ save game số 500
if os.path.exists(step_500_path):
    shutil.copy(step_500_path, final_path)

model = PPO.load(final_path)

DATA_DIR = os.path.join("d:\\khongtrongluc", "data")
env = SumoGymEnv(
    sumocfg=os.path.join(DATA_DIR, 'run1.sumocfg'),
    time_light_xml=os.path.join(DATA_DIR, 'time_light.xml'),
    output_xml=os.path.join(DATA_DIR, 'dulieu_matdo.xml')
)

obs, _ = env.reset()
best_fitness, best_chromosome = -float('inf'), None

for _ in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, _, _, info = env.step(int(action))
    if reward > best_fitness:
        best_fitness = float(reward)
        best_chromosome = [int(v) for v in info['chromosome']]

export = {
    'chromosome':  best_chromosome,
    'fitness':     best_fitness,
    'method':      'PPO',
    'description': '[GreenNS, YellowNS, GreenEW, YellowEW] in seconds'
}
out_path = os.path.join("d:\\khongtrongluc", "hardware", "config", "best_chromosome_rl.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(export, f, indent=2)

print("\n[OK] Đã trích xuất JSON:")
print(json.dumps(export, indent=2))
