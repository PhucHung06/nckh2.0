# benchmark/run_comparison.py
"""
Chạy benchmark so sánh Fixed-timing vs GA vs PPO.
Fixed (30s/30s cố định) là baseline tham chiếu cho bài báo.

Cách dùng:
  python benchmark/run_comparison.py
  python benchmark/run_comparison.py --trials 20 --ga-gens 15
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulation.sumo_env import SumoEnvironment
from simulation.ga import GeneticAlgorithm
from simulation.sumo_gym_env import SumoGymEnv
from stable_baselines3 import PPO
from benchmark.metrics import TrialResult, save_results, print_summary

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'rl',
                          'models', 'ppo_random', 'ppo_traffic_random_v2.zip')


def run_fixed_trial(env: SumoEnvironment, trial_id: int) -> TrialResult:
    """Baseline cố định: 30s NS / 4s Yellow / 30s EW / 4s Yellow."""
    chromosome = [30, 4, 30, 4]
    start = time.time()
    metrics = env.evaluate_metrics(chromosome)
    return TrialResult(trial_id=trial_id, method='Fixed',
                       fitness=metrics['fitness'], chromosome=chromosome,
                       time_s=time.time() - start,
                       avg_timeLoss=metrics['timeLoss'],
                       avg_waitingTime=metrics['waitingTime'],
                       avg_density=metrics['density'],
                       avg_speed=metrics['speed'])


def run_ga_trial(env: SumoEnvironment, trial_id: int,
                 n_gens: int = 10) -> TrialResult:
    ga = GeneticAlgorithm(pop_size=6, mutation_rate=0.1)
    ga.init_population()
    best_fitness, best_chromosome = -float('inf'), None
    start = time.time()

    for _ in range(n_gens):
        pop_fit = [(c, env.evaluate(c)) for c in ga.population]
        pop_fit.sort(key=lambda x: x[1], reverse=True)
        if pop_fit[0][1] > best_fitness:
            best_fitness, best_chromosome = pop_fit[0][1], pop_fit[0][0]
        
        ga.evolve(pop_fit)

    metrics = env.evaluate_metrics(best_chromosome)

    return TrialResult(trial_id=trial_id, method='GA',
                       fitness=metrics['fitness'], chromosome=best_chromosome,
                       time_s=time.time() - start,
                       avg_timeLoss=metrics['timeLoss'],
                       avg_waitingTime=metrics['waitingTime'],
                       avg_density=metrics['density'],
                       avg_speed=metrics['speed'])


def run_rl_trial(model: PPO, gym_env: SumoGymEnv,
                 evaluator_env: SumoEnvironment,
                 trial_id: int) -> TrialResult:
    start = time.time()
    obs, _ = gym_env.reset()
    
    terminated = False
    truncated = False
    
    # Chạy cho đến khi kết thúc mô phỏng (hết 360 steps)
    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = gym_env.step(int(action))

    # QUAN TRỌNG: Đóng TraCI để SUMO lưu và nhả file dulieu_matdo.xml ra
    gym_env.close()

    # Dùng hàm chấm điểm XML chung của GA để so sánh công bằng tuyệt đối
    metrics = evaluator_env.get_metrics()

    return TrialResult(trial_id=trial_id, method='PPO',
                       fitness=metrics['fitness'], chromosome=[0,0,0,0],
                       time_s=time.time() - start,
                       avg_timeLoss=metrics['timeLoss'],
                       avg_waitingTime=metrics['waitingTime'],
                       avg_density=metrics['density'],
                       avg_speed=metrics['speed'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials',  type=int, default=10)
    parser.add_argument('--ga-gens', type=int, default=10)
    args = parser.parse_args()

    env = SumoEnvironment(
        os.path.join(DATA_DIR, 'run1.sumocfg'),
        os.path.join(DATA_DIR, "xml/time_light.xml"),
        os.path.join(DATA_DIR, "xml/dulieu_matdo.xml"),
    )
    gym_env = SumoGymEnv(
        sumocfg = os.path.join(DATA_DIR, 'run1.sumocfg'),
        max_steps = 60, # 60 steps * 5s = 300s (5 phut)
        delta_time = 5,
        label = "benchmark"
    )
    model = PPO.load(MODEL_PATH)

    results = []
    for i in range(1, args.trials + 1):
        print(f"\n[Trial {i}/{args.trials}]")
        results.append(run_fixed_trial(env, i))
        results.append(run_ga_trial(env, i, n_gens=args.ga_gens))
        results.append(run_rl_trial(model, gym_env, env, i))

    save_results(results, RESULT_DIR)
    print_summary(results)


if __name__ == '__main__':
    main()
