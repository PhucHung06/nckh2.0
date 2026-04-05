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
                          'models', 'ppo_traffic', 'final_model')


def run_fixed_trial(env: SumoEnvironment, trial_id: int) -> TrialResult:
    """Baseline cố định: 30s NS / 4s Yellow / 30s EW / 4s Yellow."""
    chromosome = [30, 4, 30, 4]
    start = time.time()
    fitness = env.evaluate(chromosome)
    return TrialResult(trial_id=trial_id, method='Fixed',
                       fitness=fitness, chromosome=chromosome,
                       time_s=time.time() - start)


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
        new_pop = [pop_fit[0][0]]
        while len(new_pop) < 6:
            p1 = ga.selection(pop_fit)
            p2 = ga.selection(pop_fit)
            c1, c2 = ga.crossover(p1, p2)
            new_pop.extend([ga.mutate(c1), ga.mutate(c2)])
        ga.population = new_pop[:6]

    return TrialResult(trial_id=trial_id, method='GA',
                       fitness=best_fitness, chromosome=best_chromosome,
                       time_s=time.time() - start)


def run_rl_trial(model: PPO, gym_env: SumoGymEnv,
                 trial_id: int) -> TrialResult:
    start = time.time()
    obs, _ = gym_env.reset()
    best_fitness, best_chromosome = -float('inf'), None

    for _ in range(20):    # 20 inference steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, info = gym_env.step(int(action))
        if reward > best_fitness:
            best_fitness = reward
            best_chromosome = info['chromosome']

    return TrialResult(trial_id=trial_id, method='PPO',
                       fitness=best_fitness, chromosome=best_chromosome,
                       time_s=time.time() - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials',  type=int, default=10)
    parser.add_argument('--ga-gens', type=int, default=10)
    args = parser.parse_args()

    env = SumoEnvironment(
        os.path.join(DATA_DIR, 'run1.sumocfg'),
        os.path.join(DATA_DIR, 'time_light.xml'),
        os.path.join(DATA_DIR, 'dulieu_matdo.xml'),
    )
    gym_env = SumoGymEnv(
        os.path.join(DATA_DIR, 'run1.sumocfg'),
        os.path.join(DATA_DIR, 'time_light.xml'),
        os.path.join(DATA_DIR, 'dulieu_matdo.xml'),
    )
    model = PPO.load(MODEL_PATH)

    results = []
    for i in range(1, args.trials + 1):
        print(f"\n[Trial {i}/{args.trials}]")
        results.append(run_fixed_trial(env, i))
        results.append(run_ga_trial(env, i, n_gens=args.ga_gens))
        results.append(run_rl_trial(model, gym_env, i))

    save_results(results, RESULT_DIR)
    print_summary(results)


if __name__ == '__main__':
    main()
