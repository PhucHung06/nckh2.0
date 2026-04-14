import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulation.sumo_env import SumoEnvironment
from simulation.ga import GeneticAlgorithm

# Cau hinh
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SUMO_CFG = os.path.join(DATA_DIR, 'run1.sumocfg')
TIME_LIGHT_XML = os.path.join(DATA_DIR, 'time_light.xml')
OUTPUT_XML = os.path.join(DATA_DIR, 'dulieu_matdo.xml')

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
PLOT_DIR = os.path.join(RESULT_DIR, 'comparison_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

def run_ga_with_tracking(generations=30, pop_size=6, mutation_rate=0.1):
    env = SumoEnvironment(SUMO_CFG, TIME_LIGHT_XML, OUTPUT_XML)
    ga = GeneticAlgorithm(pop_size=pop_size, mutation_rate=mutation_rate)
    
    print("Bat dau ghi nhan qúa trinh hoi tu cua GA...")
    ga.init_population()
    
    history = []
    
    for gen in range(generations):
        pop_fit = []
        for c in ga.population:
            fitness = env.evaluate(c)
            pop_fit.append((c, fitness))
            
        pop_fit.sort(key=lambda x: x[1], reverse=True)
        
        best_fitness = pop_fit[0][1]
        mean_fitness = sum(f for _, f in pop_fit) / len(pop_fit)
        
        history.append({
            'Generation': gen + 1,
            'Best Fitness': best_fitness,
            'Mean Fitness': mean_fitness
        })
        print(f"Gen {gen+1}/{generations} - Best: {best_fitness:.2f}, Mean: {mean_fitness:.2f}")
        
        # Sinh the he moi (elitism)
        new_pop = [pop_fit[0][0]]
        while len(new_pop) < pop_size:
            p1 = ga.selection(pop_fit)
            p2 = ga.selection(pop_fit)
            c1, c2 = ga.crossover(p1, p2)
            new_pop.extend([ga.mutate(c1), ga.mutate(c2)])
        ga.population = new_pop[:pop_size]
        
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(RESULT_DIR, 'ga_learning_curve.csv'), index=False)
    return df

def plot_ga_curve(df):
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='Generation', y='Best Fitness', label='Best Fitness', marker='o')
    sns.lineplot(data=df, x='Generation', y='Mean Fitness', label='Mean Fitness', marker='s', linestyle='--')
    
    plt.title('GA - Qua trinh hoi tu (Fitness Curve)')
    plt.xlabel('The he (Generation)')
    plt.ylabel('Fitness Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = os.path.join(PLOT_DIR, 'ga_fitness_curve.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nDa luu bieu do: {out_path}")

if __name__ == '__main__':
    # Chay GA trong 30 the he de thay ro su hoi tu
    df = run_ga_with_tracking(generations=30)
    plot_ga_curve(df)
