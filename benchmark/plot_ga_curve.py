import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cau hinh
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
HISTORY_PATH = os.path.join(DATA_DIR, 'ga_history.csv')

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
PLOT_DIR = os.path.join(RESULT_DIR, 'comparison_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_ga_curve():
    if not os.path.exists(HISTORY_PATH):
        print(f"❌ Khong tim thay file log tai: {HISTORY_PATH}")
        print("Ban can chay 'python simulation/main_ga.py' truoc de sinh ra du lieu.")
        return

    print(f"📊 Dang doc lich su hoi tu GA tu: {HISTORY_PATH}")
    df = pd.read_csv(HISTORY_PATH)

    plt.figure(figsize=(9, 6))
    sns.set_style("whitegrid")
    
    # Ve Best Fitness (Indigo)
    sns.lineplot(data=df, x='Generation', y='Best Fitness', label='Best Fitness (SOTA)', 
                 marker='o', color='#5d5fef', linewidth=2.5)
    
    # Ve Mean Fitness (Teal)
    sns.lineplot(data=df, x='Generation', y='Mean Fitness', label='Mean Fitness', 
                 marker='s', linestyle='--', color='#24b0ba', alpha=0.7)
    
    plt.title('Genetic Algorithm - Convergence Analysis', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Generation (Epoch)', fontsize=12)
    plt.ylabel('Fitness Score (Higher is Better)', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, shadow=True, loc='lower right')
    
    out_path = os.path.join(PLOT_DIR, 'ga_fitness_curve.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Luu phien ban csv sang thu muc results de de quan ly
    df.to_csv(os.path.join(RESULT_DIR, 'ga_learning_curve.csv'), index=False)
    
    print(f"✅ Da luu bieu do vao: {out_path}")

if __name__ == '__main__':
    plot_ga_curve()
