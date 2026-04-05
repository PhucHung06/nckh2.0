# benchmark/visualize_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
PLOT_DIR   = os.path.join(RESULT_DIR, 'comparison_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

df      = pd.read_csv(os.path.join(RESULT_DIR, 'comparison_results.csv'))
ORDER   = ['Fixed', 'GA', 'PPO']
PALETTE = {'Fixed': '#9E9E9E', 'GA': '#2196F3', 'PPO': '#FF9800'}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Fixed vs GA vs PPO — Toi uu den giao thong',
             fontsize=13, fontweight='bold')

sns.boxplot(data=df, x='method', y='fitness', order=ORDER,
            palette=PALETTE, ax=axes[0])
axes[0].set_title('Fitness Score (cao hon = tot hon)')
axes[0].set_xlabel('Phuong phap')
axes[0].set_ylabel('Fitness Score')

sns.boxplot(data=df, x='method', y='time_s', order=ORDER,
            palette=PALETTE, ax=axes[1])
axes[1].set_title('Thoi gian tinh toan (giay)')
axes[1].set_xlabel('Phuong phap')
axes[1].set_ylabel('Thoi gian (s)')

plt.tight_layout()
out = os.path.join(PLOT_DIR, 'comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Bieu do luu: {out}")
plt.show()

print("\n=== Thong ke ===")
print(df.groupby('method')[['fitness', 'time_s']].agg(['mean', 'std']).round(4))
