import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Cau hinh duong dan
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'rl', 'logs2.0')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
PLOT_DIR = os.path.join(RESULT_DIR, 'comparison_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

def get_valid_tfevents(log_dir, tag='rollout/ep_rew_mean'):
    """Tim thu muc log PPO gan nhat thuc su co data (chua tag can tim)"""
    subdirs = glob.glob(os.path.join(log_dir, 'PPO_*'))
    subdirs.sort(key=os.path.getmtime, reverse=True)
    
    for subdir in subdirs:
        event_files = glob.glob(os.path.join(subdir, 'events.out.tfevents.*'))
        if event_files:
            # Thu kiem tra xem co data khong
            ea = EventAccumulator(event_files[0])
            ea.Reload()
            if tag in ea.Tags()['scalars']:
                return event_files[0]
                
    return None

def extract_tb_data(event_file, tag='rollout/ep_rew_mean'):
    """Trich xuat du lieu tu TensorBoard"""
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    if tag not in ea.Tags()['scalars']:
        print(f"Khong tim thay tag '{tag}' trong {event_file}")
        return pd.DataFrame()
        
    events = ea.Scalars(tag)
    
    # Chuyen thanh DataFrame
    steps = [e.step for e in events]
    values = [e.value for e in events]
    
    # Gia lap std (ban chat tb chi luu mean)
    df = pd.DataFrame({'Timestep': steps, 'Mean Reward': values})
    if len(df) > 5:
        df['Reward Std'] = df['Mean Reward'].rolling(window=5, min_periods=1).std().fillna(0)
    else:
        df['Reward Std'] = 0.0
        
    return df

def plot_ppo_curve(df):
    if df.empty:
        print("Khong co du lieu de ve bieu do!")
        return
        
    plt.figure(figsize=(9, 6))
    sns.set_style("whitegrid")
    
    sns.lineplot(data=df, x='Timestep', y='Mean Reward', color='#3498db', label='Mean Episode Reward', linewidth=2)
    
    # Ve dải std
    plt.fill_between(df['Timestep'], 
                     df['Mean Reward'] - df['Reward Std'], 
                     df['Mean Reward'] + df['Reward Std'], 
                     color='#3498db', alpha=0.15, label='± 1 Std Dev')
                     
    plt.title('PPO Deep RL - Learning Convergence Curve', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Mean Episode Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, shadow=True, loc='lower right')
    
    out_path = os.path.join(PLOT_DIR, 'ppo_learning_curve.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Luu ca file CSV
    csv_path = os.path.join(RESULT_DIR, 'ppo_learning_curve.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Da luu bieu do vao: {out_path}")
    print(f"Da luu data vao: {csv_path}")

if __name__ == '__main__':
    tag_name = 'rollout/ep_rew_mean'
    event_file = get_valid_tfevents(LOG_DIR, tag=tag_name)
    
    if not event_file:
        print("Khong tim thay log TensorBoard nao co du lieu huan luyen hop le. Ban can chay train_ppo.py truoc!")
    else:
        print(f"Dang doc du lieu tu (chi lay log co du lieu): {event_file}")
        df = extract_tb_data(event_file, tag=tag_name)
        plot_ppo_curve(df)
