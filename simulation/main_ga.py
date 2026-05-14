import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulation.sumo_env import SumoEnvironment
from simulation.ga import GeneticAlgorithm

# Cấu hình đường dẫn
DATA_DIR       = os.path.join(os.path.dirname(__file__), '..', 'data')
SUMO_CFG       = os.path.join(DATA_DIR, 'run1.sumocfg')
TIME_LIGHT_XML = os.path.join(DATA_DIR, 'xml/time_light.xml')
OUTPUT_XML     = os.path.join(DATA_DIR, 'xml/dulieu_matdo.xml')

# Tham số cấu hình GA
GENERATIONS = 30  # Để demo nên đặt 5-10 thế hệ cho nhanh. Chạy thật đặt 50-100.
POP_SIZE = 8   # Số lượng cá thể trong quần thể
MUTATION_RATE = 0.1 # Tỷ lệ đột biến 10%

def main():
    print("Starting Traffic Light Optimization using GA & SUMO...")
    
    # 1. Khởi tạo môi trường SUMO và Thuật toán GA
    env = SumoEnvironment(SUMO_CFG, TIME_LIGHT_XML, OUTPUT_XML)
    ga = GeneticAlgorithm(pop_size=POP_SIZE, mutation_rate=MUTATION_RATE)
    
    # 2. Sinh quần thể ban đầu
    ga.init_population()
    
    best_overall_chromosome = None
    best_overall_fitness = -float('inf')
    ga_history = []

    # 3. Vòng lặp tiến hóa
    for gen in range(GENERATIONS):
        print(f"\n--- Thế hệ {gen + 1}/{GENERATIONS} ---")
        
        # Đánh giá Fitness cho từng cá thể trong quần thể
        pop_with_fitness = []
        for i, chromosome in enumerate(ga.population):
            # Chỗ này sẽ tốn thời gian vì SUMO chạy mô phỏng
            fitness = env.evaluate(chromosome)
            pop_with_fitness.append((chromosome, fitness))
            print(f"  Cá thể {i+1} {chromosome} -> Fitness: {fitness:.4f}")
            
            # Cập nhật cá thể tốt nhất toàn cục
            if fitness > best_overall_fitness:
                best_overall_fitness = fitness
                best_overall_chromosome = chromosome.copy()

        # Thống kê thế hệ
        current_gen_fitnesses = [x[1] for x in pop_with_fitness]
        ga_history.append({
            'Generation': gen + 1,
            'Best Fitness': max(current_gen_fitnesses),
            'Mean Fitness': sum(current_gen_fitnesses) / len(current_gen_fitnesses)
        })

        # Sắp xếp quần thể giảm dần theo fitness
        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
        print(f"Best gen {gen + 1}: {pop_with_fitness[0][0]} (Fitness: {pop_with_fitness[0][1]:.4f})")

        # 4. Sinh thế hệ mới bằng logic tiến hóa tập trung
        ga.evolve(pop_with_fitness)

    # 5. Kết luận
    print("\n=======================================================")
    print("COMPLETED TRAINING!")
    
    # Lưu lịch sử hội tụ để vẽ biểu đồ
    import pandas as pd
    history_path = os.path.join(DATA_DIR, 'ga_history.csv')
    pd.DataFrame(ga_history).to_csv(history_path, index=False)
    print(f"Lịch sử hội tụ đã lưu tại: {history_path}")

    print(f"Optimal chromosome (Light durations): {best_overall_chromosome}")
    print(f"Best Fitness score: {best_overall_fitness:.4f}")
    
    # Ghi lại bộ tốt nhất vào file lần cuối để lưu giữ
    print("Dang luu cau hinh tot nhat vao time_light.xml va cap nhat ngatu.net.xml...")
    env.write_time_light_xml(best_overall_chromosome)

    # Xuất cấu hình cho Hardware
    out_path = os.path.join(os.path.dirname(__file__), '..', 'hardware', 'config', 'best_chromosome_ga.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import json
    export_data = {
        'chromosome': best_overall_chromosome,
        'fitness': best_overall_fitness,
        'method': 'GA',
        'description': '[GreenNS, YellowNS, GreenEW, YellowEW] in seconds'
    }
    with open(out_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"Cấu hình GA đã xuất cho hardware tại: {out_path}")

    # 6. Tự động cập nhật code cho Dashboard
    update_dashboard_ga_config(best_overall_chromosome)

def update_dashboard_ga_config(best_chromosome):
    dashboard_path = os.path.join(os.path.dirname(__file__), '..', 'demo_live_twin_v2.py')
    if not os.path.exists(dashboard_path):
        return
    
    print(f"🔄 Đang tự động cập nhật GA_CHROMOSOME vào {os.path.basename(dashboard_path)}...")
    try:
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        import re
        # Tìm dòng GA_CHROMOSOME = [...]
        new_content = re.sub(
            r'GA_CHROMOSOME = \[[^\]]+\]', 
            f'GA_CHROMOSOME = {best_chromosome}', 
            content
        )
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Đã cập nhật GA_CHROMOSOME thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi cập nhật Dashboard: {e}")

if __name__ == "__main__":
    main()
