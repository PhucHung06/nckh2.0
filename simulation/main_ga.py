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
GENERATIONS = 10    # Để demo nên đặt 5-10 thế hệ cho nhanh. Chạy thật đặt 50-100.
POP_SIZE = 6       # Số lượng cá thể trong quần thể
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

        # Sắp xếp quần thể giảm dần theo fitness
        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
        print(f"Best gen {gen + 1}: {pop_with_fitness[0][0]} (Fitness: {pop_with_fitness[0][1]:.4f})")

        # 4. Sinh thế hệ mới bằng logic tiến hóa tập trung
        ga.evolve(pop_with_fitness)

    # 5. Kết luận
    print("\n=======================================================")
    print("COMPLETED TRAINING!")
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

if __name__ == "__main__":
    main()
