import random

class GeneticAlgorithm:
    def __init__(self, pop_size=10, mutation_rate=0.1):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        # Giới hạn gen (Bounds): [GreenNS, YellowNS, GreenEW, YellowEW]
        self.bounds = [(5, 90), (3, 5), (5, 90), (3, 5)]
        self.population = []

    def create_individual(self):
        """Tạo ra 1 cá thể với bộ gen ngẫu nhiên trong giới hạn"""
        return [random.randint(b[0], b[1]) for b in self.bounds]

    def init_population(self):
        """Khởi tạo quần thể ban đầu"""
        self.population = [self.create_individual() for _ in range(self.pop_size)]

    def selection(self, population_with_fitness):
        """Chọn lọc Tournament: Chọn ngẫu nhiên 3 cá thể, lấy cá thể tốt nhất (Fitness cao nhất)"""
        tournament = random.sample(population_with_fitness, 3)
        # tournament[i] có dạng: (chromosome, fitness_score)
        best = max(tournament, key=lambda item: item[1])
        return best[0]

    def crossover(self, parent1, parent2):
        """Lai chéo 1 điểm (Single-point Crossover) ở giữa đoạn gen"""
        split_point = 2 # Vì có 4 gen, cắt ở giữa (index 2)
        child1 = parent1[:split_point] + parent2[split_point:]
        child2 = parent2[:split_point] + parent1[split_point:]
        return child1, child2

    def mutate(self, chromosome):
        """Đột biến gen: Random lại một giá trị trong giới hạn nếu trúng tỷ lệ đột biến"""
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = random.randint(self.bounds[i][0], self.bounds[i][1])
        return chromosome