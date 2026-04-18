import random

class GeneticAlgorithm:
    def __init__(self, pop_size=10, mutation_rate=0.1, elitism_count=2, tournament_size=3):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elitism_count = min(elitism_count, pop_size)
        self.tournament_size = max(2, min(tournament_size, pop_size))
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
        """Chọn lọc Tournament: Chọn ngẫu nhiên một số cá thể, lấy cá thể tốt nhất"""
        tournament = random.sample(population_with_fitness, self.tournament_size)
        # tournament[i] có dạng: (chromosome, fitness_score)
        best = max(tournament, key=lambda item: item[1])
        return best[0]

    def crossover(self, parent1, parent2):
        """Lai chéo 1 điểm với vị trí ngẫu nhiên"""
        split_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:split_point] + parent2[split_point:]
        child2 = parent2[:split_point] + parent1[split_point:]
        return child1, child2

    def mutate(self, chromosome):
        """Đột biến gen: Random lại một giá trị trong giới hạn nếu trúng tỷ lệ đột biến"""
        new_chromosome = list(chromosome)
        for i in range(len(new_chromosome)):
            if random.random() < self.mutation_rate:
                new_chromosome[i] = random.randint(self.bounds[i][0], self.bounds[i][1])
        return new_chromosome

    def evolve(self, population_with_fitness):
        """Tiến hóa một thế hệ mới từ quần thể hiện tại"""
        new_population = []
        
        # 1. Elitism: Giữ lại những cá thể tốt nhất (Bản sao để tránh in-place)
        population_with_fitness.sort(key=lambda item: item[1], reverse=True)
        for i in range(self.elitism_count):
            new_population.append(list(population_with_fitness[i][0]))
            
        # 2. Diversity injection: Thêm một cá thể tạo mới hoàn toàn
        if len(new_population) < self.pop_size:
            new_population.append(self.create_individual())

        # 3. Lai ghép và đột biến để lấp đầy phần còn lại
        while len(new_population) < self.pop_size:
            parent1 = self.selection(population_with_fitness)
            parent2 = self.selection(population_with_fitness)
            
            child1, child2 = self.crossover(parent1, parent2)
            
            new_population.append(self.mutate(child1))
            if len(new_population) < self.pop_size:
                new_population.append(self.mutate(child2))
                
        self.population = new_population
        return new_population