import numpy as np
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dist(self, p):
        return np.sqrt((p.x - self.x) ** 2 + (p.y - self.y) ** 2)

class CustomDBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = []
        self.visited = []

    def fit(self, points):
        self.labels = [0] * len(points)  # Инициализация меток
        cluster_id = 0

        for i in range(len(points)):
            if self.labels[i] != 0:  # Если точка уже помечена, пропускаем
                continue

            neighbors = self.find_neighbors(points, i)

            if len(neighbors) < self.min_samples:  # Шум
                self.labels[i] = -1
            else:
                cluster_id += 1
                self.expand_cluster(points, i, neighbors, cluster_id)

    def find_neighbors(self, points, point_idx):
        neighbors = []
        point = points[point_idx]

        for i in range(len(points)):
            if point.dist(points[i]) < self.eps:
                neighbors.append(i)

        return neighbors

    def expand_cluster(self, points, point_idx, neighbors, cluster_id):
        self.labels[point_idx] = cluster_id
        i = 0

        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if self.labels[neighbor_idx] == -1:  # Если это шум
                self.labels[neighbor_idx] = cluster_id  # Присваиваем кластер

            if self.labels[neighbor_idx] == 0:  # Если точка не помечена
                self.labels[neighbor_idx] = cluster_id
                new_neighbors = self.find_neighbors(points, neighbor_idx)

                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)

            i += 1

