import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class KMeansCustom:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.labels = None
        self.centroids = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, data):

        random_index = np.random.choice(data.shape[0], size=self.n_clusters, replace=False)
        self.centroids = data[random_index]

        for i in range(self.max_iter):

            distances = self.compute_distances(data)
            self.labels = np.argmin(distances, axis=1)
            self.plot(data, i + 1)

            new_centroids = self.find_new_centroids(data)

            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids
            

    def compute_distances(self, data):

        distances = np.zeros((data.shape[0], self.centroids.shape[0]))

        for i in range(data.shape[0]):
            for j in range(self.centroids.shape[0]):
                distances[i, j] = np.sqrt(np.sum((data[i] - self.centroids[j]) ** 2))

        return distances

    def find_new_centroids(self, data):
        new_centroids = []

        for j in range(self.n_clusters):
            cluster_data = data[self.labels == j]
            new_centroids.append(cluster_data.mean(axis=0))
        return np.array(new_centroids)

    def plot(self, data, step):
        fig, ax = plt.subplots(4, 4, figsize=(12, 8))
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                ax[i][j].scatter(data[:, i:i + 1], data[:, j:j + 1], c=self.labels, cmap='plasma')
                ax[i][j].scatter(self.centroids[:, i:i+1], self.centroids[:, j:j+1], color='red', marker='x', s=100)
        plt.suptitle(f'Step {step} - K-Means Clustering')
        plt.show()

flowers = load_iris()
iris_data = flowers.data

kmeans_custom = KMeansCustom(n_clusters=3)
kmeans_custom.fit(iris_data)