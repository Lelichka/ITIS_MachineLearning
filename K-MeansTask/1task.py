import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

flowers = load_iris()
data = flowers.data

J_array = []
k_values = range(1, 9)

for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    #print(kmeans.inertia_)
    J_array.append(kmeans.inertia_)

plt.plot(k_values, J_array, marker='o')
plt.xlabel('k')
plt.ylabel('J(k)')
plt.show()


