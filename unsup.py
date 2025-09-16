from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# generate synthetic data with make_blobs
from sklearn.datasets import make_blobs
data, labels = make_blobs(n_samples=300, centers=12, cluster_std=0.60, random_state=42)

# # show the data
# plt.scatter(data[:, 0], data[:, 1], s=30)
# plt.show()

# # use elbow method to find the optimal number of clusters
# inertia = []
# K = range(1, 15)
# for k in K:
#     kmeans = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300, random_state=42)
#     kmeans.fit(data)
#     inertia.append(kmeans.inertia_)

# # plot the elbow curve
# plt.figure(figsize=(8, 4))
# plt.plot(K, inertia, 'bx-')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method For Optimal k')
# plt.show()

# use silhouette score to find the optimal number of clusters
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
silhouette_scores = []
K = range(2, 15)
for k in K:
    kmeans = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300, random_state=42)
    labels = kmeans.fit_predict(data)
    silhouette_scores.append(silhouette_score(data, labels))
# plot the silhouette scores
plt.figure(figsize=(8, 4))
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Optimal k')
plt.show()

# best k
best_k = K[np.argmax(silhouette_scores)]

# fit the KMeans model
kmeans = KMeans(n_clusters=8, init='random', n_init=10, max_iter=300, random_state=42)

predictions = kmeans.fit_predict(data)
centers = kmeans.cluster_centers_
print(centers)
print(predictions)

# plot the clusters with sns
# plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predictions, palette='Set1', s=50, alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=150, marker='X', label='Centroids')
# plt.title('KMeans Clustering')
# plt.legend()
plt.show()