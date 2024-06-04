import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

# Función para calcular la distancia euclidiana
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

# Función para inicializar centroides aleatorios
def initialize_centroids(X, k):
    np.random.seed(42)
    random_indices = np.random.permutation(X.shape[0])
    centroids = X[random_indices[:k]]
    return centroids

# Función para asignar clusters
def assign_clusters(X, centroids):
    clusters = []
    for x in X:
        distances = [euclidean_distance(x, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

# Función para actualizar centroides
def update_centroids(X, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[clusters == i]
        if len(cluster_points) > 0:
            new_centroid = cluster_points.mean(axis=0)
        else:
            new_centroid = X[np.random.choice(X.shape[0])]
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

# Función K-means completa
def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# Fetch dataset
wine = fetch_ucirepo(id=109)

# Data (as pandas dataframes)
X = wine.data.features

# Select three attributes for clustering: 'Alcohol', 'Malicacid', 'Ash'
selected_features = ['Alcohol', 'Malicacid', 'Ash']
X_selected = X[selected_features].values

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Apply custom K-means with 3 clusters
clusters, centroids = kmeans(X_scaled, k=3)

# Add the clusters to the DataFrame
X_selected_df = pd.DataFrame(X_scaled, columns=selected_features)
X_selected_df['Cluster'] = clusters

# Visualize the results in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter = ax.scatter(X_selected_df['Alcohol'], X_selected_df['Malicacid'], X_selected_df['Ash'], c=X_selected_df['Cluster'], cmap='viridis')

# Plot centroids
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=300, c='red', marker='X', label='Centroids')

ax.set_title('Clusters de K-means en la base de datos de vino')
ax.set_xlabel('Alcohol')
ax.set_ylabel('Malic Acid')
ax.set_zlabel('Ash')
plt.legend()
plt.show()
