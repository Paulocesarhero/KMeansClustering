import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

def euclidean_distance(a, b):
    """
    Calcula la distancia euclidiana entre dos puntos.
    """
    return np.sqrt(np.sum((a - b)**2))

def initialize_centroids(X, k, random_state=42):
    """
    Inicializa los centroides aleatoriamente desde los datos.
    """
    np.random.seed(random_state)
    random_indices = np.random.permutation(X.shape[0])
    centroids = X[random_indices[:k]]
    return centroids

def assign_clusters(X, centroids):
    """
    Asigna cada punto de datos al clúster más cercano.
    """
    clusters = []
    for x in X:
        distances = [euclidean_distance(x, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

def update_centroids(X, clusters, k):
    """
    Actualiza los centroides basados en los puntos asignados a cada clúster.
    """
    new_centroids = []
    for i in range(k):
        cluster_points = X[clusters == i]
        if len(cluster_points) > 0:
            new_centroid = cluster_points.mean(axis=0)
        else:
            new_centroid = X[np.random.choice(X.shape[0])]
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

def kmeans(X, k, max_iters=100):
    """
    Ejecuta el algoritmo K-means.
    """
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# Cargar el dataset
wine = fetch_ucirepo(id=109)

# Seleccionar atributos y escalar datos
selected_features = ['Alcohol', 'Malicacid', 'Ash']
X_selected = wine.data.features[selected_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Aplicar K-means con 3 clusters (cambiar 'k' para ver el efecto)
k = 3
clusters, centroids = kmeans(X_scaled, k=k)

# Agregar los clusters al DataFrame
X_selected_df = pd.DataFrame(X_scaled, columns=selected_features)
X_selected_df['Cluster'] = clusters

# Visualizar los resultados en 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Gráfico de dispersión
scatter = ax.scatter(
    X_selected_df['Alcohol'],
    X_selected_df['Malicacid'],
    X_selected_df['Ash'],
    c=X_selected_df['Cluster'],
    cmap='viridis'
)

# Graficar centroides
ax.scatter(
    centroids[:, 0],
    centroids[:, 1],
    centroids[:, 2],
    s=300,
    c='red',
    marker='X',
    label='Centroids'
)

# Configuración del gráfico
ax.set_title(f'Clusters de K-means en la base de datos de vino (k={k})')
ax.set_xlabel('Alcohol')
ax.set_ylabel('Malic Acid')
ax.set_zlabel('Ash')
ax.legend()
plt.show()

# Imprimir centroides
print(f'Centroides para k={k}:')
print(centroids)
