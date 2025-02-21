# /classification/clustering_DTW.py
import numpy as np
from dtw import dtw
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class DTWClustering:
    def __init__(self, n_clusters=3):
        """
        Initialize the DTW clustering.

        Args:
            n_clusters (int): Number of clusters for k-means.
        """
        self.n_clusters = n_clusters
        self.distances = None

    def compute_dtw_distances(self, data):
        """
        Compute DTW distances between all pairs of time series.

        Args:
            data (np.ndarray): Input data of shape (n_samples, n_timesteps, n_features).
        """
        n = len(data)
        self.distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist, _, _, _ = dtw(data[i], data[j], dist=euclidean)
                self.distances[i, j] = dist

    def cluster_and_visualize(self, data, n_components=2):
        """
        Perform clustering and visualize the results.

        Args:
            data (np.ndarray): Input data of shape (n_samples, n_timesteps, n_features).
            n_components (int): Number of components for t-SNE (2 or 3).
        """
        # Compute DTW distances
        self.compute_dtw_distances(data)

        # Perform clustering
        kmeans = KMeans(n_clusters=self.n_clusters)
        clusters = kmeans.fit_predict(self.distances)

        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=n_components, random_state=42)
        reduced_distances = tsne.fit_transform(self.distances)

        # Plot the clusters
        if n_components == 2:
            plt.scatter(reduced_distances[:, 0], reduced_distances[:, 1], c=clusters, cmap='viridis')
            plt.colorbar()
            plt.title('Clustering of Movement Data (2D)')
        elif n_components == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(reduced_distances[:, 0], reduced_distances[:, 1], reduced_distances[:, 2], c=clusters, cmap='viridis')
            plt.title('Clustering of Movement Data (3D)')
        plt.show()

if __name__ == "__main__":
    movement_vectors = {
        'standing': np.load('output_vectors/standing_vectors.npy'),
        'sitting': np.load('output_vectors/sitting_vectors.npy'),
        'falling': np.load('output_vectors/falling_vectors.npy')
    }
    all_vectors = np.vstack(list(movement_vectors.values()))
    clustering = DTWClustering(n_clusters=3)
    clustering.cluster_and_visualize(all_vectors, n_components=2)