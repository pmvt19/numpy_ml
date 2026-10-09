import numpy as np
from utils import pairwise_dist


class KMeans():
    def __init__(self, k):
        self.k = k
        self.d = None

    def _generate_initial_centroids(self, data: np.ndarray):
        N, d = data.shape

        centroids = []

        # Pick first centroid
        idx = np.random.randint(N)
        centroids.append(data[idx])

        # Pick remaining 2..k centroids
        for _ in range(1, self.k):
            # Compute Distances from data to current centroids
            dist_mat = pairwise_dist(data, np.array(centroids))

            # Find the closest centroid to each data point
            min_dist_to_centroid = np.min(dist_mat, axis=1)

            # Create probabilities from closest distance to current centroid to sample 
            # each data point as the next centroid
            sampling_probability = min_dist_to_centroid / np.sum(min_dist_to_centroid)
            sampling_probability[sampling_probability < 0] = 0

            # Sample Centroid
            next_centroid_idx = np.random.choice(np.arange(N), p=sampling_probability)
            next_centroid = data[next_centroid_idx]

            centroids.append(next_centroid)

        return np.array(centroids)

    def fit(self, data: np.ndarray, max_num_iters: int = 100):
        N, d = data.shape

        prev_centroids = None
        centroids = self._generate_initial_centroids(data)
        for _ in range(max_num_iters):
            # Compute Distances between data points and current centroids
            dist_mat = pairwise_dist(data, centroids)

            # Update cluster assignment
            best_cluster_per_point = np.argmin(dist_mat, axis=1)

            # Arrange Data in Respective Assigned Cluster in an (N, d, k) matrix
            data_store = np.zeros((N, d, self.k))
            data_store[np.arange(N), :, best_cluster_per_point] = data

            # Create a parallel mask in an (N, d, k) matrix to track which elements were
            # written to
            data_store_mask = np.zeros((N, d, self.k)).astype(np.bool)
            data_store_mask[np.arange(N), :, best_cluster_per_point] = True

            # Compute new centroids
            points_per_cluster = np.sum(np.any(data_store_mask, axis=1), axis=0).reshape(-1, 1) # (k, 1)
            summed_data_points_per_cluster = np.sum(data_store, axis=0).T # (k, d)
            new_centroids = summed_data_points_per_cluster / points_per_cluster

            # Update Centroids
            prev_centroids = centroids
            centroids = new_centroids

            if np.all(prev_centroids == new_centroids):
                break

        self.centroids = centroids
        return centroids

    def predict(self, data: np.ndarray):
        # Compute pairwise distances between data and computed centroids
        dist_mat = pairwise_dist(data, self.centroids)

        # Find the closest centroid to each data point and return that as the label
        labels = np.argmin(dist_mat, axis=1)
        return labels

if __name__ == "__main__": 
    from data_generator import GridDataSet
    import matplotlib.pyplot as plt

    dataset = GridDataSet()
    kmeans = KMeans(9)

    centroids = kmeans.fit(dataset.train_data)
    labels = kmeans.predict(dataset.test_data)

    plt.scatter(dataset.test_data[:, 0], dataset.test_data[:, 1], c=labels)
    plt.show()