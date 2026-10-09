import numpy as np
import scipy as sp

from utils import pairwise_dist

class KNN:
    def fit(self, data: np.ndarray, labels: np.ndarray):
        self.train_data = data
        self.train_labels = labels

    def predict(self, data: np.ndarray, k):
        dist_mat = pairwise_dist(data, self.train_data)

        sorted_examples = np.argsort(dist_mat, axis=1)
        k_sorted_examples = sorted_examples[:, :k]

        k_sorted_examples_labels = self.train_labels[k_sorted_examples]

        labels, _ = sp.stats.mode(k_sorted_examples_labels, axis=1)
        return labels

if __name__ == "__main__":
    from data_generator import GridDataSet

    dataset = GridDataSet()

    knn = KNN()
    knn.fit(dataset.train_data, dataset.train_labels)

    predicted_labels = knn.predict(dataset.test_data, k=3)

    print(f"Accuracy: {np.mean(predicted_labels == dataset.test_labels)}")
