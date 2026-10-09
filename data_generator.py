import numpy as np
import matplotlib.pyplot as plt

class GridDataSet:
    def __init__(self):
        # Create Cluster Ground Truth Means
        means = []
        for i in range(1, 4):
            for j in range(1, 4):
                means.append((i,j))

        train_data = []
        test_data = []

        scale = 0.3
        num_train_points_per_class = 80
        num_test_points_per_class = 30

        for label, mean in enumerate(means):
            data = np.random.normal(loc=mean, scale=scale, size=(num_train_points_per_class,2))
            labeled_data = np.hstack((data, np.ones((num_train_points_per_class, 1)) * label))

            data = np.random.normal(loc=mean, scale=scale, size=(num_test_points_per_class,2))
            labeled_test_data = np.hstack((data, np.ones((num_test_points_per_class, 1)) * label))

            train_data.append(labeled_data)
            test_data.append(labeled_test_data)

        self.train_data = np.vstack(train_data)
        self.test_data = np.vstack(test_data)

        # Separate Data and Labels
        self.train_labels = self.train_data[:, -1]
        self.test_labels = self.test_data[:, -1]

        self.train_data = self.train_data[:, :2]
        self.test_data = self.test_data[:, :2]

    def visualize(self):
        # Plot Training data
        plt.scatter(self.train_data[:, 0], self.train_data[:, 1], c=self.train_labels)
        plt.title("Labeled Training Data")
        plt.show()

        # Plot Test data
        plt.scatter(self.test_data[:, 0], self.test_data[:, 1], c=self.test_labels)
        plt.title("Labeled Testing Data")
        plt.show()

        # Plotting Train & Test Data
        plt.scatter(self.train_data[:, 0], self.train_data[:, 1], c=self.train_labels)
        plt.scatter(self.test_data[:, 0], self.test_data[:, 1], c=self.test_labels, alpha=0.3)
        plt.title("Labeled Training & Testing Data")
        plt.show()

if __name__ == "__main__":
    dataset = GridDataSet()
    dataset.visualize()
