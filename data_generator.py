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


class LinearDataSet:
    def __init__(self):
        self.m = 6.3
        self.b = 92.4

        self.lower_bound = -250
        self.higher_bound = 250

        self.train_data_size = (200, )
        self.test_data_size = (100, )

        self.noise_scale = 50

        self.train_data = np.random.uniform(self.lower_bound, self.higher_bound, size=self.train_data_size)
        self.test_data = np.random.uniform(self.lower_bound, self.higher_bound, size=self.test_data_size)

        self.train_labels_clean = self.train_data * self.m + self.b
        self.train_labels_noisy = self.train_labels_clean + np.random.normal(scale=self.noise_scale, size=self.train_data_size)

        self.test_labels_clean = self.test_data * self.m + self.b
        self.test_labels_noisy = self.test_labels_clean + np.random.normal(scale=self.noise_scale, size=self.test_data_size)

    def visualize(self):
        # Plot Training Data
        plt.scatter(self.train_data, self.train_labels_clean, color='blue', label='Clean Labels')
        plt.scatter(self.train_data, self.train_labels_noisy, color='red', label='Noisy Labels')
        plt.title("Linear Training Data")
        plt.legend()
        plt.show()

        # Plot Testing Data
        plt.scatter(self.test_data, self.test_labels_clean, color='blue', label='Clean Labels')
        plt.scatter(self.test_data, self.test_labels_noisy, color='red', label='Noisy Labels')
        plt.title("Linear Testing Data")
        plt.legend()
        plt.show()

class BinaryClassificationDataset:
    def __init__(self):

        self.class_0_mean = (5, 5)
        self.class_0_scale = 3

        self.class_1_mean = (-2, -4)
        self.class_1_scale = 3

        self.train_class_size = 100
        self.test_class_size = 20

        class0_train_data = np.random.normal(loc=self.class_0_mean, scale=self.class_0_scale, size=(self.train_class_size,2))
        class1_train_data = np.random.normal(loc=self.class_1_mean, scale=self.class_1_scale, size=(self.train_class_size,2))

        class0_test_data = np.random.normal(loc=self.class_0_mean, scale=self.class_0_scale, size=(self.test_class_size,2))
        class1_test_data = np.random.normal(loc=self.class_1_mean, scale=self.class_1_scale, size=(self.test_class_size,2))

        self.train_data = np.vstack((class0_train_data, class1_train_data))
        self.test_data = np.vstack((class0_test_data, class1_test_data))

        self.train_labels = np.vstack((np.zeros((self.train_class_size, 1)), np.ones((self.train_class_size, 1))))
        self.test_labels = np.vstack((np.zeros((self.test_class_size, 1)), np.ones((self.test_class_size, 1))))

    def visualize(self):
        # Plot Training Data
        plt.scatter(self.train_data[:, 0], self.train_data[:, 1], c=self.train_labels)
        plt.show()

        # Plot Testing Data
        plt.scatter(self.test_data[:, 0], self.test_data[:, 1], c=self.test_labels)
        plt.show()

        # Plotting Train & Test Data
        plt.scatter(self.train_data[:, 0], self.train_data[:, 1], c=self.train_labels)
        plt.scatter(self.test_data[:, 0], self.test_data[:, 1], c=self.test_labels, alpha=0.3)
        plt.title("Labeled Training & Testing Data")
        plt.show()

if __name__ == "__main__":
    dataset = BinaryClassificationDataset()
    dataset.visualize()
