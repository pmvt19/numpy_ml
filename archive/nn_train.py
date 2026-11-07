import numpy as np
import matplotlib.pyplot as plt


def train(self, epochs: int, batch_size: int, data: np.ndarray, labels: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray, show_training_curve: bool = False):
    losses = []
    test_losses = []

    acc = []
    test_acc = []

    N, d = data.shape
    for i in range(epochs):
        # Choose random batch
        idxes = np.random.choice(N, size=(batch_size,))

        batch_data = data[idxes]
        batch_labels = labels[idxes]

        # Forward Pass of Network
        yhat, loss = self.forward(batch_data, batch_labels)

        # Backward Pass and Gradient Update
        self.backward()

        # Clear the gradient caches
        self.clear()

        print(f"Loss: {loss}")
        losses.append(loss)

        # Compute Accuracy for Train and Test Data
        yhat, _ = self.forward(data, None)
        accuracy = np.mean((yhat > 0.5).astype(np.int32) == labels)
        acc.append(accuracy)

        yhat, test_loss = self.forward(test_data, test_labels)
        test_accuracy = np.mean((yhat > 0.5).astype(np.int32) == test_labels)
        test_acc.append(test_accuracy)
        test_losses.append(test_loss)

    if show_training_curve:
        plt.plot(losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curves while Training")
        plt.savefig('assets/nn_training_curve.svg')

        plt.clf()
        plt.plot(acc, label='Training Accuracy')
        plt.plot(test_acc, label='Testing Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Training and Testing Accuracies")
        plt.savefig('assets/nn_accuracy_curve.svg')