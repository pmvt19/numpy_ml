import numpy as np

class LinearLayer:
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.normal(size=(input_size, output_size)) # (d, m)
        self.bias = np.random.normal(size=(output_size, 1)) # (m, 1)

        self.cache = {}

    def forward(self, data: np.ndarray):
        """
            data: (B, d) where B is batch size and d is the dimensionality of the data
        """

        h = data @ self.weights + self.bias.T # (B, d) X (d, m) + (m, 1) -> (B, m)

        # Used for gradient updates
        self.cache['data'] = data
        return h

    def backward(self, de_dh: np.ndarray, lr: float = 1e-3):
        """
            de_dh: (B, m)

            Need de_dw: 
                de_dh * dh_dw = de_dw

                dh_dw = data

                Size/Dimension Analysis:

                de_dh * dh_dw = de_dw
                de_dh * data  = de_dw
                (B, m) (B, d) = (m, d)
            
            Need de_db:
                de_dh * dh_db = de_db

                dh_db = 1

                Size/Dimension Analysis:

                de_dh * dh_db = de_db
                de_dh * 1s  = de_db
                (B, m) (B, 1) = (m, 1)

            Need de_dx (where x is the input or 'data' in this case)
                de_dx = de_dh * dh_dx

                dh_dx = self.weights

                Size/Dimension Analysis:
                de_dx = de_dh * dh_dx
                (B, d) = (B, m) * (d, m)
        """

        data = self.cache['data']
        B, d = data.shape

        # Compute Downstream Gradient
        de_dx = de_dh @ self.weights.T

        # Compute Gradient for local weight update
        de_dw = (de_dh.T @ data).T # (d, m)
        de_db = de_dh.T @ np.ones((B, 1)) # (m, 1)

        # Step the weights in the opposite direction of greatest ascent
        self.weights = self.weights - lr * de_dw
        self.bias = self.bias - lr * de_db

        return de_dx

    def clear(self):
        """
            Clears the cache state
        """
        self.cache = {}


class ReLu:
    def __init__(self):
        self.cache = {}

    def forward(self, data: np.ndarray):
        """
            data: (B, d)
        """
        # Copy Data to not disrupt potentially cached values that would share the same
        # memory address
        copied_data = data.copy()
        self.cache['data'] = data
        copied_data[copied_data < 0] = 0
        return copied_data

    def backward(self, de_dh: np.ndarray):
        data = self.cache['data']
        de_dh[data < 0] = 0
        return de_dh

    def clear(self):
        self.cache = {}

class BinaryCrossEntropyLoss:
    def __init__(self):
        self.cache = {}

    def sigmoid(self, data: np.ndarray):
        """
        Computes a numerically stable element-wise sigmoid for a NumPy array.
        """
        # Convert input to a numpy array if it isn't one already
        data = np.asarray(data, dtype=float)
        
        return np.where(
            data >= 0,
            1.0 / (1.0 + np.exp(-data)),
            np.exp(data) / (1.0 + np.exp(data))
        )

    def forward(self, data: np.ndarray, y: np.ndarray):
        """
            data: (B, 1)
            y:    (B, 1)
        """
        # Logits to Probabilities
        yhat = self.sigmoid(data) # (B, 1)
        losses = -y * np.log(yhat) - (1 - y) * np.log(1 - yhat) # (B, 1)

        # Cache
        self.cache['data'] = data
        self.cache['yhat'] = yhat
        self.cache['y'] = y

        return yhat, np.mean(losses)

    def backward(self):
        yhat = self.cache['yhat'] # (B, 1)
        y = self.cache['y'] # (B, 1)

        de_dz = yhat - y # (B, 1)
        return de_dz

    def clear(self):
        self.cache = {}


class Network:
    def __init__(self):
        self.layers = [
            LinearLayer(2, 10),
            ReLu(),
            LinearLayer(10, 1),
        ]

        self.loss = BinaryCrossEntropyLoss()

    def forward(self, data: np.ndarray, labels: np.ndarray):
        h = data
        for layer in self.layers:
            h = layer.forward(h)

        yhat, loss = self.loss.forward(h, labels)
        return yhat, loss

    def backward(self):
        grad = self.loss.backward()

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def clear(self):
        for layer in self.layers:
            layer.clear()
        self.loss.clear()

    def train(self, epochs: int, batch_size: int, data: np.ndarray, labels: np.ndarray):
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

if __name__ == "__main__":
    from data_generator import BinaryClassificationDataset

    dataset = BinaryClassificationDataset()

    net = Network()

    yhats, _ = net.forward(dataset.train_data, dataset.train_labels)
    print(f"Train Accuracy: {np.mean((yhats > 0.5).astype(np.int32) == dataset.train_labels)}")

    yhats, _ = net.forward(dataset.test_data, dataset.test_labels)
    print(f"Test Accuracy: {np.mean((yhats > 0.5).astype(np.int32) == dataset.test_labels)}")

    net.train(epochs=100, batch_size=15, data=dataset.train_data, labels=dataset.train_labels)

    yhats, _ = net.forward(dataset.train_data, dataset.train_labels)
    print(f"Train Accuracy: {np.mean((yhats > 0.5).astype(np.int32) == dataset.train_labels)}")

    yhats, _ = net.forward(dataset.test_data, dataset.test_labels)
    print(f"Accuracy: {np.mean((yhats > 0.5).astype(np.int32) == dataset.test_labels)}")

    
    