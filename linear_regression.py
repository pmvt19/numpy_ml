import numpy as np

class LinearRegression:
    def __init__(self):
        self.m = None
        self.b = None

    def _get_padded_data(self, data: np.ndarray, labels: np.ndarray):
        X = np.hstack((data.reshape(-1, 1), np.ones_like(data.reshape(-1, 1))))
        y = labels
        return X, y

    def _compute_l2_loss(self, w: np.ndarray, data: np.ndarray, labels: np.ndarray):
        pass

    def _fit_gradient_descent(self, data: np.ndarray, labels: np.ndarray):
        X, y = self._get_padded_data(data, labels)
        N, d = X.shape

        batch_size = 100
        epochs = 100000
        lr = 1e-5

        w = np.random.normal(size=(2,))
        for i in range(epochs):
            # Select Data Points and Corresponding Labels
            idx = np.random.choice(N, size=(batch_size,), replace=False)
            x = X[idx]
            gt_y = y[idx]

            # Forward Pass
            y_hat  = (w.reshape(1, -1) @ x.T).T

            # Gradient Calculation
            dl_dw = (y_hat - gt_y.reshape(-1, 1)) * x
            dl_dw = np.mean(dl_dw, axis=0)

            # Weight Update
            w = w - lr * dl_dw

        self.m, self.b = w

    def _fit_closed_form(self, data: np.ndarray, labels: np.ndarray):
        X, y = self._get_padded_data(data, labels)

        self.m, self.b = np.linalg.inv(X.T @ X) @ X.T @ y

    def fit(self, data: np.ndarray, labels: np.ndarray, method='sgd'):
        if method == "sgd":
            self._fit_gradient_descent(data, labels)
        elif method == "closed_form":
            self._fit_closed_form(data, labels)
        else:
            raise NotImplementedError

    def predict(self, data):
        return self.m * data + self.b

if __name__ == "__main__":
    from data_generator import LinearDataSet

    dataset = LinearDataSet()

    regression = LinearRegression()
    regression.fit(dataset.train_data, dataset.train_labels_noisy, method='sgd')
    print(regression.m, regression.b)
