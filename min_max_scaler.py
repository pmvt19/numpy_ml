import numpy as np

class MinMaxScaler:
    def __init__(self):
        self.mins = None
        self.maxs = None

    def fit(self, data: np.ndarray):
        """
            data: (N, d)
        """
        # Shift Data to have a min value of 0
        self.mins = np.min(data, axis=0, keepdims=True)
        shifted_data = data - self.mins

        # Scale Dat to have a max value of 1
        self.maxs = np.max(shifted_data, axis=0, keepdims=True)
        scaled_data = shifted_data / self.maxs

        return scaled_data

    def predict(self, data: np.ndarray):
        return (data - self.mins) / self.maxs

    def inverse(self, y_data: np.ndarray):
        return (y_data * self.maxs + self.mins)

if __name__ == "__main__":
    from data_generator import GridDataSet

    dataset = GridDataSet()

    scaler = MinMaxScaler()
    scaler.fit(dataset.train_data)
