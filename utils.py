import numpy as np

def pairwise_dist(points1: np.ndarray, points2: np.ndarray, method='sqeuclidean'):
    dist_mat = np.sum(points1**2, axis=1, keepdims=True) + np.sum(points2**2, axis=1, keepdims=True).T + (-2 * points1@points2.T)
    if method == 'sqeuclidean':
        return dist_mat
    elif method == 'euclidean':
        return np.sqrt(dist_mat)
    else:
        raise NotImplementedError