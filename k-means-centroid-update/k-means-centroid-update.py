import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    points = np.asarray(points, dtype=float)
    centroids = np.zeros((k, points.shape[1]))
    for i in range(k):
        assigned_points = points[np.array(assignments) == i]
        if len(assigned_points) > 0:
            centroids[i] = np.mean(assigned_points, axis=0)
    return centroids.tolist()