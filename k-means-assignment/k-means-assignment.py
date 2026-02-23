import numpy as np

def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    points = np.asarray(points, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    assignments = []

    for point in points:
        distances = np.linalg.norm(centroids - point, axis=1)
        closest_centroid_index = np.argmin(distances) 
        assignments.append(int(closest_centroid_index))

    return assignments