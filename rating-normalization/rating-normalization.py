def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    # Write code here
    import numpy as np
    for i in range(len(matrix)):
        row = matrix[i]
        non_zero_ratings = [rating for rating in row if rating != 0]
        if non_zero_ratings:
            mean_rating = sum(non_zero_ratings) / len(non_zero_ratings)
            for j in range(len(row)):
                if row[j] != 0:
                    matrix[i][j] -= mean_rating
    return matrix