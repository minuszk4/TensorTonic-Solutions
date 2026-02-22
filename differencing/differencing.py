def differencing(series, order):
    """
    Apply d-th order differencing to the time series.
    """
    # Write code here
    differenced = series.copy()
    for _ in range(order):
        differenced = [differenced[i] - differenced[i - 1] for i in range(1, len(differenced))]
    return differenced