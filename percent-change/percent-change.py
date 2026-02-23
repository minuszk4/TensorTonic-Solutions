def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """
    # Write code here
    ans = []
    for i in range(1, len(series)):
        if series[i - 1] != 0:
            change = (series[i] - series[i - 1]) / series[i - 1]
        else:
            change = 0.0
        ans.append(change)
    return ans
