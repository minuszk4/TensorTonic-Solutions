import numpy as np

def f1_micro(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes must match")

    correct = np.sum(y_true == y_pred)
    total = len(y_true)

    return correct / total if total > 0 else 0.0