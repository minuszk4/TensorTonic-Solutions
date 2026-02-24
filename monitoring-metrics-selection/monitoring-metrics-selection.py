import numpy as np

def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    Must return a list of (metric_name, value) tuples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if system_type == 'classification':
        n = len(y_true)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = (tp + tn) / n if n > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        return [
            ('accuracy', accuracy),
            ('f1', f1_score),
            ('precision', precision),
            ('recall', recall),
        ]

    elif system_type == 'regression':
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)

        return [
            ('mae', mae),
            ('rmse', rmse),
        ]

    else:  # ranking
        n = len(y_true)

        sorted_indices = np.argsort(-y_pred)
        sorted_true = y_true[sorted_indices]

        precision_at_3 = np.sum(sorted_true[:3]) / 3 if n >= 3 else 0.0
        recall_at_3 = (
            np.sum(sorted_true[:3]) / np.sum(y_true)
            if np.sum(y_true) > 0 else 0.0
        )

        return [
            ('precision_at_3', precision_at_3),
            ('recall_at_3', recall_at_3),
        ]