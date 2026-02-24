import numpy as np

def detect_drift(reference_counts, production_counts, threshold):


    reference_counts = np.asarray(reference_counts, dtype=float)
    production_counts = np.asarray(production_counts, dtype=float)

    total_reference = np.sum(reference_counts)
    total_production = np.sum(production_counts)

    if total_reference == 0 or total_production == 0:
        return False

    p_ref = reference_counts / total_reference
    p_prod = production_counts / total_production

    tvd = 0.5 * np.sum(np.abs(p_ref - p_prod))

    return {"score": float(tvd), "drift_detected": True if tvd > threshold else False}