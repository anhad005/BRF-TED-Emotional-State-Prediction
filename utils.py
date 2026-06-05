import numpy as np

def shannon_entropy(probs):
    probs = np.asarray(probs)
    probs = probs[probs > 0]

    return float(
        -np.sum(probs * np.log2(probs))
    )