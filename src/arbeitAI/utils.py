import numpy as np
from sklearn.preprocessing import Normalizer

def to_model(sp: np.ndarray):
    if len(sp.shape) == 1:
        sp = [sp]
    return Normalizer().fit(sp).transform(sp)
