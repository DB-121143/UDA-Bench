import numpy as np

def calc_center(points):
    """
    a list of embedding of np.ndarray, calc the geo center
    """
    return np.mean(points, axis=0)
