import numpy as np


def shift(l1, l2):
    return np.sqrt((l1[0]-l2[0])**2 + (l1[1]-l2[1])**2 )
