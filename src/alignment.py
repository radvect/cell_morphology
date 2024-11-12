import numpy as np
from geomstats.geometry.pre_shape import PreShapeSpace
import matplotlib.pyplot as plt

import geomstats.backend as gs

# def align_barycenter(cell, centroid_x, centroid_y, flip):
#     """ 
#     Align the the barycenter of the cell to ref centeriod and flip the cell against the xaxis of the centriod if flip is True. 

#     :param 2D np array cell: cell to align
#     :param float centroid_x: the x coordinates of the projected BASE_CURVE
#     :param float centroid_y: the y coordinates of the projected BASE_CURVE
#     :param bool flip: flip the cell against x = centroid x if True 
#     """
    
#     cell_bc = np.mean(cell, axis=0)
#     aligned_cell = cell+[centroid_x, centroid_y]-cell_bc

#     if flip:
#         aligned_cell[:, 0] = 2*centroid_x-aligned_cell[:, 0]
#         # Flip the order of the points
#         med_index = int(np.floor(aligned_cell.shape[0]/2))
#         flipped_aligned_cell = np.concatenate((aligned_cell[med_index:], aligned_cell[:med_index]), axis=0)
#         flipped_aligned_cell = np.flipud(flipped_aligned_cell)
#         aligned_cell = flipped_aligned_cell
#     return aligned_cell

AMBIENT_DIM = 2
k_sampling_points = 1000
PRESHAPE_SPACE = PreShapeSpace(ambient_dim=AMBIENT_DIM, k_landmarks=k_sampling_points)

PRESHAPE_SPACE.equip_with_group_action("rotations")
PRESHAPE_SPACE.equip_with_quotient()


def exhaustive_align(curve, base_curve):
    """Align curve to base_curve to minimize the L² distance.

    Returns
    -------
    aligned_curve : discrete curve
    """
    nb_sampling = len(curve)
    distances = gs.zeros(nb_sampling)
    base_curve = gs.array(base_curve)
    for shift in range(nb_sampling):
        reparametrized = [curve[(i + shift) % nb_sampling] for i in range(nb_sampling)]
        aligned = PRESHAPE_SPACE.fiber_bundle.align(
            point=gs.array(reparametrized), base_point=base_curve
        )
        distances[shift] = PRESHAPE_SPACE.embedding_space.metric.norm(
            gs.array(aligned) - gs.array(base_curve)
        )
    shift_min = gs.argmin(distances)

    reparametrized_min = [
        curve[(i + shift_min) % nb_sampling] for i in range(nb_sampling)
    ]
    aligned_curve = PRESHAPE_SPACE.fiber_bundle.align(
        point=gs.array(reparametrized_min), base_point=base_curve
    )
    return aligned_curve