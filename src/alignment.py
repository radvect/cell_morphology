import numpy as np
from geomstats.geometry.pre_shape import PreShapeSpace
import matplotlib.pyplot as plt

import geomstats.backend as gs
import src.projection as proj


def exhaustive_align(curve, base_curve):

    AMBIENT_DIM = 2
    PRESHAPE_SPACE = PreShapeSpace(ambient_dim=AMBIENT_DIM, k_landmarks=len(curve))

    PRESHAPE_SPACE.equip_with_group_action("rotations")
    PRESHAPE_SPACE.equip_with_quotient()
    curve_projected  = proj.project_on_kendell_space(curve,PRESHAPE_SPACE)



    nb_sampling = len(curve_projected)
    distances = gs.zeros(nb_sampling)
    base_curve = gs.array(base_curve)
    for shift in range(nb_sampling):
        reparametrized = [curve_projected[(i + shift) % nb_sampling] for i in range(nb_sampling)]
        aligned = PRESHAPE_SPACE.fiber_bundle.align(
            point=gs.array(reparametrized), base_point=base_curve
        )
        distances[shift] = PRESHAPE_SPACE.embedding_space.metric.norm(
            gs.array(aligned) - gs.array(base_curve)
        )
    shift_min = gs.argmin(distances)

    reparametrized_min = [
        curve_projected[(i + shift_min) % nb_sampling] for i in range(nb_sampling)
    ]
    aligned_curve = PRESHAPE_SPACE.fiber_bundle.align(
        point=gs.array(reparametrized_min), base_point=base_curve
    )
    return aligned_curve