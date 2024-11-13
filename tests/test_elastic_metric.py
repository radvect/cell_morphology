from geomstats.geometry.discrete_curves import ElasticMetric
from geomstats.geometry.discrete_curves import DiscreteCurvesStartingAtOrigin
import os
import numpy as np
from src.interpolation import interpolate, preprocess
from src.alignment import exhaustive_align
from src.projection import project_on_kendell_space
import matplotlib.pyplot as plt
import geomstats.backend as gs


def test_geodesic_func_elastic(cell_number,k_sampling_points,cell_frame_start=None, cell_frame_finish=None):
    a = 3
    b = 1

    if(cell_frame_start==None):
        initial_frame_border = np.load(f'cells/cell_{cell_number}/frame_{1}/outline.npy')
    else:
        initial_frame_border = np.load(f'cells/cell_{cell_number}/frame_{cell_frame_start}/outline.npy')
    interpolated_initial_frame_border= interpolate(initial_frame_border, k_sampling_points)
    preprocessed_initial_frame_border = preprocess(interpolated_initial_frame_border)
    aligned_initial_frame_border =  preprocessed_initial_frame_border # Doing alignment relatively to the first image

    if(cell_frame_finish==None):
        number_of_frames = sum(os.path.isdir(os.path.join(f"cells/cell_{cell_number}", entry)) for entry in os.listdir(f"cells/cell_{cell_number}"))
        final_frame_border = np.load(f'cells/cell_{cell_number}/frame_{number_of_frames}/outline.npy')
    else:
        final_frame_border = np.load(f'cells/cell_{cell_number}/frame_{cell_frame_finish}/outline.npy')  

    interpolated_final_frame_border= interpolate(final_frame_border, k_sampling_points)
    preprocessed_final_frame_border = preprocess(interpolated_final_frame_border)
        
    aligned_final_frame_border = exhaustive_align(preprocessed_final_frame_border,aligned_initial_frame_border)



    CURVES_SPACE_ELASTIC = DiscreteCurvesStartingAtOrigin(
        ambient_dim=2, k_sampling_points=k_sampling_points, equip=False
    )
    CURVES_SPACE_ELASTIC.equip_with_metric(ElasticMetric, a=a, b=b)


    cell_start_at_origin = CURVES_SPACE_ELASTIC.projection(aligned_initial_frame_border)
    cell_end_at_origin = CURVES_SPACE_ELASTIC.projection(aligned_final_frame_border)

    geodesic_func = CURVES_SPACE_ELASTIC.metric.geodesic(
        initial_point=cell_start_at_origin, end_point=cell_end_at_origin
    )


    n_times =30
    times = gs.linspace(0.0, 1.0, n_times)
    geod_points = geodesic_func(times)
    fig = plt.figure(figsize=(10, 2))
    plt.title("Geodesic between two cells")
    plt.axis("off")

    for i, curve in enumerate(geod_points):
        fig.add_subplot(2, n_times // 2, i + 1)
        plt.plot(curve[:, 0], curve[:, 1])
        plt.axis("equal")
        plt.axis("off")

    plt.savefig("pic/geodesic_dist.png")

test_geodesic_func_elastic(150,1000)