from geomstats.geometry.discrete_curves import ElasticMetric
from geomstats.geometry.discrete_curves import DiscreteCurvesStartingAtOrigin
import os
import numpy as np
from src.interpolation import interpolate, preprocess
from src.alignment import exhaustive_align
from src.projection import project_on_kendell_space
import matplotlib.pyplot as plt
import geomstats.backend as gs




riemann_distances = []
times = []
centroids = []
a = 3
b = 1

CURVES_SPACE_ELASTIC = DiscreteCurvesStartingAtOrigin(
    ambient_dim=2, k_sampling_points=1000, equip=False
)
CURVES_SPACE_ELASTIC.equip_with_metric(ElasticMetric, a=a, b=b)

def calculate_distance(border,reference_shape):

    return CURVES_SPACE_ELASTIC.metric.dist(CURVES_SPACE_ELASTIC.projection(border), reference_shape)


for cell_i in range(1,205):
    number_of_frames = sum(os.path.isdir(os.path.join(f"cells/cell_{cell_i}", entry)) for entry in os.listdir(f"cells/cell_{cell_i}"))  

    iter_distance = np.zeros(number_of_frames)
    iter_time = np.zeros(number_of_frames)
    iter_centroid = np.array([np.random.rand(2) for _ in range(number_of_frames)])
    BASE_LINE = np.load(f'cells/cell_{cell_i}/frame_1/outline.npy')
    BASE_LINE= interpolate(BASE_LINE,1000)
    BASE_LINE = preprocess(BASE_LINE)
    BASE_LINE= project_on_kendell_space(BASE_LINE)
    BASE_LINE_projected = (CURVES_SPACE_ELASTIC.projection(BASE_LINE))
    for i in range(number_of_frames):
        border_cell = np.load(f'cells/cell_{cell_i}/frame_{i+1}/outline.npy')
        cell_interpolation= interpolate(border_cell,1000)
        cell_preprocess = preprocess(cell_interpolation)
        border_cell = cell_preprocess
        border_cell = project_on_kendell_space(border_cell)
        aligned_border = exhaustive_align(border_cell,BASE_LINE)
        iter_distance[i] = calculate_distance(aligned_border, BASE_LINE_projected)

        iter_time[i] = np.load(f'cells/cell_{cell_i}/frame_{i+1}/time.npy')
        iter_centroid[i] = np.load(f'cells/cell_{cell_i}/frame_{i+1}/centroid.npy')
    print(iter_distance)
    print(iter_time)
    print(iter_centroid)
    riemann_distances.append(iter_distance)
    times.append(iter_time)
    centroids.append(iter_centroid)
    

with open('results/riemann_distances.npy', 'wb') as f:
    np.save(f, np.array(riemann_distances, dtype=object))
with open('results/times.npy', 'wb') as f:
    np.save(f, np.array(times, dtype=object))
with open('results/centrouds.npy', 'wb') as f:
    np.save(f, np.array(centroids, dtype=object))