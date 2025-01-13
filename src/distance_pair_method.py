import os
from geomstats.geometry.discrete_curves import ElasticMetric
from geomstats.geometry.discrete_curves import DiscreteCurvesStartingAtOrigin
import os
import numpy as np
from src.interpolation import interpolate, preprocess
from src.alignment import align
from src.projection import project_on_kendall_space
#import matplotlib.pyplot as plt
import geomstats.backend as gs



riemann_distances = []
times = []
centroids = []
a = 1
b = 1/2

CURVES_SPACE_ELASTIC = DiscreteCurvesStartingAtOrigin(
    ambient_dim=2, k_sampling_points=1000, equip=False
)
CURVES_SPACE_ELASTIC.equip_with_metric(ElasticMetric, a=a, b=b)

def calculate_distance(border,reference_shape):

    return CURVES_SPACE_ELASTIC.metric.dist(CURVES_SPACE_ELASTIC.projection(border), CURVES_SPACE_ELASTIC.projection(reference_shape))


for cell_i in range(1,2):
    number_of_frames = sum(os.path.isdir(os.path.join(f"cells/cell_{cell_i}", entry)) for entry in os.listdir(f"cells/cell_{cell_i}"))  

    iter_distance = np.zeros(number_of_frames)
    iter_time = np.zeros(number_of_frames)
    iter_centroid = np.array([np.random.rand(2) for _ in range(number_of_frames)])
    BASE_LINE = np.load(f'cells/cell_{cell_i}/frame_1/outline.npy')
    BASE_LINE= interpolate(BASE_LINE,1000)
    BASE_LINE = preprocess(BASE_LINE)
    #BASE_LINE= project_on_kendall_space(BASE_LINE)
    for i in range(number_of_frames):
        border_cell = np.load(f'cells/cell_{cell_i}/frame_{i+1}/outline.npy')
        cell_interpolation= interpolate(border_cell,1000)
        cell_preprocess = preprocess(cell_interpolation)
        border_cell = cell_preprocess
        border_cell = project_on_kendall_space(cell_interpolation)
        aligned_border = align(border_cell, BASE_LINE, rescale=True, rotation=False, reparameterization=True, k_sampling_points=1000)
        iter_distance[i] = calculate_distance(aligned_border, BASE_LINE)
        print(iter_distance[i])
        iter_time[i] = np.load(f'cells/cell_{cell_i}/frame_{i+1}/time.npy')
        iter_centroid[i] = np.load(f'cells/cell_{cell_i}/frame_{i+1}/centroid.npy')
        BASE_LINE = aligned_border #border_cell

    #print(iter_distance)
    #print(iter_time)
    #print(iter_centroid)
    riemann_distances.append(iter_distance)
    times.append(iter_time)
    centroids.append(iter_centroid)
    
# file_path = os.path.join(temp_dir, 'morph', 'riemann_distances.npy')

# with open("/scratch/st-am823-1/Pavel/riemann_distances.npy", 'wb') as f:
#     np.save(f, np.array(riemann_distances, dtype=object))
# #file_path = os.path.join("/scratch/st-am823-1/Pavel/riemann_distances.npy", 'morph', 'times.npy')

# with open("/scratch/st-am823-1/Pavel/times.npy", 'wb') as f:
#     np.save(f, np.array(times, dtype=object))
# #file_path = os.path.join("/scratch/st-am823-1/Pavel/centroids.npy", 'morph', 'centroids.npy')

# with open("/scratch/st-am823-1/Pavel/centroids.npy", 'wb') as f:
#     np.save(f, np.array(centroids, dtype=object))
