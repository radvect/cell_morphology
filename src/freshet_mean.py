from geomstats.geometry.discrete_curves import ElasticMetric
from geomstats.geometry.discrete_curves import DiscreteCurvesStartingAtOrigin
import os
import numpy as np
from src.interpolation import interpolate, preprocess
from src.alignment import exhaustive_align
from src.projection import project_on_kendell_space
import matplotlib.pyplot as plt
import geomstats.backend as gs

BASE_LINE = np.load(f'cells/cell_1/frame_1/outline.npy')
BASE_LINE= interpolate(BASE_LINE,1000)
BASE_LINE = preprocess(BASE_LINE)
BASE_LINE= project_on_kendell_space(BASE_LINE)

cell_shapes_list = []
for cell_i in range(1,4):
    number_of_frames = sum(os.path.isdir(os.path.join(f"cells/cell_{cell_i}", entry)) for entry in os.listdir(f"cells/cell_{cell_i}"))
    for line in range(number_of_frames):
        border_cell = np.load(f'cells/cell_{cell_i}/frame_{line+1}/outline.npy')
        cell_interpolation= interpolate(border_cell,1000)
        cell_preprocess = preprocess(cell_interpolation)
        border_cell = cell_preprocess
        border_cell = project_on_kendell_space(border_cell)
        
        cell_shapes_list.append(exhaustive_align(border_cell,BASE_LINE))
    print(cell_i)
cell_shapes = gs.array(cell_shapes_list)
print(cell_shapes.shape)


from geomstats.learning.frechet_mean import FrechetMean
a = 3
b = 1
CURVES_SPACE_ELASTIC = DiscreteCurvesStartingAtOrigin(
    ambient_dim=2, k_sampling_points=1000, equip=False
)
CURVES_SPACE_ELASTIC.equip_with_metric(ElasticMetric, a=a, b=b)

mean = FrechetMean(CURVES_SPACE_ELASTIC)

cell_shapes_at_origin = CURVES_SPACE_ELASTIC.projection(cell_shapes)
mean.fit(cell_shapes_at_origin[:])

mean_estimate = mean.estimate_

plt.plot(mean_estimate[:, 0], mean_estimate[:, 1], "black")
#plt.show()

print(gs.sum(gs.isnan(mean_estimate)))
mean_estimate_clean = mean_estimate[~gs.isnan(gs.sum(mean_estimate, axis=1)), :]
print(mean_estimate_clean.shape)
mean_estimate_clean = interpolate(mean_estimate_clean, 1000 - 1)
print(gs.sum(gs.isnan(mean_estimate_clean)))
print(mean_estimate_clean.shape)

plt.plot(mean_estimate_clean[:, 0], mean_estimate_clean[:, 1], "black")
plt.show()