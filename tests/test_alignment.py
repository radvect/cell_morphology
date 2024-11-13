from src.interpolation import interpolate, preprocess
from src.alignment import exhaustive_align
from src.projection import project_on_kendell_space
import numpy as np
import matplotlib.pyplot as plt
import os



def plot_align(number_of_cell,ref_frame, frame_to_align, k_sampling_points):

    border_ref = np.load(f'cells/cell_{number_of_cell}/frame_{ref_frame}/outline.npy')
    cell_interpolation_ref= interpolate(border_ref, k_sampling_points)
    cell_preprocess_ref = preprocess(cell_interpolation_ref)
    border_ref = cell_preprocess_ref

    border_cell = np.load(f'cells/cell_{number_of_cell}/frame_{frame_to_align}/outline.npy')
    cell_interpolation= interpolate(border_cell, k_sampling_points)
    cell_preprocess = preprocess(cell_interpolation)
    border_cell = cell_preprocess
 
    border_align = exhaustive_align(border_cell, border_ref)



    fig = plt.figure(figsize=(15, 5))
    border_ref = project_on_kendell_space(border_ref)
    fig.add_subplot(131)
    plt.plot(border_ref[:, 0], border_ref[:, 1])
    plt.plot(border_ref[0, 0], border_ref[0, 1], "ro")
    plt.axis("equal")
    plt.title(f"Reference")
    
    fig.add_subplot(132)
    border_cell = project_on_kendell_space(border_cell)
    plt.plot(border_cell[:, 0], border_cell[:, 1])
    plt.plot(border_cell[0, 0], border_cell[0, 1], "ro")
    plt.axis("equal")
    plt.title(f"Initial curve")
    
    fig.add_subplot(133)
    border_align = project_on_kendell_space(border_align)
    plt.plot(border_align[:, 0], border_align[:, 1])
    plt.plot(border_align[0, 0], border_align[0, 1], "ro")
    plt.axis("equal")
    plt.title(f"Aligned curve")
    
    plt.show()
    plt.savefig("pic/alignment.png")


plot_align(1,1,7, 1000)