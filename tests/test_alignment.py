from src.interpolation import interpolate, preprocess
from src.alignment import exhaustive_align
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_align(number_of_cell,k_sampling_points):
    border = np.load(f'cells/cell_{number_of_cell}/frame_{1}/outline.npy')
    cell_interpolation_ref= interpolate(border, k_sampling_points)
    cell_preprocess_ref = preprocess(cell_interpolation_ref)
    border_ref = cell_preprocess_ref

    border_cell = np.load(f'cells/cell_{number_of_cell}/frame_{7}/outline.npy')
    cell_interpolation= interpolate(border_cell, k_sampling_points)
    cell_preprocess = preprocess(cell_interpolation)
    border = cell_preprocess
 
    border_align = exhaustive_align(border, border_ref)

    fig = plt.figure(figsize=(15, 5))

    fig.add_subplot(131)
    plt.plot(border_ref[:, 0], border_ref[:, 1])
    plt.axis("equal")
    plt.title(f"Reference")
    plt.axis("off")

    fig.add_subplot(132)
    plt.plot(border_cell[:, 0], border_cell[:, 1])
    plt.axis("equal")
    plt.title(f"Initial curve")
    plt.axis("off")



    fig.add_subplot(133)
    plt.plot(border_align[:, 0], border_align[:, 1])

    plt.axis("equal")
    plt.title(f"Aligned curve")
    plt.axis("off")
    #plt.show()
    plt.savefig("pic/alignment.png")


plot_align(1,1000)