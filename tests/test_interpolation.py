from src.interpolation import interpolate, preprocess
import numpy as np
import matplotlib.pyplot as plt
import projection as proj

def plot_interpolate(number_of_cell, number_of_frame,k_sampling_points):
    border = np.load(f'cells/cell_{number_of_cell}/frame_{number_of_frame}/outline.npy')
    cell_interpolation= interpolate(border, k_sampling_points)
    cell_preprocess = preprocess(cell_interpolation)
    fig = plt.figure(figsize=(15, 5))

    fig.add_subplot(131)
    plt.plot(border[:, 0], border[:, 1])
    plt.axis("equal")
    plt.title(f"Original curve ({len(border)} points)")
    plt.axis("off")

    fig.add_subplot(132)
    plt.plot(cell_interpolation[:, 0], cell_interpolation[:, 1])
    plt.axis("equal")
    plt.title(f"Interpolated curve ({k_sampling_points} points)")
    plt.axis("off")



    fig.add_subplot(133)
    plt.plot(cell_preprocess[:, 0], cell_preprocess[:, 1])
    if(len(cell_preprocess)!=1000):
        print(number_of_cell)
    plt.axis("equal")
    plt.title(f"Preprocessed curve ({len(cell_preprocess)} points)")
    plt.axis("off")
    #plt.show()
    plt.savefig("pic/interpolation.png")

