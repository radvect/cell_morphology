from src.interpolation import interpolate
import numpy as np
import matplotlib.pyplot as plt


def plot_interpolate(number_of_cell, number_of_frame,k_sampling_points):
    border = np.load(f'cells/cell_{number_of_cell}/frame_{number_of_frame}/outline.npy')
    cell_interpolation= interpolate(border, k_sampling_points)
    fig = plt.figure(figsize=(15, 5))

    fig.add_subplot(121)
    plt.plot(border[:, 0], border[:, 1])
    plt.axis("equal")
    plt.title(f"Original curve ({len(border)} points)")
    plt.axis("off")

    fig.add_subplot(122)
    plt.plot(cell_interpolation[:, 0], cell_interpolation[:, 1])
    plt.axis("equal")
    plt.title(f"Interpolated curve ({k_sampling_points} points)")
    plt.axis("off")

    plt.savefig("pic/interpolation.png")


plot_interpolate(15, 3,1000)