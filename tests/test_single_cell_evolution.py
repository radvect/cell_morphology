import numpy as np
import matplotlib.pyplot as plt
import os


def cell_evolution(number_of_cell):
    fig, ax = plt.subplots(figsize=(10, 10), layout='constrained')

    number_of_frames = sum(os.path.isdir(os.path.join(f"cells/cell_{number_of_cell}", entry)) for entry in os.listdir(f"cells/cell_{number_of_cell}"))
    colors = plt.cm.tab20(np.linspace(0, 1, number_of_frames))
    for i in range(1,number_of_frames+1):
        time = np.load(f'cells/cell_{number_of_cell}/frame_{i}/time.npy')
        border = np.load(f'cells/cell_{number_of_cell}/frame_{i}/outline.npy')
        centroid = np.load(f'cells/cell_{number_of_cell}/frame_{i}/centroid.npy')

        
        color = colors[i - 1]

        ax.plot(border[:, 0], border[:, 1], label=time, color=color)
        ax.scatter(centroid[0], centroid[1], color=color)
    plt.legend()    
    plt.savefig(f"pic/single_cell_{number_of_cell}.png", dpi=300, bbox_inches='tight')

cell_evolution(15)