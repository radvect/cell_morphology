import numpy as np
import matplotlib.pyplot as plt
import os
import geomstats.backend as gs
fig, ax = plt.subplots(figsize=(10, 10), layout='constrained')

def all_cells_plotting(frame_number):
    number_of_cells =  sum(os.path.isdir(os.path.join(f"cells/", entry)) for entry in os.listdir(f"cells/"))
    colors = plt.cm.tab20(np.linspace(0, 1, number_of_cells))
    for i in range(1,number_of_cells+1):
        if not os.path.exists(f'cells/cell_{i}/frame_{frame_number}/outline.npy'):
            continue

        border = np.load(f'cells/cell_{i}/frame_{frame_number}/outline.npy')
        
    
        color = colors[i - 1]

        ax.plot(border[:, 0], border[:, 1], color=color)
    plt.legend()    

    plt.savefig(f"pic/all_cells_plotting_{frame_number}.png", dpi=300, bbox_inches='tight')
all_cells_plotting(10)