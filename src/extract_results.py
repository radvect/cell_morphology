import numpy as np
import matplotlib.pyplot as plt

riemann = np.load('/home/pavel/cell_morphology/nov30/riemann_distances.npy', allow_pickle=True)
times = np.load('/home/pavel/cell_morphology/nov30/times.npy', allow_pickle=True)
centr = np.load('/home/pavel/cell_morphology/nov30/centroids.npy', allow_pickle=True)

def get_shift(cell, frame):
    if frame == 0:
        shift = [0, 0]
    else: 
        shift = [centr[cell][frame][0] - centr[cell][frame-1][0], centr[cell][frame][1] - centr[cell][frame-1][1]]
    return shift

def get_abs_velocity(cell, frame):
    if frame == 0:
        v = 0
    else: 
        shift = np.sqrt((centr[cell][frame][0] - centr[cell][frame-1][0])**2 +
                        (centr[cell][frame][1] - centr[cell][frame-1][1])**2)
        dt = times[cell][frame]
        v = shift / dt
    return v

def get_times(cell, frame):
    if frame == 0:
        dt = 0
    else: 
        dt = times[cell][frame]
    print(dt)
    return dt

def get_riemann_dist(cell, frame):
    print(riemann[0][:])
    return riemann[cell][frame]


velocities = []
riemann_distances = []
time_data = []
displacements = []
num_cells = len(centr)

for cell in range(1, 204):
    num_frames = len(centr[cell])
    for frame in range(num_frames):
        riemann_test = get_riemann_dist(cell, frame)
        if(riemann_test<10):
            velocities.append(get_abs_velocity(cell, frame))
            riemann_distances.append(get_riemann_dist(cell, frame))
            time_data.append(get_times(cell, frame))
            if frame > 0:
                displacement = np.sqrt((centr[cell][frame][0] - centr[cell][0][0])**2 +
                                        (centr[cell][frame][1] - centr[cell][0][1])**2)
            else:
                displacement = 0
            displacements.append(displacement)
        else: 
            print(f"cell {cell}, frame {frame}. Riemann Distance {riemann_test} ")
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].scatter(velocities, riemann_distances, alpha=0.5, s=10, label="Riemann vs Velocity")
axs[0, 0].set_title("Riemann Distance vs Velocity")
axs[0, 0].set_xlabel("Velocity")
axs[0, 0].set_ylabel("Riemann Distance")
axs[0, 0].grid(True, linestyle='--', alpha=0.7)

axs[0, 1].scatter(time_data, riemann_distances, alpha=0.5, s=10, label="Riemann vs Time", color='orange')
axs[0, 1].set_title("Riemann Distance vs Time")
axs[0, 1].set_xlabel("Time")
axs[0, 1].set_ylabel("Riemann Distance")
axs[0, 1].grid(True, linestyle='--', alpha=0.7)

axs[1, 0].scatter(time_data, displacements, alpha=0.5, s=10, label="Displacement vs Time", color='green')
axs[1, 0].set_title("Displacement of Center of Mass vs Time")
axs[1, 0].set_xlabel("Time")
axs[1, 0].set_ylabel("Displacement")
axs[1, 0].grid(True, linestyle='--', alpha=0.7)

axs[1, 1].scatter(displacements, riemann_distances, alpha=0.5, s=10, label="Riemann vs Displacement", color='red')
axs[1, 1].set_title("Riemann Distance vs Displacement")
axs[1, 1].set_xlabel("Displacement")
axs[1, 1].set_ylabel("Riemann Distance")
axs[1, 1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()