import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

riemann = np.load('/home/pavel/cell_morphology/nov30/riemann_distances.npy', allow_pickle=True)
times = np.load('/home/pavel/cell_morphology/nov30/times.npy', allow_pickle=True)
centr = np.load('/home/pavel/cell_morphology/nov30/centroids.npy', allow_pickle=True)

for i in range(len(centr)): 
    x_coords = [c[0] for c in centr[i]]
    y_coords = [c[1] for c in centr[i]]
    print(x_coords)
    x_smooth = gaussian_filter1d(x_coords, sigma=2.5)
    y_smooth = gaussian_filter1d(y_coords, sigma=2.5)
    
    smooth_centr = [[x, y] for x, y in zip(x_smooth, y_smooth)]
    centr[i] = smooth_centr


total_plots = len(centr)
plots_per_page = 12  
rows, cols = 4, 3  
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


def get_velocity_angle_rel(cell, frame):
    
    if frame == 0:
        angle_degrees = 0
    else: 

        x2 = centr[cell][frame][0]
        x1 = centr[cell][frame-1][0]
        y2 = centr[cell][frame][1] 
        y1 = centr[cell][frame-1][1]
   
        dx = x2 - x1
        dy = y2 - y1
        

        angle_radians = math.atan2(dy, dx)

        angle_degrees = math.degrees(angle_radians)

        return angle_degrees


def get_times(cell, frame):
    if frame == 0:
        dt = 0
    else: 
        dt = times[cell][frame]

    return dt

def get_riemann_dist(cell, frame):

    return riemann[cell][frame]

velocities = []
riemann_distances = []
time_data = []
displacements = []

def anglerel_riemann_absvel_time_with_smoothing_only(sigma=2):
    with PdfPages("anglerel_riemann_absvel_times_NO_SMOOTHING.pdf") as pdf:
        for page_start in range(0, total_plots, plots_per_page):
            # Создаем страницу с форматом A4
            fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69)) 
            axes = axes.flatten()  # Преобразуем в одномерный массив для удобства работы
            
            # Генерируем графики для текущей страницы
            for i in range(plots_per_page):
    
                velocities = []
                riemann_distances = []
                time_data = []
                time_data1 = []
                displacements = []
                rel_angle = []
                plot_index = page_start + i
                num_frames = len(centr[plot_index])
                if(num_frames<=10):
                    continue
                for frame in range(1, num_frames):
                    time_data.append(get_times(plot_index, frame))
                    rel_angle.append(get_velocity_angle_rel(plot_index, frame))
                for frame in range(1, num_frames):
                    riemann_distances.append(get_riemann_dist(plot_index, frame))
                    time_data1.append(get_times(plot_index, frame))
                    velocities.append(get_abs_velocity(plot_index, frame))

                if plot_index >= total_plots:  
                    axes[i].axis('off')
                    continue
                
                # Применяем сглаживание
                # riemann_distances = gaussian_filter1d(riemann_distances, sigma=sigma)
                # velocities = gaussian_filter1d(velocities, sigma=sigma)
                # rel_angle = gaussian_filter1d(rel_angle, sigma=sigma)

                # Основной график (слева)
                ax = axes[i]
                ax.plot(time_data1, riemann_distances, label="Riemann", color="blue")
                ax.plot(time_data1, velocities, label="Velocity", color="green")
                ax.set_xlabel("Time")
                ax.set_ylabel("Riemann Distance / Velocity", fontsize=8)
                ax.legend(loc="upper left", fontsize=6)
                ax.tick_params(axis="both", which="major", labelsize=6)

                # Вторичная шкала для углов (справа)
                ax_angle = ax.twinx()
                ax_angle.plot(time_data, rel_angle, label="Angle", color="red")
                ax_angle.set_ylabel("Angle (degrees)", fontsize=8, color="red")
                ax_angle.tick_params(axis="y", labelsize=6, colors="red")
                ax_angle.legend(loc="upper right", fontsize=6)

                ax.set_title(f"Cell {plot_index + 1}", fontsize=8)

            # Отключаем лишние оси
            for j in range(len(axes)):
                if j >= plots_per_page:
                    axes[j].axis('off')

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)



from hmmlearn import hmm 

def extremum_points(sigma=2):
    with PdfPages("extremum_points.pdf") as pdf:
        for page_start in range(0, total_plots, plots_per_page):
            # Создаем страницу с форматом A4
            fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69)) 
            axes = axes.flatten()  # Преобразуем в одномерный массив для удобства работы
            
            # Генерируем графики для текущей страницы
            for i in range(plots_per_page):
    
                velocities = []
                riemann_distances = []
                time_data = []
                time_data1 = []
                rel_angle = []
                plot_index = page_start + i
                num_frames = len(centr[plot_index])
                
                for frame in range(1, num_frames):
                    time_data.append(get_times(plot_index, frame))
                    rel_angle.append(get_velocity_angle_rel(plot_index, frame))
                for frame in range(1, num_frames):
                    riemann_distances.append(get_riemann_dist(plot_index, frame))
                    time_data1.append(get_times(plot_index, frame))
                    velocities.append(get_abs_velocity(plot_index, frame))

                if plot_index >= total_plots:  
                    axes[i].axis('off')
                    continue
                
                # Применяем сглаживание
                riemann_distances = gaussian_filter1d(riemann_distances, sigma=sigma)
                velocities = gaussian_filter1d(velocities, sigma=sigma)
                rel_angle = gaussian_filter1d(rel_angle, sigma=sigma)

                riemann_maxima = argrelextrema(riemann_distances, np.greater)[0]  
                riemann_minima = argrelextrema(riemann_distances, np.less)[0]    

                riemann_extremum = list(riemann_maxima) + list(riemann_minima)




                velocities_maxima = argrelextrema(velocities, np.greater)[0]
                velocities_minima = argrelextrema(velocities, np.less)[0]

                velocities_extremum = list(velocities_maxima) + list(velocities_minima)

                angle_maxima = argrelextrema(rel_angle, np.greater)[0]
                angle_minima = argrelextrema(rel_angle, np.less)[0]

                angle_extremum = list(angle_maxima) + list(angle_minima)
                
                
                # Основной график
                ax = axes[i]
                ax.plot(time_data1, riemann_distances, label="Riemann", color="blue")
                ax.plot(time_data1, velocities, label="Velocity", color="green")
                ax.set_xlabel("Time")
                ax.set_ylabel("Riemann Distance / Velocity", fontsize=8)

                # Отображение точек экстремумов на оси x
                ax.scatter(np.array(time_data1)[riemann_maxima], [0]*len(riemann_maxima), color="red", label="Riemann Maxima", marker="^")
                ax.scatter(np.array(time_data1)[riemann_minima], [0]*len(riemann_minima), color="blue", label="Riemann Minima", marker="v")
                ax.scatter(np.array(time_data1)[velocities_maxima], [0]*len(velocities_maxima), color="orange", label="Velocity Maxima", marker="^")
                ax.scatter(np.array(time_data1)[velocities_minima], [0]*len(velocities_minima), color="green", label="Velocity Minima", marker="v")
                ax.scatter(np.array(time_data)[angle_maxima], [0]*len(angle_maxima), color="purple", label="Angle Maxima", marker="^")
                ax.scatter(np.array(time_data)[angle_minima], [0]*len(angle_minima), color="pink", label="Angle Minima", marker="v")

                ax_angle = ax.twinx()
                ax_angle.plot(time_data, rel_angle, label="Angle", color="red")
                ax_angle.set_ylabel("Angle (degrees)", fontsize=8, color="red")
                ax_angle.tick_params(axis="y", labelsize=6, colors="red")
                ax_angle.legend(loc="upper right", fontsize=6)

                ax.legend(loc="upper left", fontsize=6)
                ax.set_title(f"Cell {plot_index + 1}", fontsize=8)
                ax.tick_params(axis="both", which="major", labelsize=6)

            # Отключаем лишние оси
            for j in range(len(axes)):
                if j >= plots_per_page:
                    axes[j].axis('off')

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)





def extremum_points_markov(sigma=2):
    with PdfPages("extremum_points.pdf") as pdf:
        for page_start in range(0, total_plots, plots_per_page):
            fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69)) 
            axes = axes.flatten()  
            

            for i in range(plots_per_page):
    
                velocities = []
                riemann_distances = []
                time_data = []
                time_data1 = []
                rel_angle = []
                plot_index = page_start + i
                num_frames = len(centr[plot_index])
                
                for frame in range(1, num_frames):
                    time_data.append(get_times(plot_index, frame))
                    rel_angle.append(get_velocity_angle_rel(plot_index, frame))
                for frame in range(1, num_frames):
                    riemann_distances.append(get_riemann_dist(plot_index, frame))
                    time_data1.append(get_times(plot_index, frame))
                    velocities.append(get_abs_velocity(plot_index, frame))

                if plot_index >= total_plots:  
                    axes[i].axis('off')
                    continue
                
                riemann_distances = gaussian_filter1d(riemann_distances, sigma=sigma)
                velocities = gaussian_filter1d(velocities, sigma=sigma)
                rel_angle = gaussian_filter1d(rel_angle, sigma=sigma)


                riemann_maxima = argrelextrema(riemann_distances, np.greater)[0]  
                riemann_minima = argrelextrema(riemann_distances, np.less)[0]    

                riemann_extremum = list(riemann_maxima) + list(riemann_minima)




                velocities_maxima = argrelextrema(velocities, np.greater)[0]
                velocities_minima = argrelextrema(velocities, np.less)[0]

                velocities_extremum = list(velocities_maxima) + list(velocities_minima)

                angle_maxima = argrelextrema(rel_angle, np.greater)[0]
                angle_minima = argrelextrema(rel_angle, np.less)[0]

                angle_extremum = list(angle_maxima) + list(angle_minima)
                
                ax = axes[i]
                ax.plot(time_data1, riemann_distances, label="Riemann", color="blue")
                ax.plot(time_data1, velocities, label="Velocity", color="green")
                ax.set_xlabel("Time")
                ax.set_ylabel("Riemann Distance / Velocity", fontsize=8)

                ax.scatter(np.array(time_data1)[riemann_maxima], [0]*len(riemann_maxima), color="red", label="Riemann Maxima", marker="^")
                ax.scatter(np.array(time_data1)[riemann_minima], [0]*len(riemann_minima), color="blue", label="Riemann Minima", marker="v")
                ax.scatter(np.array(time_data1)[velocities_maxima], [0]*len(velocities_maxima), color="orange", label="Velocity Maxima", marker="^")
                ax.scatter(np.array(time_data1)[velocities_minima], [0]*len(velocities_minima), color="green", label="Velocity Minima", marker="v")
                ax.scatter(np.array(time_data)[angle_maxima], [0]*len(angle_maxima), color="purple", label="Angle Maxima", marker="^")
                ax.scatter(np.array(time_data)[angle_minima], [0]*len(angle_minima), color="pink", label="Angle Minima", marker="v")

                ax.legend(loc="upper left", fontsize=6)
                ax.set_title(f"Cell {plot_index + 1}", fontsize=8)
                ax.tick_params(axis="both", which="major", labelsize=6)

            for j in range(len(axes)):
                if j >= plots_per_page:
                    axes[j].axis('off')

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

#extremum_points_markov()

# extremum_points()


anglerel_riemann_absvel_time_with_smoothing_only()