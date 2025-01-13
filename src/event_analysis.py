import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

riemann = np.load('/home/pavel/cell_morphology/nov30/riemann_distances.npy', allow_pickle=True)
times = np.load('/home/pavel/cell_morphology/nov30/times.npy', allow_pickle=True)
centr = np.load('/home/pavel/cell_morphology/nov30/centroids.npy', allow_pickle=True)

# for i in range(len(centr)): 
#     x_coords = [c[0] for c in centr[i]]
#     y_coords = [c[1] for c in centr[i]]
#     print(x_coords)
#     x_smooth = gaussian_filter1d(x_coords, sigma=2)
#     y_smooth = gaussian_filter1d(y_coords, sigma=2)
    
#     smooth_centr = [[x, y] for x, y in zip(x_smooth, y_smooth)]
#     centr[i] = smooth_centr

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
        if(cell == 203):
            if(frame == 1): 
                print(dx)
                print(dy)
                print(angle_radians)
                print(angle_degrees)
        return angle_degrees


def get_times(cell, frame):
    if frame == 0:
        dt = 0
    if frame == -1:
        return 0    
    else: 
        dt = times[cell][frame]

    return dt

def get_riemann_dist(cell, frame):

    return riemann[cell][frame]

velocities = []
riemann_distances = []
time_data = []
displacements = []

def abs_velocity_times():
    with PdfPages("abs_velocity_times.pdf") as pdf:
        for page_start in range(0, total_plots, plots_per_page):
            # Создаем страницу с форматом A4
            fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69)) 
            axes = axes.flatten()  # Преобразуем в одномерный массив для удобства работы
            
            # Генерируем графики для текущей страницы
            for i in range(plots_per_page):
    
                velocities = []
                riemann_distances = []
                time_data = []
                displacements = []
                plot_index = page_start + i
                num_frames = len(centr[plot_index])
                
                for frame in range(num_frames):
                    velocities.append(get_abs_velocity(plot_index, frame))
                    #riemann_distances.append(get_riemann_dist(plot_index, frame))
                    time_data.append(get_times(plot_index, frame))
            

                if plot_index >= total_plots:  # Если графики закончились
                    axes[i].axis('off')
                    continue
                
                axes[i].plot(time_data,velocities)
                axes[i].set_xlabel("Time")
                axes[i].set_ylabel("Velocity")
                axes[i].set_title(f"Cell {plot_index + 1}", fontsize=8)
                axes[i].tick_params(axis='both', which='major', labelsize=6)









            for j in range(len(axes)):
                if j >= plots_per_page:
                    axes[j].axis('off')


            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)  

import matplotlib.pyplot as plt
import h5py
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
import h5py
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages




def riemann_vel_times():

    with PdfPages("riemann_vel_times.pdf") as pdf:
        for page_start in range(0, total_plots, plots_per_page):
            # Создаем страницу с форматом A4
            fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69)) 
            axes = axes.flatten()  # Преобразуем в одномерный массив для удобства работы
            
            # Генерируем графики для текущей страницы
            for i in range(plots_per_page):
    
                velocities = []
                riemann_distances = []
                time_data = []
                displacements = []
                plot_index = page_start + i
                num_frames = len(centr[plot_index])
                
                for frame in range(num_frames):
                    riemann_distances.append(get_riemann_dist(plot_index, frame))
                    time_data.append(get_times(plot_index, frame))
                    velocities.append(get_abs_velocity(plot_index, frame))
            

                if plot_index >= total_plots:  
                    axes[i].axis('off')
                    continue
                
                axes[i].plot(time_data, velocities,label = 'velocity')
                axes[i].plot(time_data, riemann_distances, label = 'riemann')
                axes[i].set_xlabel("Time")
                axes[i].legend()
                axes[i].set_title(f"Cell {plot_index + 1}", fontsize=8)
                axes[i].tick_params(axis='both', which='major', labelsize=6)









            for j in range(len(axes)):
                if j >= plots_per_page:
                    axes[j].axis('off')


            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)  


def riemann_vel_times():

    with PdfPages("riemann_vel_times.pdf") as pdf:
        for page_start in range(0, total_plots, plots_per_page):
            # Создаем страницу с форматом A4
            fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69)) 
            axes = axes.flatten()  # Преобразуем в одномерный массив для удобства работы
            
            # Генерируем графики для текущей страницы
            for i in range(plots_per_page):
    
                velocities = []
                riemann_distances = []
                time_data = []
                displacements = []
                plot_index = page_start + i
                num_frames = len(centr[plot_index])
                
                for frame in range(num_frames):
                    riemann_distances.append(get_riemann_dist(plot_index, frame))
                    time_data.append(get_times(plot_index, frame))
                    velocities.append(get_abs_velocity(plot_index, frame))
            

                if plot_index >= total_plots:  
                    axes[i].axis('off')
                    continue
                
                axes[i].plot(time_data, velocities,label = 'velocity')
                axes[i].plot(time_data, riemann_distances, label = 'riemann')
                axes[i].set_xlabel("Time")
                axes[i].legend()
                axes[i].set_title(f"Cell {plot_index + 1}", fontsize=8)
                axes[i].tick_params(axis='both', which='major', labelsize=6)









            for j in range(len(axes)):
                if j >= plots_per_page:
                    axes[j].axis('off')


            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)  

def angle_rel_times():
    with PdfPages("angle_rel_times.pdf") as pdf:
        for page_start in range(0, total_plots, plots_per_page):
            # Создаем страницу с форматом A4
            fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69)) 
            axes = axes.flatten()  # Преобразуем в одномерный массив для удобства работы
            
            # Генерируем графики для текущей страницы
            for i in range(plots_per_page):
    
                velocities = []
                riemann_distances = []
                time_data = []
                displacements = []
                rel_angle = []
                plot_index = page_start + i
                num_frames = len(centr[plot_index])
                
                for frame in range(1,num_frames):
                    #riemann_distances.append(get_riemann_dist(plot_index, frame))
                    time_data.append(get_times(plot_index, frame))
                    #velocities.append(get_abs_velocity(plot_index, frame))
                    rel_angle.append(get_velocity_angle_rel(plot_index, frame))
            

                if plot_index >= total_plots:  
                    axes[i].axis('off')
                    continue
                
                axes[i].plot(time_data, rel_angle,label = 'velocity')
                #axes[i].plot(time_data, riemann_distances, label = 'riemann')
                axes[i].set_xlabel("Time")
                axes[i].set_ylabel("rel_angle")
                axes[i].set_title(f"Cell {plot_index + 1}", fontsize=8)
                axes[i].tick_params(axis='both', which='major', labelsize=6)









            for j in range(len(axes)):
                if j >= plots_per_page:
                    axes[j].axis('off')


            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)  


def anglerel_riemann_absvel_time():
    with PdfPages("angle_vel.pdf") as pdf:
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
                
                # Основной график (слева)
                ax = axes[i]
                #ax.plot(time_data1, riemann_distances, label="Riemann", color="blue")
                ax.plot(time_data1, velocities, label="Velocity", color="green")
                ax.set_xlabel("Time")
                ax.set_ylabel("Velocity", fontsize=8)
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


def plot_angle_vel_cell(cell_num):
    """
    Построение графиков для одной конкретной клетки.
    
    Parameters:
        cell_index (int): Индекс клетки (нумерация начинается с 0).
    """
    import matplotlib.pyplot as plt

    velocities = []
    riemann_distances = []
    time_data = []
    time_data1 = []
    rel_angle = []
    
    num_frames = len(centr[cell_num-1])  # Количество кадров для выбранной клетки
    
    # Сбор данных для графика
    for frame in range(1, num_frames):
        time_data.append(get_times(cell_num-1, frame))
        rel_angle.append(get_velocity_angle_rel(cell_num-1, frame))
    for frame in range(1, num_frames):
        riemann_distances.append(get_riemann_dist(cell_num-1, frame))
        time_data1.append(get_times(cell_num-1, frame))
        velocities.append(get_abs_velocity(cell_num-1, frame))
    
    # Создание графика
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Основной график (скорость и расстояние)
    ax.plot(time_data1, velocities, label="Velocity", color="green", linewidth=1.5)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Velocity", fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(loc="upper left", fontsize=8)
    
    # Вторичная шкала (углы)
    ax_angle = ax.twinx()
    ax_angle.plot(time_data, rel_angle, label="Angle", color="red", linewidth=1.5)
    ax_angle.set_ylabel("Angle (degrees)", fontsize=10, color="red")
    ax_angle.tick_params(axis="y", labelsize=8, colors="red")
    ax_angle.legend(loc="upper right", fontsize=8)
    
    # Заголовок
    ax.set_title(f"Cell {cell_num}", fontsize=12)
    
    plt.tight_layout()
    plt.show()


# plot_angle_vel_cell(87)

def trajectories():
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages("trajectories.pdf") as pdf:
        for page_start in range(0, total_plots, plots_per_page):
            fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69))  # Размер страницы A4
            axes = axes.flatten()  # Преобразование массива осей в плоский список
            
            for i in range(plots_per_page):
                plot_index = page_start + i  # Индекс текущей клетки
                
                if plot_index >= total_plots:  # Если клеток больше не осталось
                    axes[i].axis('off')  # Отключить неиспользуемые оси
                    continue
                
                centroids = np.array(centr[plot_index])  # Координаты центроидов клетки
                x_coords = centroids[:, 0]  # X-координаты
                y_coords = centroids[:, 1]  # Y-координаты
                time_steps = np.arange(len(x_coords))  # Временные шаги
                
                # Построение траектории клетки
                scatter = axes[i].scatter(
                    x_coords[1:], y_coords[1:],  
                    c=time_steps[1:],  # Цвет соответствует времени
                    cmap='plasma',             
                    marker='o',
                    edgecolor='k',
                    s=40,
                    alpha=0.7
                )
                axes[i].scatter(
                    x_coords[0], y_coords[0],  # Начальная точка
                    c='black',
                    marker='o',
                    edgecolor='k',
                    s=50,
                    alpha=0.9,
                    label='Start'
                )
                axes[i].plot(x_coords, y_coords, linestyle='-', color='gray', alpha=0.5)  # Линия траектории
                axes[i].set_title(f"Cell {plot_index + 1}", fontsize=8)
                axes[i].set_xlabel("X", fontsize=6)
                axes[i].set_ylabel("Y", fontsize=6)
                axes[i].tick_params(axis='both', which='major', labelsize=6)

                # Добавление цветовой шкалы только если график не пуст
                if len(x_coords) > 1:
                    cbar = fig.colorbar(scatter, ax=axes[i], orientation='vertical', fraction=0.046, pad=0.04)
                    cbar.set_label("Time Step (t)", rotation=270, labelpad=8, fontsize=6)
                    cbar.ax.tick_params(labelsize=6)
            
            plt.tight_layout()  # Упорядочивание элементов на странице
            pdf.savefig(fig)  # Сохранение текущей страницы в PDF
            plt.close(fig)  # Закрытие фигуры для освобождения памяти




def trajectories_inverted():
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages("trajectories_inv.pdf") as pdf:
        for page_start in range(0, total_plots, plots_per_page):
            fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69))  # Размер страницы A4
            axes = axes.flatten()  # Преобразование массива осей в плоский список
            
            for i in range(plots_per_page):
                plot_index = page_start + i  # Индекс текущей клетки
                
                if plot_index >= total_plots:  # Если клеток больше не осталось
                    axes[i].axis('off')  # Отключить неиспользуемые оси
                    continue
                
                centroids = np.array(centr[plot_index])  # Координаты центроидов клетки
                x_coords = centroids[:, 0]  # X-координаты
                y_coords = centroids[:, 1]  # Y-координаты
                time_steps = np.arange(len(x_coords))  # Временные шаги
                
                # Построение траектории клетки
                scatter = axes[i].scatter(
                    x_coords[1:], y_coords[1:],  
                    c=time_steps[1:],  # Цвет соответствует времени
                    cmap='plasma',             
                    marker='o',
                    edgecolor='k',
                    s=40,
                    alpha=0.7
                )
                axes[i].scatter(
                    x_coords[0], y_coords[0],  # Начальная точка
                    c='black',
                    marker='o',
                    edgecolor='k',
                    s=50,
                    alpha=0.9,
                    label='Start'
                )
                axes[i].plot(x_coords, y_coords, linestyle='-', color='gray', alpha=0.5)  # Линия траектории
                axes[i].set_title(f"Cell {plot_index + 1}", fontsize=8)
                axes[i].set_xlabel("X", fontsize=6)
                axes[i].set_ylabel("Y", fontsize=6)
                axes[i].tick_params(axis='both', which='major', labelsize=6)

                # Инвертирование оси Y для установки (0, 0) в верхнем левом углу
                axes[i].invert_yaxis()

                # Добавление цветовой шкалы только если график не пуст
                if len(x_coords) > 1:
                    cbar = fig.colorbar(scatter, ax=axes[i], orientation='vertical', fraction=0.046, pad=0.04)
                    cbar.set_label("Time Step (t)", rotation=270, labelpad=8, fontsize=6)
                    cbar.ax.tick_params(labelsize=6)
            
            plt.tight_layout()  # Упорядочивание элементов на странице
            pdf.savefig(fig)  # Сохранение текущей страницы в PDF
            plt.close(fig)  # Закрытие фигуры для освобождения памяти


#trajectories_inverted()



#riemann_times()





#trajectories()



# anglerel_riemann_absvel_time()



def plot_riemann_cell(plot_index):


    velocities = []
    riemann_distances = []
    time_data = []
    
    num_frames = len(centr[plot_index-1])
    
    for frame in range(1, num_frames):
        # Здесь логика такая же, как в вашем коде
        # В качестве значения берется riemann_dist/Δt
        dist_value = get_riemann_dist(plot_index-1, frame) / (get_times(plot_index-1, frame) - get_times(plot_index-1, frame - 1))
        riemann_distances.append(dist_value)
        time_data.append(get_times(plot_index-1, frame))
    
    # Создаем один график
    plt.figure(figsize=(8,6))
    plt.plot(time_data, riemann_distances, marker='o', linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Riemann Velocities")
    plt.title(f"Cell {plot_index}")
    plt.grid(True)
    plt.show()

# plot_riemann_cell(87)



import h5py
def riemann_times_with_events(cell_num):

    cell_index = cell_num - 1

    riemann_distances = []
    time_data = []


    num_frames = len(centr[cell_index])
    for frame in range(1, num_frames):
        dt = get_times(cell_index, frame) - get_times(cell_index, frame - 1)
        riemann_distances.append(get_riemann_dist(cell_index, frame) / dt)
        time_data.append(get_times(cell_index, frame))

    with h5py.File('time_events.h5', 'r') as f:
        track_i_data = f[f'/track_{cell_index + 1}'][:]
        first_two_rows = track_i_data[:2]
        time_points = np.intersect1d(first_two_rows[0, :], first_two_rows[1, :])
        print(f"Cell #{cell_index + 1} Data: {first_two_rows}")
        print(f"Cell #{cell_index + 1} Time Points: {time_points}")

    plt.figure(figsize=(8, 6))
    plt.plot(time_data, riemann_distances, label='Riemann velocity', color='blue')
    plt.xlabel("Time")
    plt.ylabel("Riemann Velocity")
    plt.title(f"Cell {cell_index + 1}", fontsize=12)
    plt.grid(True)

    for tp in time_points:
        if tp - 1 < len(time_data):  
            x = time_data[int(tp) - 1]
            y = riemann_distances[int(tp) - 1]
            plt.scatter(x, y, color="red", label="Event time" if tp == time_points[0] else "")

    plt.legend()
    plt.tight_layout()
    plt.show()



#riemann_times()
from scipy.stats import sem
from collections import defaultdict

def average_riemann_distances():
    type_data = defaultdict(list)

    # Собираем данные из файла time_events.h5
    with h5py.File('time_events.h5', 'r') as f:
        for plot_index in range(total_plots):
            track_i_data = f[f'/track_{plot_index+1}'][:]
            first_three_rows = track_i_data[:3]
            event_indices = first_three_rows[0, :].astype(int) - 1
            interval_types = first_three_rows[2, :]

            for start_idx, interval_type in enumerate(interval_types):
                if start_idx + 1 < len(event_indices):
                    start = event_indices[start_idx] + 1  # Пропускаем i = 0
                    end = event_indices[start_idx + 1]
                    segment = [get_riemann_dist(plot_index, idx)//(get_times(plot_index, idx) - get_times(plot_index, idx - 1)) for idx in range(start, end + 1)]
                    type_data[int(interval_type) if not np.isnan(interval_type) else "unclassified"].extend(segment)

    # Построение средних значений римановских дистанций с доверительными интервалами
    plt.figure(figsize=(8, 6))
    types = ["Immobile", "Confined Diffusion", "Free Diffusion", "Directed Diffusion", "Unclassified"]
    colors = ["brown", "blue", "cyan", "magenta", "black"]
    means = []
    conf_intervals = []

    for key in [0, 1, 2, 3, "unclassified"]:
        distances = type_data[key]
        if distances:  # Проверка, что данные есть
            mean = np.mean(distances)
            ci = sem(distances) * 1.96  # 95% доверительный интервал
            means.append(mean)
            conf_intervals.append(ci)
        else:
            means.append(0)
            conf_intervals.append(0)

    x = np.arange(len(types))

    # Рисуем каждую точку отдельно с её цветом и доверительным интервалом
    for idx, (mean, ci, color) in enumerate(zip(means, conf_intervals, colors)):
        plt.errorbar(x[idx], mean, yerr=ci, fmt='o', color=color, ecolor=color, elinewidth=2, capsize=5)

    plt.xticks(x, types)
    plt.ylabel("Mean Riemann Distance")
    plt.title("Average Riemann Distances by Diffusion Type with Confidence Intervals")
    plt.tight_layout()
    plt.savefig("mean_riemann_distances_with_ci.png")
    plt.show()



import matplotlib.pyplot as plt
import h5py
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
from scipy.stats import sem, ttest_ind 
def riemann_times():
    type_data = defaultdict(list)  # Словарь для сбора данных по типам движения

    with PdfPages("riemann_with_classification_95.pdf") as pdf:
        for page_start in range(0, total_plots, plots_per_page):
            # Создаем страницу с форматом A4
            fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69))
            axes = axes.flatten()  # Преобразуем в одномерный массив для удобства работы

            for i in range(plots_per_page):
                plot_index = page_start + i

                if plot_index >= total_plots:  # Если графики закончились
                    axes[i].axis('off')
                    continue

                # Инициализация данных
                riemann_distances = []
                time_data = []

                # Чтение данных из файла time_events.h5
                with h5py.File('time_events.h5', 'r') as f:
                    track_i_data = f[f'/track_{plot_index+1}'][:]
                    first_three_rows = track_i_data[:3]
                    event_indices = first_three_rows[0, :].astype(int) - 1  # Индексы событий (сдвиг на -1)
                    interval_types = first_three_rows[2, :]  # Типы интервалов

                # Извлекаем реальные времена и римановские дистанции
                time_data = [times[plot_index][idx] for idx in range(1, len(times[plot_index]))]  # Пропускаем i = 0
                riemann_distances = [
                    get_riemann_dist(plot_index, idx)/(get_times(plot_index, idx) - get_times(plot_index, idx-1))
                    for idx in range(1, len(times[plot_index]))
                ]

                # Определяем цвета для каждого типа интервала
                interval_colors = {
                    0: "brown",      # Immobile
                    1: "blue",       # Confined diffusion
                    2: "cyan",       # Free diffusion
                    3: "magenta",    # Directed diffusion
                    "unclassified": "black"  # Unclassified
                }

                # Рисуем график Riemann distances с окраской по интервалам и собираем данные
                for start_idx, interval_type in enumerate(interval_types):
                    start = event_indices[start_idx]
                    end = event_indices[start_idx + 1] if start_idx + 1 < len(event_indices) else len(time_data) - 1

                    # Проверка на корректность индексов
                    if start < len(time_data) and end < len(time_data):
                        time_segment = time_data[start:end + 1]  # Включаем конечный элемент
                        segment = riemann_distances[start:end + 1]

                        # Обрабатываем nan значения типов
                        interval_type = int(interval_type) if not np.isnan(interval_type) else "unclassified"
                        color = interval_colors.get(interval_type, "black")

                        # Добавляем данные в словарь по типам движения
                        type_data[interval_type].extend(segment)

                        # Рисуем сегмент
                        axes[i].plot(time_segment, segment, color=color)

                # Добавляем легенду только на первый график
                if i == 0:
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], color="brown", lw=2, label="Immobile"),
                        Line2D([0], [0], color="blue", lw=2, label="Confined Diffusion"),
                        Line2D([0], [0], color="cyan", lw=2, label="Free Diffusion"),
                        Line2D([0], [0], color="magenta", lw=2, label="Directed Diffusion"),
                        Line2D([0], [0], color="black", lw=2, label="Unclassified")
                    ]
                    axes[i].legend(handles=legend_elements, loc="upper right", fontsize=6)

                # Оформление графика
                axes[i].set_xlabel("Time")
                axes[i].set_ylabel("Riemann velocity")
                axes[i].set_title(f"Cell {plot_index + 1}", fontsize=8)
                axes[i].tick_params(axis='both', which='major', labelsize=6)

            # Отключаем лишние оси
            for j in range(len(axes)):
                if j >= plots_per_page:
                    axes[j].axis('off')

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def average_riemann_distances():
    type_data = defaultdict(list)

    # Собираем данные из файла time_events.h5
    with h5py.File('time_events.h5', 'r') as f:
        for plot_index in range(total_plots):
            track_i_data = f[f'/track_{plot_index+1}'][:]
            first_three_rows = track_i_data[:3]
            event_indices = first_three_rows[0, :].astype(int) - 1
            interval_types = first_three_rows[2, :]

            for start_idx, interval_type in enumerate(interval_types):
                if start_idx + 1 < len(event_indices):
                    start = event_indices[start_idx] + 1  # Пропускаем i = 0
                    end = event_indices[start_idx + 1]
                    segment = [get_riemann_dist(plot_index, idx) for idx in range(start, end + 1)]
                    type_data[int(interval_type) if not np.isnan(interval_type) else "unclassified"].extend(segment)

    # Построение средних значений римановских дистанций с доверительными интервалами
    plt.figure(figsize=(8, 6))
    types = ["Immobile", "Confined Diffusion", "Free Diffusion", "Directed Diffusion", "Unclassified"]
    colors = ["brown", "blue", "cyan", "magenta", "black"]
    means = []
    conf_intervals = []

    for key in [0, 1, 2, 3, "unclassified"]:
        distances = type_data[key]
        if distances:  # Проверка, что данные есть
            mean = np.mean(distances)
            ci = sem(distances) * 1.96  # 95% доверительный интервал
            means.append(mean)
            conf_intervals.append(ci)
        else:
            means.append(0)
            conf_intervals.append(0)

    x = np.arange(len(types))

    # Рисуем каждую точку отдельно с её цветом и доверительным интервалом
    for idx, (mean, ci, color) in enumerate(zip(means, conf_intervals, colors)):
        plt.errorbar(x[idx], mean, yerr=ci, fmt='o', color=color, ecolor=color, elinewidth=2, capsize=5)

    plt.xticks(x, types)
    plt.ylabel("Mean Riemann Distance")
    plt.title("Average Riemann Distances by Diffusion Type with Confidence Intervals")
    plt.tight_layout()
    plt.savefig("mean_riemann_distances_095.png")
    plt.show()
 
    keys = [0, 1, 2, 3, "unclassified"]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            group1 = type_data[keys[i]]
            group2 = type_data[keys[j]]
            if group1 and group2:  # Проверка, что обе группы не пустые
                t_stat, p_val = ttest_ind(group1, group2, equal_var=False)  # Welch's T-test
                print(f"T-test {types[i]} and {types[j]}: p-value = {p_val:.4e}, >0.05: {p_val > 0.05}")


def average_velocity():
    type_data = defaultdict(list)

    # Собираем данные из файла time_events.h5
    with h5py.File('time_events.h5', 'r') as f:
        for plot_index in range(total_plots):
            track_i_data = f[f'/track_{plot_index+1}'][:]
            first_three_rows = track_i_data[:3]
            event_indices = first_three_rows[0, :].astype(int) - 1
            interval_types = first_three_rows[2, :]

            for start_idx, interval_type in enumerate(interval_types):
                if start_idx + 1 < len(event_indices):
                    start = event_indices[start_idx] + 1  # Пропускаем i = 0
                    end = event_indices[start_idx + 1]
                    segment = [get_abs_velocity(plot_index, idx) for idx in range(start, end + 1)]
                    type_data[int(interval_type) if not np.isnan(interval_type) else "unclassified"].extend(segment)

    # Построение средних значений скоростей с доверительными интервалами
    plt.figure(figsize=(8, 6))
    types = ["Immobile", "Confined Diffusion", "Free Diffusion", "Directed Diffusion", "Unclassified"]
    colors = ["brown", "blue", "cyan", "magenta", "black"]
    means = []
    conf_intervals = []

    for key in [0, 1, 2, 3, "unclassified"]:
        velocities = type_data[key]
        if velocities:  # Проверка, что данные есть
            mean = np.mean(velocities)
            ci = sem(velocities) * 1.96  # 95% доверительный интервал
            means.append(mean)
            conf_intervals.append(ci)
        else:
            means.append(0)
            conf_intervals.append(0)

    x = np.arange(len(types))

    # Рисуем каждую точку отдельно с её цветом и доверительным интервалом
    for idx, (mean, ci, color) in enumerate(zip(means, conf_intervals, colors)):
        plt.errorbar(x[idx], mean, yerr=ci, fmt='o', color=color, ecolor=color, elinewidth=2, capsize=5)

    plt.xticks(x, types)
    plt.ylabel("Mean Velocity")
    plt.title("Motion types")
    plt.tight_layout()
    plt.savefig("mean_velocity_095.png")
    plt.show()

def average_angle():
    type_data = defaultdict(list)

    # Собираем данные из файла time_events.h5
    with h5py.File('time_events.h5', 'r') as f:
        for plot_index in range(total_plots):
            track_i_data = f[f'/track_{plot_index+1}'][:]
            first_three_rows = track_i_data[:3]
            event_indices = first_three_rows[0, :].astype(int) - 1
            interval_types = first_three_rows[2, :]

            for start_idx, interval_type in enumerate(interval_types):
                if start_idx + 1 < len(event_indices):
                    start = event_indices[start_idx] + 1  # Пропускаем i = 0
                    end = event_indices[start_idx + 1]
                    segment = [get_velocity_angle_rel(plot_index, idx) for idx in range(start, end + 1)]
                    type_data[int(interval_type) if not np.isnan(interval_type) else "unclassified"].extend(segment)

    # Построение средних значений углов с доверительными интервалами
    plt.figure(figsize=(8, 6))
    types = ["Immobile", "Confined Diffusion", "Free Diffusion", "Directed Diffusion", "Unclassified"]
    colors = ["brown", "blue", "cyan", "magenta", "black"]
    means = []
    conf_intervals = []

    for key in [0, 1, 2, 3, "unclassified"]:
        angles = type_data[key]
        if angles:  # Проверка, что данные есть
            mean = np.mean(angles)
            ci = sem(angles) * 1.96  # 95% доверительный интервал
            means.append(mean)
            conf_intervals.append(ci)
        else:
            means.append(0)
            conf_intervals.append(0)

    x = np.arange(len(types))

    # Рисуем каждую точку отдельно с её цветом и доверительным интервалом
    for idx, (mean, ci, color) in enumerate(zip(means, conf_intervals, colors)):
        plt.errorbar(x[idx], mean, yerr=ci, fmt='o', color=color, ecolor=color, elinewidth=2, capsize=5)

    plt.xticks(x, types)
    plt.ylabel("Mean Angle")
    plt.title("Motion types")
    plt.tight_layout()
    plt.savefig("mean_angle_095.png")
    plt.show()

# average_riemann_distances()
# average_velocity()
# average_angle()


#riemann_times()


def riemann_single_cell_classification(cell_number):
    """
    Построение графика Riemann distances для клетки с заданным номером (cell_number).
    """
    # Преобразование номера клетки в индекс
    cell_index = cell_number - 1  # Поскольку нумерация начинается с 1, а индексация с 0

    # Инициализация данных
    riemann_distances = []
    time_data = []
    type_data = defaultdict(list)  # Словарь для сбора данных по типам движения

    # Чтение данных из файла time_events.h5
    with h5py.File('time_events_90.h5', 'r') as f:
        track_i_data = f[f'/track_{cell_index+1}'][:]
        first_three_rows = track_i_data[:3]
        event_indices = first_three_rows[0, :].astype(int) - 1  # Индексы событий (сдвиг на -1)
        interval_types = first_three_rows[2, :]  # Типы интервалов

    # Извлекаем реальные времена и римановские дистанции
    time_data = [times[cell_index][idx] for idx in range(1, len(times[cell_index]))]  # Пропускаем i = 0
    riemann_distances = [
        get_riemann_dist(cell_index, idx) / (get_times(cell_index, idx) - get_times(cell_index, idx - 1))
        for idx in range(1, len(times[cell_index]))
    ]

    # Определяем цвета для каждого типа интервала
    interval_colors = {
        0: "brown",      # Immobile
        1: "blue",       # Confined diffusion
        2: "cyan",       # Free diffusion
        3: "magenta",    # Directed diffusion
        "unclassified": "black"  # Unclassified
    }

    # Создаем график
    fig, ax = plt.subplots(figsize=(8, 6))

    # Рисуем график Riemann distances с окраской по интервалам
    for start_idx, interval_type in enumerate(interval_types):
        start = event_indices[start_idx]
        end = event_indices[start_idx + 1] if start_idx + 1 < len(event_indices) else len(time_data) - 1

        # Проверка на корректность индексов
        if start < len(time_data) and end < len(time_data):
            time_segment = time_data[start:end + 1]  # Включаем конечный элемент
            segment = riemann_distances[start:end + 1]

            # Обрабатываем nan значения типов
            interval_type = int(interval_type) if not np.isnan(interval_type) else "unclassified"
            color = interval_colors.get(interval_type, "black")

            # Добавляем данные в словарь по типам движения
            type_data[interval_type].extend(segment)

            # Рисуем сегмент
            ax.plot(time_segment, segment, color=color)

    # Добавляем легенду
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="brown", lw=2, label="Immobile"),
        Line2D([0], [0], color="blue", lw=2, label="Confined Diffusion"),
        Line2D([0], [0], color="cyan", lw=2, label="Free Diffusion"),
        Line2D([0], [0], color="magenta", lw=2, label="Directed Diffusion"),
        Line2D([0], [0], color="black", lw=2, label="Unclassified")
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # Оформление графика
    ax.set_xlabel("Time")
    ax.set_ylabel("Riemann velocity")
    ax.set_title(f"Cell {cell_number}", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Сохраняем график и показываем
    plt.tight_layout()
    plt.savefig(f"riemann_single_cell_{cell_number}_classification_90.png")
    plt.show()

riemann_single_cell_classification(87)