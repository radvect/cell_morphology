import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

riemann = np.load('/home/pavel/cell_morphology/nov30/riemann_distances.npy', allow_pickle=True)
times = np.load('/home/pavel/cell_morphology/nov30/times.npy', allow_pickle=True)
centr = np.load('/home/pavel/cell_morphology/nov30/centroids.npy', allow_pickle=True)

print(times[79])
print(centr.shape)
print(centr[0].shape)
print(centr[0][0].shape)


import numpy as np
from scipy.io import savemat


data = centr

# Подготовка данных для сохранения в структуре
tracks = {}
for i, trajectory in enumerate(data):
    n_frames = trajectory.shape[0]
    row = np.zeros(n_frames * 8)  # Создаем строку для текущей траектории
    for j, (x, y) in enumerate(trajectory):
        start_idx = j * 8  # Начальная позиция для текущего кадра
        row[start_idx] = x  # X координата
        row[start_idx + 1] = y  # Y координата
        # Остальные 6 элементов уже нули
    tracks[f"track_{i+1}"] = row  # Сохраняем траекторию как отдельное поле

# Сохранение данных в .mat файл
output_path = "trajectory_data.mat"
savemat(output_path, {'tracks': tracks})

print(f"Файл сохранен как {output_path}")

# print(tracks)