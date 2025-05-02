import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_nose_trajectory(file_path, title, color='b'):
    """
    Визуализирует траекторию носа из JSON-файла в 3D.

    Args:
        file_path (str): Путь к JSON-файлу.
        title (str): Заголовок графика.
        color (str): Цвет траектории (по умолчанию синий).
    """
    # Читаем JSON-файл
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Извлекаем координаты носа из каждого кадра
    nose_coords = [frame['landmarks']['nose'] for frame in data['data']]
    x = [coord[0] for coord in nose_coords]
    y = [coord[1] for coord in nose_coords]
    z = [coord[2] for coord in nose_coords]
    
    # Создаем 3D-график
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color=color, label=title)
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    ax.set_zlabel('Z (м)')
    ax.set_title(title)
    ax.legend()
    plt.show()