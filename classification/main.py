import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

def plot_nose_trajectory(file_path, action_label):  
    with open(file_path, 'r') as f:
        data = json.load(f)
    nose_coords = [frame['landmarks']['nose'] for frame in data['data']]
    x = [coord[0] for coord in nose_coords]
    y = [coord[1] for coord in nose_coords]
    z = [coord[2] for coord in nose_coords]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label=action_label)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    plt.show()

plot_nose_trajectory('dataset/cameraLandmarks/falling/falling_01_cameralandmarksdata_ivan.json', 'Falling')
plot_nose_trajectory('dataset/cameraLandmarks/lying/lying_04_cameralandmarksdata_ivan.json', 'Lying')