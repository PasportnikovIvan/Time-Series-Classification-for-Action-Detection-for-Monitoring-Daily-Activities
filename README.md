# ivan-motion-classification
# Time Series Classification for Imitation Learning

## Project Overview
This project focuses on the classification of time series data representing human actions and dynamic gestures to enable robots to learn by imitation. Given that actions may vary in duration and complexity based on various factors (e.g., objects involved, execution speed, initial positions), we will employ time series classification to recognize these actions. This project will explore different classification approaches, including both traditional methods (e.g., Dynamic Time Warping) and deep learning models (e.g., LSTM, Transformer networks).

### Key Objectives
1. **Review and Compare Classification Methods**: Evaluate distance-based methods (e.g., Dynamic Time Warping) and deep learning methods (e.g., LSTM, Transformer network).
2. **Simulator and Dataset Preparation**: Use CoppeliaSim to create a dataset of time series data representing robotic actions. The dataset will include labels for various actions (e.g., Push to left, rotate, make a circle) and account for varying parameters like initial end-effector positions, manipulated objects, and execution speeds.
3. **Clustering and Visualization**: Cluster actions using Dynamic Time Warping and visualize distances between time series to analyze parameter impact on classification.
4. **Gesture Classification and Analysis**: Apply neural networks for classifying time series and compare results against unsupervised DTW-based classes.
5. **Additional Analysis Using Existing Dataset**: Incorporate pre-collected data of dynamic gestures performed by hand movements to further validate the model.

## Project Details
- **Topic Title**: Classification of Time Series for Imitation Learning
- **Supervisor**: Mgr. Karla Štěpánová, Ph.D., Department of Cybernetics
- **Semester**: B241 - Zimní (Winter) 2024/2025; B242 - Letní (Summer) 2024/2025
- **Field of Study**: Electrical Engineering, Electronics, Bioinformatics, Informatics, Cybernetics, Software Engineering, etc.

## Research and Development Phases

### Step 1: Data Collection - Daily Activities and Gestures
We define a set of common activities and gestures that will be used to train and evaluate the model:

#### Activities
- **Daily Home Activities**: 
  - Sitting, standing, walking, lying down.
  - Potential hazards: falling, fainting, hitting furniture, lying down too quickly, etc.
- **Work-related Activities**:
  - Hammering, screwing, cutting wood.
  - Possible injuries: hitting a finger or leg, tool mishandling, fainting.
  
#### Specific Gestures
- Communication gestures including thumbs up, yes (nodding), no (shaking head), stop, come here, and help.

### Step 2: MediaPipe for 3D Keypoint and Object Detection
Utilize MediaPipe's body tracking capabilities and 3D object detection (e.g., AruCo markers) to capture and label data:
1. **3D Keypoints Detection**: Body pose estimation to identify movements and gestures.
2. **Object Detection**: Recognize objects in the scene using RealSense camera and MediaPipe.

### Step 3: Data Collection and Preprocessing
1. **Data Recording**: Capture data with RealSense, including 3D keypoints, object positions, sound data, and contextual labels (e.g., room type, available objects).
2. **Data Segmentation**: Split data into training, validation, and test sets for analysis.

### Step 4: Classifier Development
1. **Classifier Models**: Implement models to detect and classify standard and non-standard movements.
2. **Dynamic Time Warping (DTW)**: Align new actions to existing data for comparative analysis.
3. **Neural Network Classifier**: Use LSTM or Transformer models for temporal classification.

### Step 5: Expanded Classification Features
Incorporate additional information for enhanced classification accuracy:
- **Contextual Data**: Include room-based conditions (e.g., lying down typical in bedroom) to improve contextual accuracy.
- **Routines**: Use historical activity data to predict the likelihood of upcoming actions.

## Requirements
- **Programming Language**: Python
- **Libraries and Tools**:
  - [MediaPipe](https://google.github.io/mediapipe/solutions/pose.html) for 3D keypoint and object detection.
  - [RealSense SDK](https://github.com/IntelRealSense/librealsense) for video and depth data.
  - Deep learning libraries (e.g., PyTorch or TensorFlow for LSTM/Transformer models).
  - [CoppeliaSim](https://www.coppeliarobotics.com/) for robotic simulation and dataset creation.

## Literature and References
1. Jiang, Weiwei. "Time series classification: Nearest neighbor versus deep learning models." SN Applied Sciences 2.4 (2020): 1-17.
2. Fawaz, Hassan Ismail, et al. "Deep learning for time series classification: a review." Data mining and knowledge discovery 33.4 (2019): 917-963.
3. Hsu, Che-Jui, et al. "Flexible dynamic time warping for time series classification." Procedia Computer Science 51 (2015): 2838-2842.
4. Michal Mikeska: Time-Series Classification for Action Detection in Imitation learning, Bachelor thesis.
5. [HOMER+ Dataset](https://github.com/GT-RAIL/rail_tasksim/tree/homer/routines) - For routine-based prediction.
6. Su, Zhidong, and Weihua Sheng. "Context-Aware Conversation Adaptation for Human-Robot Interaction." IROS 2024.

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>

2. **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt

3. **RealSense Setup**:
    Follow the instructions in the [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense) repository for camera setup.

4. **CoppeliaSim**:
    Download and install CoppeliaSim from [here](https://www.coppeliarobotics.com/).

##Usage
Usage of the project.

##License
This project is licensed under CIIRC CVUT.