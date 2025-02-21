# /classification/main.py
import numpy as np
from classification.data_organization import DataOrganizer
from classification.classification_SVD import SVDClassifier
from classification.classification_LSTM import LSTMClassifier
from classification.clustering_DTW import DTWClustering

def main():
    # Organize Data
    camera_file_paths = [
        'dataset/cameraLandmarks/falling/falling_01_cameralandmarksdata_ivan.json',
        'dataset/cameraLandmarks/falling/falling_02_cameralandmarksdata_ivan.json',
        'dataset/cameraLandmarks/standing/standing_01_cameralandmarksdata_ivan.json',
        'dataset/cameraLandmarks/standing/standing_02_cameralandmarksdata_ivan.json',
        'dataset/cameraLandmarks/sitting/sitting_01_cameralandmarksdata_ivan.json',
        'dataset/cameraLandmarks/sitting/sitting_02_cameralandmarksdata_ivan.json'
    ]
    global_file_paths = [
        'dataset/globalLandmarks/falling/falling_01_globallandmarksdata_ivan.json',
        'dataset/globalLandmarks/falling/falling_02_globallandmarksdata_ivan.json',
        'dataset/globalLandmarks/standing/standing_01_globallandmarksdata_ivan.json',
        'dataset/globalLandmarks/standing/standing_02_globallandmarksdata_ivan.json',
        'dataset/globalLandmarks/sitting/sitting_01_globallandmarksdata_ivan.json',
        'dataset/globalLandmarks/sitting/sitting_02_globallandmarksdata_ivan.json'
    ]
    file_paths = camera_file_paths
    
    organizer = DataOrganizer(file_paths)
    organizer.extract_all_data()
    organizer.pad_sequences(max_length=100)
    organizer.save_vectors('output_vectors')

    standing_data = organizer.get_movement_data('standing')
    if standing_data is not None:
        print(f"Standing data shape: {standing_data.shape}")
        
    movement_vectors = {
        'standing': np.load('output_vectors/standing_vectors.npy'),
        'sitting': np.load('output_vectors/sitting_vectors.npy'),
        'falling': np.load('output_vectors/falling_vectors.npy')
    }

    # SVD Classifier
    svd_classifier = SVDClassifier(n_components=50, n_neighbors=3)
    X_train, X_test, y_train, y_test = svd_classifier.prepare_data(movement_vectors)
    svd_classifier.fit(X_train, y_train)
    svd_accuracy = svd_classifier.evaluate(X_test, y_test)
    print(f"SVD Classifier Accuracy: {svd_accuracy:.4f}")

    # LSTM Classifier
    max_length = 100
    num_classes = len(movement_vectors)
    lstm_classifier = LSTMClassifier(input_shape=(max_length, 3), num_classes=num_classes)
    X_train, X_test, y_train, y_test = lstm_classifier.prepare_data(movement_vectors, max_length=max_length)
    lstm_classifier.fit(X_train, y_train, X_test, y_test, epochs=10, batch_size=32)
    lstm_loss, lstm_accuracy = lstm_classifier.evaluate(X_test, y_test)
    print(f"LSTM Classifier Accuracy: {lstm_accuracy:.4f}")

    # Perform Clustering Using DTW
    all_vectors = np.vstack(list(movement_vectors.values()))
    clustering = DTWClustering(n_clusters=3)
    clustering.cluster_and_visualize(all_vectors, n_components=2)

if __name__ == "__main__":
    main()