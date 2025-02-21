# /classification/classification_SVD.py
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class SVDClassifier:
    def __init__(self, n_components=50, n_neighbors=3):
        """
        Initialize the SVD classifier.

        Args:
            n_components (int): Number of components for SVD.
            n_neighbors (int): Number of neighbors for k-NN.
        """
        self.svd = TruncatedSVD(n_components=n_components)
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    def prepare_data(self, movement_vectors):
        """
        Prepare data for SVD and classification.

        Args:
            movement_vectors (dict): Dictionary of movement vectors.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        all_vectors = []
        labels = []
        for i, (movement, vectors) in enumerate(movement_vectors.items()):
            all_vectors.extend(vectors)
            labels.extend([i] * len(vectors))
        X = np.vstack(all_vectors)
        y = np.array(labels)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def fit(self, X_train, y_train):
        """
        Fit the SVD and k-NN models.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Training labels.
        """
        reduced_X_train = self.svd.fit_transform(X_train)
        self.knn.fit(reduced_X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.

        Args:
            X_test (np.ndarray): Test data.
            y_test (np.ndarray): Test labels.

        Returns:
            float: Accuracy of the model.
        """
        reduced_X_test = self.svd.transform(X_test)
        return self.knn.score(reduced_X_test, y_test)

if __name__ == "__main__":
    movement_vectors = {
        'standing': np.load('output_vectors/standing_vectors.npy'),
        'sitting': np.load('output_vectors/sitting_vectors.npy'),
        'falling': np.load('output_vectors/falling_vectors.npy')
    }
    classifier = SVDClassifier(n_components=50, n_neighbors=3)
    X_train, X_test, y_train, y_test = classifier.prepare_data(movement_vectors)

    # Train the model
    classifier.fit(X_train, y_train)

    accuracy = classifier.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")