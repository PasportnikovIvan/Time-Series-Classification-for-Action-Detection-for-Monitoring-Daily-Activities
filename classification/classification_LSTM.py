# /classification/classification_LSTM.py
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

class LSTMClassifier:
    def __init__(self, input_shape, num_classes):
        """
        Initialize the LSTM classifier.

        Args:
            input_shape (tuple): Shape of the input data (timesteps, features).
            num_classes (int): Number of output classes.
        """
        self.model = self.create_model(input_shape, num_classes)

    def create_model(self, input_shape, num_classes):
        """
        Create an LSTM model.

        Args:
            input_shape (tuple): Shape of the input data (timesteps, features).
            num_classes (int): Number of output classes.

        Returns:
            Sequential: Compiled LSTM model.
        """
        model = Sequential([
            LSTM(64, input_shape=input_shape),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def prepare_data(self, movement_vectors, max_length=100):
        """
        Prepare data for LSTM training.

        Args:
            movement_vectors (dict): Dictionary of movement vectors.
            max_length (int): Maximum length of sequences.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        all_vectors = []
        labels = []
        for i, (movement, vectors) in enumerate(movement_vectors.items()):
            all_vectors.extend(vectors)
            labels.extend([i] * len(vectors))
        X = np.array(all_vectors).reshape(-1, max_length, 3)  # Reshape for LSTM
        y = np.array(labels)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def fit(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
        """
        Train the LSTM model.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Training labels.
            X_test (np.ndarray): Test data.
            y_test (np.ndarray): Test labels.
            epochs (int): Number of epochs.
            batch_size (int): Batch size.

        Returns:
            History: Training history.
        """
        history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.

        Args:
            X_test (np.ndarray): Test data.
            y_test (np.ndarray): Test labels.

        Returns:
            tuple: (loss, accuracy)
        """
        return self.model.evaluate(X_test, y_test)

if __name__ == "__main__":
    movement_vectors = {
        'standing': np.load('output_vectors/standing_vectors.npy'),
        'sitting': np.load('output_vectors/sitting_vectors.npy'),
        'falling': np.load('output_vectors/falling_vectors.npy')
    }
    max_length = 100
    num_classes = len(movement_vectors)
    classifier = LSTMClassifier(input_shape=(max_length, 3), num_classes=num_classes)
    X_train, X_test, y_train, y_test = classifier.prepare_data(movement_vectors, max_length=max_length)
    history = classifier.fit(X_train, y_train, X_test, y_test, epochs=10, batch_size=32)
    loss, accuracy = classifier.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")