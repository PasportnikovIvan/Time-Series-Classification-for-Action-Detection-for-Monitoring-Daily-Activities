# cross_validation.py

import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from data_utils import collect_all_files
from dtw_distances import classify_with_knn_dtw

def cross_validate_knn_dtw(
    global_dir: str,
    actions,
    n_splits: int = 5,
    k: int = 3,
    n_clusters: int = 3,
    random_state: int = 42
):
    """
    5-fold cross-validation for your k-NN+DTW classifier.
    - global_dir: path to your dataset root (e.g. 'dataset/globalLandmarks')
    - actions: list of sub-folders (e.g. ['falling','sitting',…])
    - n_splits: number of folds
    - k: parameter for k-NN
    """
    # 1) gather all (file_path, label) tuples
    all_files = collect_all_files(actions, global_dir)
    data = [(fp, os.path.basename(os.path.dirname(fp))) for fp in all_files]

    # 2) prepare CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracies = []

    # 3) loop folds
    for fold, (train_idx, test_idx) in enumerate(kf.split(data), 1):
        train = [data[i] for i in train_idx]
        test  = [data[i] for i in test_idx]
        # classify_with_knn_dtw expects list[(path,action)] for train & test
        preds, trues = classify_with_knn_dtw(train, test, k=k, n_clusters=n_clusters)
        acc = accuracy_score(trues, preds)
        print(f"Fold {fold} — accuracy: {acc:.3f}\n")
        accuracies.append(acc)

    return accuracies