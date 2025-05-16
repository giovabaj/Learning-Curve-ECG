import os
from os.path import join
import numpy as np
import pandas as pd


def get_auc(paths):
    """
    Load and collect AUC values from multiple .npy files.

    Args:
        paths (list): List of paths to AUC files.

    Returns:
        np.ndarray: Transposed array of AUC values (shape: repetitions x sizes).
    """
    auc_all = []
    for path in paths:
        auc = np.load(path)
        auc_all.append(auc)
    return np.array(auc_all).transpose()


def main():
    """
    Iterate over detection and prediction tasks, aggregate AUCs, and save them as CSV files.
    """
    for task in ["detection", "prediction"]:
        path = f"results/{task}/"  # method 3, sample 2000, 100 repetitions, stable config
        sizes = np.sort([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])
        paths = [join(path, size, "auc_test.npy") for size in sizes]
        auc_all = get_auc(paths, sizes)
        res = pd.DataFrame(auc_all, columns=sizes)
        res.to_csv(f"results/{task}/auc_test_all.csv", index=False)


if __name__ == "__main__":
    main()
