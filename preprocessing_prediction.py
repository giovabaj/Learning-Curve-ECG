import pandas as pd
import numpy as np
import os
from os.path import join
from multiprocessing import Pool
import argparse

from scipy.signal import resample
from utils import filter_ecgs


def main():
    # parsing arguments:
    parser = argparse.ArgumentParser(description='ECG - make dataset')
    parser.add_argument('--n_samples', type=int, help='Number of samples to extract')
    parser.add_argument('--path_df', type=str, help='Path to the csv dataframe')
    parser.add_argument('--path_signals', type=str, help='Path to signals')
    parser.add_argument('--freq_resampling', type=int, default=250, help='Resampling frequency')
    parser.add_argument('--out_path', type=str, default='data/prediction', help='Output path')
    args = parser.parse_args()
    n_samples = args.n_samples
    freq_resample = args.freq_resampling
    random_seed = 42

    # Importing dataframe
    df = pd.read_csv(args.path_df, encoding="iso-8859-1", low_memory=False)
    df = df.drop(df[(df["FA"] == 0) & (df["duration"] < 1825)].index, axis=0)  # Drop no-AF patients with FUP < 5 years
    df = df.sample(int(1.2*n_samples), random_state=random_seed)  # Randomly sample patients
    print("dataframe loaded")
    print("   checking if all exams are present")
    exams = df["ExamID"].values.astype(int)
    labels = df["FA"].values
    filenames = np.array([join(args.path_signals, str(indx) + '.npy') for indx in exams])
    ncores = 10  # number of cores, e.g. 8
    pool = Pool(ncores)
    # create a list of booleans corresponding to whether each file is in path or not.
    selector = np.array(pool.map(os.path.isfile, filenames))
    exams_exist = exams[selector]
    filenames_exist = filenames[selector]
    labels = labels[selector]
    print(f"   {len(exams_exist)} exams out of {len(exams)} are present")
    print(f"   Selecting only {n_samples} of them")
    exams_exist = exams_exist[:n_samples]  # select only the required number of exams
    filenames_exist = filenames_exist[:n_samples]  # select only the required number of exams
    labels = labels[:n_samples]

    # Resample the signal to the required frequency
    n_points_resample = 10 * freq_resample  # number of time points after resampling
    ecg_data = np.empty([len(exams_exist), 12, n_points_resample], dtype='float32')
    print("   Start")
    for i, filename in enumerate(filenames_exist):
        if i % 100 == 0:
            print(f'\r   Processing file number {i + 1}', end='\r')
        d = np.load(filename).astype('float32').swapaxes(0, 1)  # [:1, :]  # keeping only first channel
        ecg_data[i, :, :] = resample(d, n_points_resample, axis=1).astype('float32')
    # Filtering the signal
    print("\n   filtering the signal")
    ecg_data = filter_ecgs(ecg_data, fs=freq_resample, low_cut=0.67, high_cut=100)

    # saving ECG matrix
    filename = join(args.out_path, f'data_{n_samples}_{freq_resample}hz')
    print(f'   Saving ECGs to {filename}')
    os.makedirs(args.out_path, exist_ok=True)
    np.savez_compressed(filename, ecg_data, labels, exams_exist)


if __name__ == "__main__":
    main()
