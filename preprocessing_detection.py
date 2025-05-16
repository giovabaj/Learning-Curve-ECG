from scipy.signal import resample
import numpy as np
from os.path import join
import re
import pandas as pd
from tqdm import tqdm
import wfdb
from joblib import Parallel, delayed
from utils import filter_ecgs


root_path = 'data/detection/original_data/'  # Path to the folder with the data


def read_paths():
    """
    Reads and constructs full paths to all ECG recordings listed in nested 'RECORDS' files.
    Returns a list of complete paths to individual ECG recordings.
    """
    with open(join(root_path, 'RECORDS'), 'r') as file:
        lines = []
        for line in file:
            line = join(root_path, line.strip())
            lines.append(line)
    paths = []
    for i in tqdm(lines):
        path = i
        with open(join(path, 'RECORDS'), 'r') as file:
            for line in file:  # Iterate over the lines of the file
                line = line.strip()  # Remove the newline character at the end of the line
                paths.append(path + line)  # Append the line to the list
    return paths


def get_metadata(path, codes_positions, dataset, codes, codes_dict):
    """
    Extracts metadata and disease code labels from a given ECG record.
    Returns a list containing general metadata followed by a one-hot encoded disease label vector.
    """
    data = wfdb.rdheader(path)
    comments = data.__dict__['comments']
    match = None
    for j in range(len(comments)):
        if re.match('Dx', comments[j]) != None:
            match = j
            break

    disease_codes = comments[match].replace('Dx: ', '')
    disease_codes_list = [int(i) for i in disease_codes.split(',')]

    to_append = [
        dataset,
        data.__dict__['record_name'],
        path,
        data.__dict__['fs'],
        data.__dict__['sig_len'],
        data.__dict__['n_sig'],
        data.__dict__['sig_len'] / data.__dict__['fs']
    ]

    to_append_diseases = np.zeros(len(codes), dtype=int)
    for j in disease_codes_list:
        to_append_diseases[codes_positions[codes_dict[j]]] = 1

    return to_append + to_append_diseases.tolist()


def get_information(name='information_file.csv'):
    """
    Loads disease code mappings, constructs metadata for all ECG records, and saves it as a CSV.
    Returns the metadata DataFrame.
    """
    codes = pd.read_csv('data/detection/original_data/Dx_map.csv')
    codes_dict = dict()
    for i in tqdm(range(len(codes))):
        key = codes.iloc[i, 1]
        value = codes.iloc[i, 2]
        codes_dict[key] = value

    codes_list = codes.iloc[:, 2].to_list()
    col = ['dataset_name', 'record_name', 'path', 'freq', 'n_sample', 'n_sig', 'time'] + codes_list
    review = pd.DataFrame(columns=col)
    codes_positions = {}
    for i in tqdm(range(len(codes_list))):
        codes_positions[codes_list[i]] = i

    paths = read_paths()
    temps = Parallel(n_jobs=-1)(
        delayed(get_metadata)(path, codes_positions, path.split('/')[-3], codes, codes_dict) for path in paths)
    for temp in temps:
        if temp != None:
            review.loc[len(review)] = temp
    review.to_csv(join(root_path, name), index=False)
    return review


def get_data(df,
             target_fs=128,
             ecg_time=10,
             ):
    """
    Loads and resamples ECG signals and saves them as .npy files, with labels.

    Parameters:
    - df: metadata DataFrame
    - target_fs: target sampling frequency after resampling
    - ecg_time: duration in seconds of the signal to extract
    """
    # Extract paths
    df = df[df['time'] == 10]  # keep only ECGs with 10 seconds length
    paths_af = list(df[df['AF'] == 1].path) + list(df[df['AFL'] == 1].path)
    paths_non_af = list(df[(df['AF'] == 0) & (df['AFL'] == 0)].path)
    labels = np.concatenate((np.ones(len(paths_af)), np.zeros(len(paths_non_af))))
    paths = paths_af + paths_non_af
    # Get ECG data
    n_points_resample = ecg_time * target_fs  # number of time points after resampling
    ecg_data = np.empty([len(paths), n_points_resample], dtype='float32')
    for i in tqdm(range(len(paths))):
        signal, meta = wfdb.rdsamp(paths[i], channel_names=['I'])  # extract 1st channel of signal
        ecg_data[i, :] = resample(signal, n_points_resample, axis=0).astype('float32')[:, 0]
    # Filter signals
    ecg_data = filter_ecgs(ecg_data, low_cut=0.67, high_cut=50, fs=target_fs, order=1)
    # Save data
    print('Saving data...')
    print(ecg_data.shape)
    np.save(join(root_path, 'ecg_data'), ecg_data)
    np.save(join(root_path, 'labels'), labels)


def main():
    """
    Pipeline:
    1. Generate metadata.
    2. Load and process ECG data.
    """
    df_info = get_information()
    get_data(df_info)


if __name__ == "__main__":
    main()
