import os
import json
import argparse
from datetime import date


def get_args():
    """
    Get arguments from terminal.
    Output: configuration arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=120, help="Maximum number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--dropout_rate', type=float, default=0.3, help="Dropout rate")
    parser.add_argument('--patience', type=int, default=30, help="Early stopping patience")
    # parser.add_argument('--path_df', type=str, default='/coorte/ASUGI/ECG_FA_5Y_v06.csv', help="Dataframe path")
    parser.add_argument('--n_folds', type=int, default=10, help="Number of CV folds")
    parser.add_argument('--n_samples', type=int, default=1000, help="Number of ECG samples")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    dropout_rate = args.dropout_rate
    patience = args.patience
    # path_df = args.path_df
    n_folds = args.n_folds
    n_samples = args.n_samples
    undersampling_flag = args.undersampling_flag
    ratio = args.ratio
    random_seed = args.random_seed

    # create folder to save results
    path_results = get_path_results(parent_results_folder="results/", n=n_samples)

    # saving configs to a json file
    config_dict = vars(args)
    with open(path_results + '/config.json', 'w') as outfile:
        json.dump(config_dict, outfile)

    return path_results, n_epochs, batch_size, lr, dropout_rate, patience, n_folds,\
        random_seed


def get_path_results(parent_results_folder, n):
    """
    Make results directory.
    Input:
     - parent_results_folder:
     - n: number of samples
    Output:
     - results folder path
    """
    # Set directory name based on number of samples
    results_folder = parent_results_folder + str(n) + "_" + str(date.today())

    path_res = os.getcwd() + "/" + results_folder

    # Make directory avoiding over-writing
    if not os.path.exists(path_res):
        path_res = path_res + "/"
        os.makedirs(path_res)  # if dir does not exist, I create it
    else:  # if it exists, I create "dir_1", if already existing I create "dir_2"....
        i = 1
        path_res = path_res + "_" + str(i) + "/"
        while os.path.exists(path_res):
            i = i + 1
            path_res = os.getcwd() + "/" + results_folder + "_" + str(i) + "/"
        os.makedirs(path_res)

    return path_res
