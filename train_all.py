import os
from os.path import join
import shutil
import sys
import time
from argparse import ArgumentParser

import yaml
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from architectures import GoodfellowNet
from datasets import EcgDataset
from loss import WBCELoss
from train_model import train_model, model_predictions


def main():
    parser = ArgumentParser()
    parser.add_argument("--config_path", help="Path to the config file", type=str)
    args = parser.parse_args()

    # Load the config file
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    # Setting the computation device and seeds
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    # import signals and labels
    labels = np.load(config["path_labels"])
    data = np.load(config["path_data"])
    # get random sample of the required size
    data_sample, _, labels_sample, _ = train_test_split(
        data,
        labels,
        train_size=config["n_samples"],
        stratify=labels,  # Ensure class proportions are maintained
        random_state=config["random_seed"]
    )

    start_total = time.time()
    for i_size, size in enumerate(config["sizes"]):  # Loop over sizes
        print(f"--- {size} ---")
        auc = np.zeros(config["n_reps"])  # initialize npy array with AUC values
        path_res = join(config["output_path"], f"sample_{config['n_samples']}", size)  # set results folder's path
        shutil.copy(args.config_path, join(path_res, "config.yaml"))  # Save config file to the results directory
        for i_rep in range(config["n_reps"]):  # loop over repetitions
            if i_rep % 10 == 0:
                print(f"\t{i_rep}")
            X_train, X_test, y_train, y_test = train_test_split(data_sample,  # Train-test split
                                                                labels_sample,
                                                                train_size=size,
                                                                stratify=labels_sample,
                                                                )
            X_train, X_val, y_train, y_val = train_test_split(X_train,  # Train-validation split
                                                              y_train,
                                                              test_size=config["val_frac"],
                                                              stratify=y_train)
            # Initialize torch datasets
            train_dataset = EcgDataset(X_train, y_train, augment=config["augment"], std_augm=config["std_augm"])
            val_dataset = EcgDataset(X_val, y_val)
            test_dataset = EcgDataset(X_test, y_test)

            start = time.time()
            path_results_fold = join(path_res, f"fold_{i_rep}")  # Create result folder for actual cv fold
            if not os.path.exists(path_results_fold):
                os.makedirs(path_results_fold)
            # Save prints to a log file
            old_stdout = sys.stdout
            log_file = open(join(path_results_fold, "logfile.log"), "w")
            sys.stdout = log_file
            # Print the total and single sets sizes
            print("Total size in the experiment:", config["n_samples"])
            print("# train samples: ", len(train_dataset), " , AF frac: ", sum(train_dataset.labels) / len(train_dataset))
            print("# test samples: ", len(test_dataset), " , AF frac: ", sum(test_dataset.labels) / len(test_dataset))
            print("# val samples: ", len(val_dataset), " , AF frac: ", sum(val_dataset.labels) / len(val_dataset))
            # Define data loaders for training and testing data in this fold
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                                                       drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)
            # Initialize model
            model = GoodfellowNet(in_channels=X_train.shape[1], len_input=X_train.shape[2], p_dropout=config["p_dropout"])
            wp = 0.5
            loss = WBCELoss(w_p=wp, w_n=1 - wp)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            lr_sched = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=0)
            # -- TRAIN --
            model = train_model(model, train_loader, val_loader, loss, optimizer, device, scheduler=lr_sched,
                                num_epochs=config["n_epochs"], path_res=path_results_fold, patience=config["patience"])
            # -- TEST --
            predictions, labels_test = model_predictions(model, device, test_loader)
            # save results and compute metrics
            np.save(join(path_results_fold, "predictions_test"), predictions)
            np.save(join(path_results_fold, "labels_test"), labels_test)
            auc[i_rep] = roc_auc_score(labels_test, predictions)
            print("Elapsed time: ", time.time() - start)
            sys.stdout = old_stdout  # writing logs to the log file
            log_file.close()
        # Save test AUC values for all itarations
        np.save(join(path_res, "auc_test"), auc)

    print("total elapsed time:", time.time() - start_total)


if __name__ == '__main__':
    main()
