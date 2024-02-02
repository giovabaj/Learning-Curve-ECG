"""In this version I switch the loops:
- I fix a certain maximum sample size (e.g. 10000)
- I
- 
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from architectures import LstmNetwork
from arg_parser import get_path_results
from datasets import EcgDataset
from loss import WBCELoss
from train import model_predictions
from train import train

# Settings
path_data = 'data/data_ecg.npy'
path_labels = 'data/labels_ecg.npy'
lr = 1e-2
patience = 300
n_epochs = 1000
n_reps = 100
val_frac = 0.2
augment = True
std_augm = 0.0002
n_samples = 1000
batch_size = 32
weight_decay = 0.05
p_dropout = 0.4
random_seed = 45

config = {
    "lr": lr,
    "patience": patience,
    "n_epochs": n_epochs,
    "n_reps": n_reps,
    "random_seed": random_seed,
    "batch_size": batch_size,
    "size": n_samples,
    "val_frac": val_frac,
    "augment": augment,
    "std_augm": std_augm,
    "weight_decay": weight_decay,
    "p_dropout": p_dropout
}

# Setting the computation device and seeds
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# import signals and labels
labels = np.load(path_labels)
data = np.load(path_data)

# get random sample of the required size
folds_file = 'folds' + str(n_samples) + '.json'

with open(folds_file) as file:
    folds = json.load(file)

ids = folds["ids"]
folds.pop('ids', None)
data_sample = data[ids, :]
labels_sample = labels[ids]

# sizes considered
sizes = [500, 550, 600, 650, 700, 750, 800]

# All indices
all_ids = np.arange(len(ids))

start_total = time.time()
# Loop over sizes
for i_size, size in enumerate(sizes):
    print(size)

    # npy matrix with AUC values
    auc = np.zeros(n_reps)

    # set results folder's path
    path_res = get_path_results(parent_results_folder="results/splitting_method_3/" + "sample_" + str(n_samples) + "/",
                                n=size)

    with open(path_res + "/config.json", 'w') as fp:
        json.dump(config, fp)

    for i_rep in range(n_reps):  # loop over folds
        if i_rep % 10 == 0:
            print(i_rep)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(data_sample,
                                                            labels_sample,
                                                            train_size=size,
                                                            stratify=labels_sample,
                                                            )

        X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                          y_train,
                                                          test_size=val_frac,
                                                          stratify=y_train)

        train_dataset = EcgDataset(X_train, y_train, augment=augment, std_augm=std_augm)
        val_dataset = EcgDataset(X_val, y_val)
        test_dataset = EcgDataset(X_test, y_test)

        # print(X_train[0, :5])

        start = time.time()
        # Create result folder for actual cv fold
        path_results_fold = path_res + "/CVfold_" + str(i_rep) + "/"
        if not os.path.exists(path_results_fold):
            os.makedirs(path_results_fold)

        # Save prints to a log file
        old_stdout = sys.stdout
        log_file = open(path_results_fold + "logfile.log", "w")
        sys.stdout = log_file

        # Print the total size to record it
        print("Total size in the experiment:", n_samples)

        print("# train samples: ", len(train_dataset), " , AF frac: ", sum(train_dataset.labels) / len(train_dataset))
        print("# test samples: ", len(test_dataset), " , AF frac: ", sum(test_dataset.labels) / len(test_dataset))
        print("# val samples: ", len(val_dataset), " , AF frac: ", sum(val_dataset.labels) / len(val_dataset))

        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)

        # -- Instantiate model, loss function and optimizer ---
        in_channels = [1, 96, 48, 32]
        out_channels = [96, 48, 32, 24]
        dilation_rate = [1, 4, 4, 6]
        kernel_size = [8, 6, 6, 4]
        max_pooling = [0, 0, 0, 16]
        stride = [1, 1, 1, 1, 1]
        ratio = 1.25
        num_layers = 1
        bidirectional = False
        many_to_many = False
        in_len = 1280
        wp = 0.75  # Best .75

        model = LstmNetwork(in_channels=in_channels,
                            out_channels=out_channels,
                            dilation=dilation_rate,
                            kernel_size=kernel_size,
                            max_pooling=max_pooling,
                            stride=stride,
                            ratio=ratio,
                            num_layers=num_layers,
                            many_to_many=many_to_many,
                            bidirectional=bidirectional,
                            in_len=in_len,
                            p_dropout=p_dropout)

        loss = WBCELoss(w_p=wp, w_n=1 - wp)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_sched = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=0)

        # -- TRAIN --
        model = train(model, train_loader, val_loader, loss,
                      optimizer, device, patience=patience,
                      path_res=path_results_fold, num_epochs=n_epochs,
                      scheduler=lr_sched)

        # -- TEST --
        predictions, labels_test = model_predictions(model, device, test_loader)
        # save results and compute metrics
        np.save(path_results_fold + "predictions_test", predictions)
        np.save(path_results_fold + "labels_test", labels_test)

        auc[i_rep] = roc_auc_score(labels_test, predictions)

        # writing logs to the log file
        print("Elapsed time: ", time.time() - start)
        sys.stdout = old_stdout
        log_file.close()

    np.save(path_res + "auc_test", auc)

print("total elapsed time:", time.time() - start_total)
