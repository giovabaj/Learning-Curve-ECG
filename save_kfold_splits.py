import json
import argparse
from sklearn.model_selection import StratifiedKFold
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

n_splits = 10
seed = 43

parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=2000, help="N. of samples")
parser.add_argument('--balance', default=False, action=argparse.BooleanOptionalAction,
                    help="Whether to undersample or not")

args = parser.parse_args()
balance = args.balance
n_samples = args.n_samples

# import signals and labels
path_data = 'data/data_ecg.npy'
path_labels = 'data/labels_ecg.npy'
labels = np.load(path_labels)
data = np.load(path_data)

# Random under sampling to get a balanced dataset
if balance:
    rus = RandomUnderSampler(sampling_strategy='majority', random_state=seed)
    ids, _ = rus.fit_resample(np.arange(len(labels)).reshape(-1, 1), labels)
    ids_ = np.random.choice(len(ids), n_samples, replace=False)
    ids = ids[ids_, 0]
else:
    ids = np.random.choice(len(labels), n_samples, replace=False)

# sample the dataset
labels = labels[ids]

print(labels.sum() / len(labels))

kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
folds = {'ids': ids.tolist()}
count = 0
for train, test in kfold.split(np.zeros(len(labels)), labels):
    folds[count] = {}
    folds[count]['train'] = train.tolist()
    folds[count]['test'] = test.tolist()
    count += 1

# dump folds to json
if balance:
    filename = 'folds' + str(n_samples) + '_balanced.json'
else:
    filename = 'folds' + str(n_samples) + '_seed' + str(seed) + '.json'

with open(filename, 'w') as fp:
    json.dump(folds, fp)
