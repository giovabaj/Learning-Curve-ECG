import torch
from torch.utils.data import Dataset


class EcgDataset(Dataset):
    """Dataset definition for the binary classification problem"""

    def __init__(self, ecg_data, labels, augment=False, std_augm=.00002):
        super().__init__()
        self.augment = augment
        self.std_augm = std_augm

        # load labels and ECGs
        self.ecgs = torch.tensor(ecg_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        """Total number of samples"""
        return len(self.labels)

    def __getitem__(self, index):
        """Generates one sample of data+label"""
        ecg_signal = self.ecgs[index]
        if self.augment:
            ecg_signal = ecg_signal + (self.std_augm ** 0.5) * torch.randn(ecg_signal.shape)
        label = self.labels[index]
        return ecg_signal, label
