import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import numpy as np
from early_stopping import EarlyStopping
from sklearn.metrics import roc_auc_score
from os.path import join


def train_single_epoch(model: nn.Module, train_loader: DataLoader, loss: nn.Module, optimizer: Optimizer, device: str):
    """
    Train model for a single epoch
    :param model: The model to be trained.
    :param train_loader: DataLoader with training data.
    :param loss: Loss function.
    :param optimizer: Optimizer for updating model parameters.
    :param device: Device to run the training on (e.g., "cpu" or "cuda").
    :return: (1) Average loss over the epoch (2) ROC-AUC score computed from the true and predicted values.
    """
    losses = []
    y_true_all = []
    y_scores_all = []
    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        y_scores = model(x)[:, 0]

        y_true_all += y.cpu().detach().numpy().tolist()
        y_scores_all += y_scores.cpu().detach().numpy().tolist()

        l = loss(y_scores, y)
        losses.append(l.item())
        l.backward()
        optimizer.step()
    return np.mean(np.array(losses)), roc_auc_score(y_true_all, y_scores_all)


def validation_single_epoch(model: nn.Module, validation_loader: DataLoader, loss: nn.Module, device: str):
    """
    Evaluates the model for a single epoch on the validation dataset.
    :param model: The model to be evaluated.
    :param validation_loader: DataLoader for the validation data.
    :param loss: Loss function.
    :param device: Device to run the evaluation on (e.g., "cpu" or "cuda").
    :return: (1) Average loss of the validation set (2) ROC-AUC score computed from the true and predicted values.
    """
    losses = []
    model.eval()
    y_true = []
    y_score = []
    for x, y in validation_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_score_temp = model(x)[:, 0]
            l = loss(y_score_temp, y)
            y_temp = y.to('cpu').numpy().tolist()
            y_score_temp = y_score_temp.to('cpu').numpy().tolist()
            y_true += y_temp
            y_score += y_score_temp
        losses.append(l.item())
    return np.mean(np.array(losses)), roc_auc_score(np.array(y_true), np.array(y_score))


def train_model(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, loss: nn.Module,
                optimizer: Optimizer, device: str, scheduler: LRScheduler = None, num_epochs: int = 50,
                path_res: str = 'train_out/', patience: int = 5):
    """
    Trains a model over multiple epochs with early stopping and optional learning rate scheduling.
    :param model: The model to be trained.
    :param train_loader: DataLoader for the training data.
    :param validation_loader: DataLoader for the validation data.
    :param loss: Loss function to optimize.
    :param optimizer: Optimizer for updating model parameters.
    :param device: Device to run the training on (e.g., "cpu" or "cuda").
    :param scheduler: Learning rate scheduler. Defaults to None.
    :param num_epochs: Maximum number of training epochs. Defaults to 50.
    :param path_res: Path to save training results and the best model. Defaults to 'train_out/'.
    :param patience: Number of epochs to wait for improvement before early stopping. Defaults to 5.
    :return: The trained model with the best validation performance (nn.Module).
    """
    model.to(device)
    # initialize arrays
    train_losses = np.ones(num_epochs) * np.inf
    validation_losses = np.ones(num_epochs) * np.inf
    train_auc = np.ones(num_epochs) * np.inf
    validation_auc = np.ones(num_epochs) * np.inf

    path_checkpoints = join(path_res, 'checkpoint.pt')
    early_stopper = EarlyStopping(patience=patience, verbose=True, path=path_checkpoints)
    for i in range(0, num_epochs):
        model.train()
        train_losses[i], train_auc[i] = train_single_epoch(model, train_loader, loss, optimizer, device)
        validation_losses[i], validation_auc[i] = validation_single_epoch(model, validation_loader, loss, device)
        print(f"Epoch {i + 1} completed. Training auc: {train_auc[i]}, Validation auc: {validation_auc[i]}")
        early_stopper(-validation_auc[i], model)
        if validation_auc[i] > 0.9 and early_stopper.patience > 50:
            print("Patience set to 50")
            early_stopper.patience = 50
        if early_stopper.early_stop:
            print("Early stopping")
            break
        if scheduler is not None:
            scheduler.step()
            print(optimizer.param_groups[0]["lr"])

    model.load_state_dict(torch.load(path_checkpoints))
    torch.save(model.state_dict(), join(path_res, "best_model.pt"))
    np.save(join(path_res, "train_losses"), train_losses)
    np.save(join(path_res, "val_losses"), validation_losses)
    np.save(join(path_res, "train_auc"), train_auc)
    np.save(join(path_res, "val_auc"), validation_auc)
    return model


def model_predictions(model: nn.Module, device: str, dataloader: DataLoader):
    """
    Makes predictions from a trained model.
    :param model: torch model to make predictions
    :param device: device to be used
    :param dataloader: torch DataLoader
    :return: predictions and true labels
    """
    model.eval()  # evaluation mode
    model.to(device)  # putting model on device
    y_pred = []  # list with predictions of each batch
    y_true = []
    with torch.no_grad():
        for ecg, y in dataloader:
            ecg = ecg.to(device)
            y_pred.append(model(ecg))
            y_true.append(y)

    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    # y_pred = softmax(y_pred, dim=1)
    return y_pred.cpu().numpy(), y_true.cpu().numpy()
