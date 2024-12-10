import torch
import numpy as np
from early_stopping import EarlyStopping
from sklearn.metrics import roc_auc_score
import os


def save_checkpoint_last_it(epoch, model, scheduler, optimizer, path, early_stopping_counter):
    if scheduler is not None:
        torch.save(scheduler.state_dict(), path + '_scheduler.pt')
    torch.save(model.state_dict(), path + '_model.pt')
    torch.save(optimizer.state_dict(), path + '_optimizer.pt')
    torch.save(epoch, path + '_epoch.pt')
    torch.save(early_stopping_counter, path + '_early_stopping.pt')


def load_checkpoint_last_it(model, scheduler, optimizer, path):
    if scheduler is not None:
        scheduler.load_state_dict(torch.load(path + '_scheduler.pt'))
    model.load_state_dict(torch.load(path + '_model.pt'))
    optimizer.load_state_dict(torch.load(path + '_optimizer.pt'))
    return int(torch.load(path + '_epoch.pt')) + 1, int(torch.load(path + '_early_stopping.pt'))


def train_single_epoch(model, train_loader, loss, optimizer, device):
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


def validation_single_epoch(model, validation_loader, loss, device):
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


def train(model, train_loader, validation_loader, loss, optimizer,
          device, scheduler=None, num_epochs=50, path_res='train_out/', patience=5):
    model.to(device)

    train_losses = np.ones(num_epochs) * np.inf
    validation_losses = np.ones(num_epochs) * np.inf
    train_auc = np.ones(num_epochs) * np.inf
    validation_auc = np.ones(num_epochs) * np.inf

    path_checkpoints = path_res + 'checkpoint.pt'
    early_stopper = EarlyStopping(patience=patience, verbose=True, path=path_checkpoints)
    path_last_it = path_res + 'last_it'
    epoch_start = 0

    if os.path.isfile(path_last_it + '_model.pt'):
        print('Load state....')
        epoch_start, early_stopper.counter = load_checkpoint_last_it(model,
                                                                     scheduler, optimizer, path_last_it)

    for i in range(epoch_start, num_epochs):
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
        save_checkpoint_last_it(i, model, scheduler, optimizer, path_last_it, early_stopper.counter)

    model.load_state_dict(torch.load(path_checkpoints))
    torch.save(model.state_dict(), path_res + "best_model.pt")
    np.save(path_res + "train_losses", train_losses)
    np.save(path_res + "val_losses", validation_losses)
    np.save(path_res + "train_auc", train_auc)
    np.save(path_res + "val_auc", validation_auc)

    return model


def model_predictions(model, device, dataloader):
    """Function to make predictions from a trained model.
    Input:
     - model: torch model to make predictions
     - device: device to be used
     - dataloader: torch DataLoader
    Output:
     - y_pred: predictions
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
