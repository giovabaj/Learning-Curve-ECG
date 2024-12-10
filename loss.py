import torch
from torch.nn import functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, gamma=2, alpha=.8):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets, smooth=1):

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE

        return focal_loss


class WBCELoss(torch.nn.Module):
    def __init__(self, w_p=None, w_n=None):
        super(WBCELoss, self).__init__()
        self.w_p = w_p
        self.w_n = w_n

    def forward(self, inputs, labels, epsilon = 1e-7):
        loss_pos = -1 * torch.mean(self.w_p * labels * torch.log(inputs + epsilon))
        loss_neg = -1 * torch.mean(self.w_n * (1-labels) * torch.log((1-inputs) + epsilon))
        loss = loss_pos + loss_neg
        return loss

