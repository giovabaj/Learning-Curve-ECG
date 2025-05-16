import torch


class WBCELoss(torch.nn.Module):
    """
    Weighted Binary Cross-Entropy Loss.

    Parameters:
    - w_p: weight for positive class
    - w_n: weight for negative class
    """
    def __init__(self, w_p=None, w_n=None):
        super(WBCELoss, self).__init__()
        self.w_p = w_p
        self.w_n = w_n

    def forward(self, inputs, labels, epsilon=1e-7):
        loss_pos = -1 * torch.mean(self.w_p * labels * torch.log(inputs + epsilon))
        loss_neg = -1 * torch.mean(self.w_n * (1-labels) * torch.log((1-inputs) + epsilon))
        loss = loss_pos + loss_neg
        return loss
