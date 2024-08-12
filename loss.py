import torch
from torch import nn

class AdaptedCrossEntropy(nn.Module):
    def __init__(self, bin_size=1., min_rating=1.):
        super().__init__()
        self.bin_size = bin_size
        self.min_rating = min_rating

    def forward(self, y_true, y_pred):
        eps = 1e-7 # default value in keras
        y_idx = (y_true - self.min_rating) / self.bin_size
        y_idx = y_idx.type(torch.int64)
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        cross_entropy = - torch.log(y_pred.gather(1, y_idx.unsqueeze(-1))) #TODO:
        return cross_entropy.sum()
