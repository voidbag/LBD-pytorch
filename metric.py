import torch
from torch import nn

class RollingMetric(nn.Module):
    def __init__(self, bin_size, min_rating, max_rating):
        super().__init__()
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.numerator = 0
        self.denominator = 0

    def update(self, numerator, denominator):
        self.numerator += numerator
        self.denominator += denominator

    def evaluate(self):
        return self.numerator / self.denominator

    def forward(self, y_true=None, y_pred=None):
        if y_true is None and y_pred is None:
            return self.evaluate()
        
        assert y_true is not None and y_pred is not None

        return self.do_forward(y_true, y_pred)

    def clear(self):
        self.numerator = 0
        self.denominator = 0
    
    def do_forward(self, y_true, y_pred):
        pass

class RollingRMSE(RollingMetric):
    def __init__(self, bin_size, min_rating, max_rating):
        super().__init__(bin_size, min_rating, max_rating)

    def do_forward(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, self.min_rating, self.max_rating)
        diff = (y_true - y_pred)
        sum_squared_error = (diff * diff).sum().item()
        self.update(sum_squared_error, y_true.size()[0])

    def evaluate(self):
        return (self.numerator / self.denominator) ** 0.5


class RollingMAE(RollingMetric):
    def __init__(self, bin_size, min_rating, max_rating):
        super().__init__(bin_size, min_rating, max_rating)

    def do_forward(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, self.min_rating, self.max_rating)
        sum_absolute_error = torch.abs(y_true - y_pred).sum().item()
        self.update(sum_absolute_error, y_true.size()[0])

class RollingMassAccuracy(RollingMetric):
    def __init__(self, bin_size, min_rating, max_rating):
        super().__init__(bin_size, min_rating, max_rating)
    
    def do_forward(self, y_true, y_pred):
        y_pred = y_pred.argmax(dim = 1)
        y_true = ((y_true - self.min_rating) / self.bin_size).type(torch.int64)
        success = y_pred == y_true
        self.update(success.sum().item(), y_true.size()[0])

  
class RollingAdaptedAccuracy(RollingMetric):
    def __init__(self, bin_size, min_rating, max_rating):
        super().__init__(bin_size, min_rating, max_rating)
    
    def do_forward(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, self.min_rating, self.max_rating)
        success = torch.abs((y_true - y_pred) / self.bin_size) < 0.5
        self.update(success.sum().item(), y_true.size()[0])


class RollingAdpatedCrossEntropy(RollingMetric):
    def __init__(self, bin_size, min_rating, max_rating):
        super().__init__(bin_size, min_rating, max_rating)

    def do_forward(self, y_true, y_pred):
        y_idx = (y_true - self.min_rating) / self.bin_size
        y_idx = y_idx.type(torch.int64)
        cross_entropy = - torch.log(y_pred.gather(1, y_idx.unsqueeze(-1)))
        self.update(cross_entropy.sum().item(), y_true.size()[0])


