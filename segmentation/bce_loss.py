import torch

def bce_loss(y_real, y_pred):
    # TODO 
    # please don't use nn.BCELoss. write it from scratch
    loss =  (y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred))).mean()
    return loss
