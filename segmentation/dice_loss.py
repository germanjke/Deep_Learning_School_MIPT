import torch

def dice_loss(y_real, y_pred):
    
    smooth = 1.
    
    iflat = y_pred.contiguous().view(-1)
    tflat = y_real.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
    
    
    #num = 2 * torch.sum(y_pred * y_real)
    #den =  torch.sum(y_pred + y_real)
    #res = 1 - (num +1) / (den+1)
    #return res 
