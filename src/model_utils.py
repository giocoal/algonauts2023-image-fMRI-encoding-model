import torch
from torchmetrics import PearsonCorrCoef

# Define the loss function
def PearsonMeanLoss(pred, target):
    # Create the PearsonCorrCoef object for the loss function
    pearson = PearsonCorrCoef(num_outputs=pred.shape[1])
    # Calculate the Pearson correlation coefficient between the prediction and target
    # where pred and target tensors are of shape (batch_size, num_variable)
    # Squared the coefficients and take the mean
    return torch.mean(torch.pow(pearson(pred, target),2))

def PearsonMedianLoss(pred, target):
    # Create the PearsonCorrCoef object for the loss function
    pearson = PearsonCorrCoef(num_outputs=pred.shape[1])
    # Calculate the Pearson correlation coefficient between the prediction and target
    # where pred and target tensors are of shape (batch_size, num_variable)
    # Squared the coefficients and take the median
    return torch.median(torch.pow(pearson(pred, target),2))