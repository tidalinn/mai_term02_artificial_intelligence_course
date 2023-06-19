'''model saving module
'''

import torch

from variables import model


# Function to save the model
def save_model():
    path = './myFirstModel.pth'
    torch.save(model.state_dict(), path)