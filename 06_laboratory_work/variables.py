'''variables module
'''

import torch.nn as nn
from torch.optim import Adam

from Network import Network


# CIFAR10 dataset consists of 50K training images. We define the batch size of 10 to load 5,000 batches of images.
batch_size = 10
number_of_labels = 10 

input_size = (3, 32, 32)

classes = (
    'plane', 
    'car', 
    'bird', 
    'cat', 
    'deer', 
    'dog', 
    'frog', 
    'horse', 
    'ship', 
    'truck'
)

# Instantiate a neural network model 
model = Network()
 
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()

optimizer = Adam(
    model.parameters(), 
    lr=0.001, 
    weight_decay=0.0001
)