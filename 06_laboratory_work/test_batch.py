'''batch testing module
'''

import numpy as np
import torch
import torchvision
from torch.autograd import Variable

from image_show import image_show
from cifar10_downloader import test_loader
from variables import batch_size, classes, model, device


# Function to test the model with a batch of images and show the labels predictions
def test_batch():

    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loader))

    # get the inputs
    images = Variable(images.to(device))
    labels = Variable(labels.to(device))

    # show all images as one image grid
    image_show(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))