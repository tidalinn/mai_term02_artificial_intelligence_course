'''accuracy testing module
'''

import torch
from torch.autograd import Variable

from variables import model
from cifar10_downloader import test_loader


# Function to test the model with the test dataset and print the accuracy for the test images
def test_accuracy():

    # Define your execution device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            
            # run the model on the test set to predict labels
            # get the inputs            
            outputs = model(images)

            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)