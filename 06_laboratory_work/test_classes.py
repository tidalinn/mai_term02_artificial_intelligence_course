'''classes testing module
'''

import torch
from torch.autograd import Variable

from cifar10_downloader import test_loader
from variables import number_of_labels, batch_size, classes, model


# Function to test what classes performed well
def test_classes():

    # Define your execution device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_correct = list(0. for i in range(number_of_labels))
    class_total = list(0. for i in range(number_of_labels))

    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(number_of_labels):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))