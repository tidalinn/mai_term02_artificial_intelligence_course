'''training module
'''

import torch
from torch.autograd import Variable

from variables import model, optimizer, loss_fn, device
from test_accuracy import test_accuracy
from save_model import save_model
from cifar10_downloader import train_loader


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    
    best_accuracy = 0.0

    # Define your execution device
    print('The model will be running on', device, 'device\n')
    
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()

            # predict classes using images from the training set
            outputs = model(images)

            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)

            # backpropagate the loss
            loss.backward()

            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value

            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = test_accuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy), '\n')
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            save_model()
            best_accuracy = accuracy