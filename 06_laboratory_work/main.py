'''main module
'''

import torch

from train import train
from test_accuracy import test_accuracy
from Network import Network
from test_batch import test_batch
from test_classes import test_classes
from convert_to_onnx import convert_to_onnx


def main():

    # Let's build our model
    train(1) # 5
    print('Finished Training')

    # Test which classes performed well
    test_accuracy()
    test_classes()
    
    # Let's load the model we just created and test the accuracy per label
    model = Network()
    
    path = 'myFirstModel.pth'
    model.load_state_dict(torch.load(path))

    # Test with batch of images
    test_batch()

    # Conversion to ONNX 
    convert_to_onnx()
    

if __name__ == '__main__':
    main()