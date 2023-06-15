'''onnx conversion module
'''

import torch
import torch.onnx 
from torch.autograd import Variable

from variables import input_size, model


#Function to Convert to ONNX 
def convert_to_onnx(): 

    # Define your execution device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, *input_size, requires_grad=True)

    dummy_input = Variable(dummy_input.to(device))

    # Export the model   
    torch.onnx.export(
        model,                                 # model being run 
        dummy_input,                           # model input (or a tuple for multiple inputs) 
        'ImageClassifier.onnx',                # where to save the model  
        export_params=True,                    # store the trained parameter weights inside the model file 
        opset_version=10,                      # the ONNX version to export the model to 
        do_constant_folding=True,              # whether to execute constant folding for optimization 
        input_names = ['modelInput'],          # the model's input names 
        output_names = ['modelOutput'],        # the model's output names 
        dynamic_axes={                         # variable length axes 
            'modelInput' : {0 : 'batch_size'},
            'modelOutput' : {0 : 'batch_size'}
        })
    
    print(' ') 
    print('\nModel has been converted to ONNX')