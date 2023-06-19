'''network module
'''

import torch.nn as nn
import torch.nn.functional as F


# Define a convolution neural network
class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*10*10, 10)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = F.relu(output) 

        output = self.conv2(output)     
        output = self.bn2(output)
        output = F.relu(output)     
        output = self.pool(output)  

        output = self.conv4(output)    
        output = self.bn4(output)                  
        output = F.relu(output) 

        output = self.conv5(output)   
        output = self.bn5(output) 
        output = F.relu(output)   
          
        output = output.view(-1, 24*10*10)
        output = self.fc1(output)

        return output