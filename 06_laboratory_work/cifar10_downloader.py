'''downloader module
'''

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from variables import batch_size


# Loading and normalizing the data.
# Define transformations for the training and test sets
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create an instance for training. 
# When we run this code for the first time, the CIFAR10 train dataset will be downloaded locally. 
train_set = CIFAR10(
    root='./data', 
    train=True, 
    transform=transformations,
    download=True
)

# Create a loader for the training set which will read the data within batch size and put into memory.
train_loader = DataLoader(
    train_set, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=0
)

print('The number of images in a training set is: ', len(train_loader) * batch_size)

# Create an instance for testing, note that train is set to False.
# When we run this code for the first time, the CIFAR10 test dataset will be downloaded locally. 
test_set = CIFAR10(
    root='./data', 
    train=False, 
    transform=transformations, 
    download=True
)

# Create a loader for the test set which will read the data within batch size and put into memory. 
# Note that each shuffle is set to false for the test loader.
test_loader = DataLoader(
    test_set, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=0
)

print('The number of images in a test set is: ', len(test_loader) * batch_size)
print('The number of batches per epoch is: ', len(train_loader))