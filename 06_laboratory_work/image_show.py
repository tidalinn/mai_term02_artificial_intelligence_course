'''image plotting module
'''

import matplotlib.pyplot as plt
import numpy as np


# Function to show the images
def image_show(img):
    plt.figure(figsize=(16, 6))

    img = img / 2 + 0.5 # unnormalize
    npimg = img.cpu().numpy()
    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()