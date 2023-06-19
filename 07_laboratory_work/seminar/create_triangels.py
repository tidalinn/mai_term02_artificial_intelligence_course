from numpy import *
from matplotlib.pyplot import *
import cv2 as cv

import warp
# open image to warp
fromim = array(cv.imread('data/sunset_tree.jpg'))
x,y = meshgrid(range(5),range(6))
x = (fromim.shape[1]/4) * x.flatten()
y = (fromim.shape[0]/5) * y.flatten()
# triangulate
#print(x.shape, y.shape)
#print(meshgrid(range(5),range(6)))
tri = warp.triangulate_points(stack((x, y), axis=1)).simplices
# open image and destination points
im = array(cv.imread('data/turningtorso1.jpg'))
tp = loadtxt('data/turningtorso1_points.txt') # destination points
# convert points to hom. coordinates
fp = vstack((y,x,ones((1,len(x)))))
tp = vstack((tp[:,1],tp[:,0],ones((1,len(tp)))))
# warp triangles
im = warp.pw_affine(fromim,im,fp,tp,tri)
# plot
figure()
imshow(im)
warp.plot_mesh(tp[1],tp[0],tri)
axis('off')
show()