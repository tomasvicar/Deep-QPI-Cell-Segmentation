import numpy as np
from skimage.io import imread
from glob import glob
from config import Config
import os


# names = glob('../data/PC3/*.png')
names = glob('../data/PNT1A/*.png')

cell_count = 0

for name in names:
    
    
    img = imread(name)
    
    tmp = np.unique(img)
    
    
    cell_count =  cell_count + len(tmp) -1




    
    







