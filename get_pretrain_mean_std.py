import numpy as np
from skimage.io import imread
from glob import glob
from config import Config
import os


names = glob(Config.data_pretrain_train_valid_path + os.sep + 'train/*.tif')

means = []
stds = []

for name in names:
    
    
    img = imread(name)
    
    
    means.append(np.mean(img))
    stds.append(np.std(img))
    

print(np.mean(means))
print(np.mean(stds))