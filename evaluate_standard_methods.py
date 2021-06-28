import sys
import numpy as np
from glob import glob
import os
import json
from shutil import rmtree

from config import Config
from test_standard_methods import test_standard_methods



results = {'mean_jacard':[],'mean_binary_jacard':[],'aps':[],'segmenter_name':[],'method':[]}


methods = ['frst_res','loewke_res','qpi_dt_res','qpi_log2_res']


for it in range(5):
    
    for method in methods:
        
        cell_type_res = '*'
                
        names_test =  glob('D:/qpi_segmentation_tmp/for_standard_methods/data_train_valid_test' + str(it)  + os.sep + 'test' +'/**/*' + cell_type_res + '_img.tif', recursive=True)
        
        mean_jacard,mean_binary_jacard,aps = test_standard_methods(method,names_test)
        
        
        results['mean_jacard'].append(mean_jacard)
        results['mean_binary_jacard'].append(mean_binary_jacard)
        results['aps'].append(aps)
        results['method'].append(method)
        
        
        
        
with open('../result_standard_methods.json', 'w') as outfile:
    json.dump(results, outfile) 
        
        
        
    
        

    











