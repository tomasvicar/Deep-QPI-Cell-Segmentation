clc;clear all;close all force;
addpath('utils')

input_folder = '../../data/PNT1A';
output_folder = '../../data_relabeld_raw/PNT1A';
 
kdo_to_klika='Vicar';

files = subdir([input_folder '/*.tif']);
files = {files(:).name};

files_masks={};
for k = 1:length(files)
    file = files{k};
    file = replace(file,'_img.tif','_mask.png');
    files_masks=[files_masks,file];
    
    

end


for k = 158:length(files)
    
    k
    
    file = files{k};
    file_mask = files_masks{k};
    
    
    
    app=resegmentation_tool(file,kdo_to_klika,file_mask,output_folder);
    
    
    
end






















