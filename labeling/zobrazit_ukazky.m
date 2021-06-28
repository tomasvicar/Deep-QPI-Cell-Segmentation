clc;clear all;close all force;
addpath('utils')


folder='ukazka';

files_png=subdir([folder filesep '*.png']);


files_png={files_png(:).name};

for img_num=1:length(files_png)
    
    
    
    mask_name=files_png{img_num};
    
    img_name=[mask_name(1:end-29) '.tif'];
    
    img=imread(img_name);
    
    
    colormap_cells=[1 0 0;0 1 0;0 0 1;0.8314 0.8314 0.0588;1 0 1;1,0.5,0;0.00,1.00,1.00;0.45,0.00,0.08];
    
    contour_line_width=0.1;
    
    color_ind_img=imread(mask_name)/25;
    
    figure()
    imshow([img,img],[-0.3,1.5])
    hold on
    
    N=size(colormap_cells,1);
    
    
    for k=1:N
        visboundaries(color_ind_img==k,'Color',colormap_cells(k,:),'EnhanceVisibility',0,'LineWidth',contour_line_width);
    end
    
end