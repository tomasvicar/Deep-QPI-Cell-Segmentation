clc;clear all;close all;
addpath('utils')


data_path='..\..\data_relabeld_raw';
save_path='..\..\data_relabel';

mkdir([save_path '/PNT1A'])
mkdir([save_path '/PC3'])



names=subdir([data_path '/*.tif']);

names={names(:).name};


pc3_num=0;
ptn_num=0;




for img_num = 1:length(names)
    name=names{img_num};
    
    name_data_save=replace(name,data_path,save_path);
    name_mask_save=replace(name_data_save,'_img.tif','_mask.png');
    
    name_data = name;
    name_mask = subdir([replace(name_data,'_img.tif','') '*.png']);
    name_mask = name_mask(1).name;
    
    copyfile(name_data,name_data_save)
    
    mask=imread(name_mask);
    
    mask = bwareaopen(mask>0,10,8);

    l=bwlabel(mask>0,8);
    
    if max(l(:))>255
       error('moc bunek') 
    end
    
    imwrite(uint8(l),name_mask_save)
end






