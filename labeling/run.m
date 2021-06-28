clc;clear all;close all force;
addpath('utils')

% folder='C:\Users\Tom\Desktop\tom\pc3';
% folder='C:\Users\Tom\Desktop\tmp_bunky\klikani_ja\tom\pc3';
% folder='C:\Users\Tom\Desktop\tmp_bunky\klikani_ja\tom\pnt'; 
folder='C:\Users\Tom\Desktop\tmp_bunky\test';


kdo_to_klika='Vicar';




%vyberte si jestli chcete prochazet nazvy automaticky nebo si vybrat cisla
%obrazku - to je to druhe zakomentovane....


files=subdir([folder filesep '*.tif']);
files={files(:).name};

for img_num=1%length(files)
    
    img_name=files{img_num}

    app=segmentation_tool(img_name,kdo_to_klika);

end






% for img_num=26:31
%     
%     img_num
% 
%     img_name=[folder filesep 'img_' num2str(img_num,'%05.f') '.tif'];
% 
%     app=segmentation_tool(img_name,kdo_to_klika);
% 
% 
% 
% end









