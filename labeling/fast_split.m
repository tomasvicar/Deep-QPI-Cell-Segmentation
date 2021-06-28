clc;clear all;close all force;

img = imread('00158_PNT1A_img.tif');

mask = imread('00158_PNT1A_mask.png');


fg = ones(size(img));


img1 = imimposemin(-img,mask>0);

mask_new=watershed(img1)>0;


imshow(img,[])
hold on
visboundaries(mask_new)

imwrite(mask_new,'tmp.png')