clc;clear all;close all force;
addpath('utils')


for it = 0:4

    files = subdir(['../for_standard_methods/data_train_valid_test' num2str(it) '/valid/*.tif']);
    files = {files.name};

    for k=1:length(files)

        name = files{k};

        II(:,:,k)=imread(name);
        GTT(:,:,k)=imread(replace(name,'_img.tif','_mask.png'));
    end



    init_r_range=[2,35];
    max_r_range=[8,90];
    t_frst_range=[0,0.7];
    kr_range=[0.5,30];
    min_mass_range=[7,300];
    alpha_range=[0.01,10];
    area_filt_range = [5,400];
    T_bg_range=[-0.2 0.25];
    
    min_hole=60;
%     T_bg=0.05;



    init_r = optimizableVariable('init_r',init_r_range);
    max_r = optimizableVariable('max_r',max_r_range);
    t_frst = optimizableVariable('t_frst',t_frst_range);
    kr = optimizableVariable('kr',kr_range);
    alpha = optimizableVariable('alpha',alpha_range);
    area_filt = optimizableVariable('area_filt',area_filt_range,'Type','integer');
    T_bg = optimizableVariable('T_bg',T_bg_range);
    vars = [init_r,max_r,t_frst,kr,alpha,area_filt,T_bg];

    fun = @(x) segm_frst_eval(II,GTT,x.init_r,x.max_r,x.t_frst,x.kr,x.alpha,min_hole,x.T_bg,x.area_filt);
    results = bayesopt(fun,vars,'NumSeedPoints',48,'MaxObjectiveEvaluations',350,'UseParallel',true);


    files = subdir(['../for_standard_methods/data_train_valid_test' num2str(it) '/test/*.tif']);
    files = {files.name};

    x = results.XAtMinObjective;

    for  k=1:length(files)

        name = files{k};

        I=imread(name);
        GT=imread(replace(name,'_img.tif','_mask.png'));

        [fPath, fName, fExt] = fileparts(name);

        segm=segm_frst(I,x.init_r,x.max_r,x.t_frst,x.kr,x.alpha,min_hole,x.T_bg,x.area_filt);

        mkdir(['../for_standard_methods/frst_res' num2str(it)])
        
        save_name=['../for_standard_methods/frst_res' num2str(it) '/' fName '.png'];

        imwrite(segm,save_name)

    end


end