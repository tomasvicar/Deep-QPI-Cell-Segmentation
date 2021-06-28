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



    area_filt_range = [5,400];
    T_bg_range=[-0.2 0.25];
    h_range=[0,50];
    min_object_size_range = [10,400];
    min_hole_size_range = [5,250];
    min_hole_range = [5,250];
    t_range =[-0.2 1.2];
    
    area_filt = optimizableVariable('area_filt',area_filt_range,'Type','integer');
    T_bg = optimizableVariable('T_bg',T_bg_range);
    h = optimizableVariable('h',h_range);
    t = optimizableVariable('t',t_range);
    min_object_size = optimizableVariable('min_object_size',min_object_size_range,'Type','integer');
    min_hole_size = optimizableVariable('min_hole_size',min_hole_size_range,'Type','integer');
    min_hole = optimizableVariable('min_hole',min_hole_range,'Type','integer');
    vars = [min_hole,T_bg,area_filt,min_object_size,min_hole_size,h,t];
    
    
    fun = @(x) segm_dt_eval(II,GTT,x.min_hole,x.T_bg,x.area_filt,x.min_object_size,x.min_hole_size,x.h,x.t);

    results = bayesopt(fun,vars,'NumSeedPoints',48,'MaxObjectiveEvaluations',350,'UseParallel',true);


    files = subdir(['../for_standard_methods/data_train_valid_test' num2str(it) '/test/*.tif']);
    files = {files.name};

    x = results.XAtMinObjective;

    for  k=1:length(files)

        name = files{k};

        I=imread(name);
        GT=imread(replace(name,'_img.tif','_mask.png'));

        [fPath, fName, fExt] = fileparts(name);

        segm=segm_dt(I,x.min_hole,x.T_bg,x.area_filt,x.min_object_size,x.min_hole_size,x.h,x.t);

        mkdir(['../for_standard_methods/qpi_dt_res' num2str(it)])
        
        save_name=['../for_standard_methods/qpi_dt_res' num2str(it) '/' fName '.png'];

        imwrite(segm,save_name)

    end


end