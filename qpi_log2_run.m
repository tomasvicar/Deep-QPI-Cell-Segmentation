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



    sigmas_range=[2,15];
    lambda_range=[1,10];
    min_mass_range=[30,120];
    h_range=[0,5];

    min_hole=60;
    T_bg=0.05;


    sigmas = optimizableVariable('sigmas',sigmas_range);
    lambda = optimizableVariable('lambda',lambda_range);
    min_mass = optimizableVariable('min_mass',min_mass_range);
    h = optimizableVariable('h',h_range);
    vars = [sigmas,lambda,min_mass,h];
    fun = @(x) segm_qpi_log2_eval(II,GTT,x.sigmas,x.lambda,x.min_mass,min_hole,T_bg,x.h);

    results = bayesopt(fun,vars,'NumSeedPoints',10,'MaxObjectiveEvaluations',200,'UseParallel',true);


    files = subdir(['../for_standard_methods/data_train_valid_test' num2str(it) '/test/*.tif']);
    files = {files.name};

    x = results.XAtMinObjective;

    for  k=1:length(files)

        name = files{k};

        I=imread(name);
        GT=imread(replace(name,'_img.tif','_mask.png'));

        [fPath, fName, fExt] = fileparts(name);

        segm=segm_qpi_log2(I,x.sigmas,x.lambda,x.min_mass,min_hole,T_bg,x.h);

        mkdir(['../for_standard_methods/qpi_log2_res' num2str(it)])
        
        save_name=['../for_standard_methods/qpi_log2_res' num2str(it) '/' fName '.png'];

        imwrite(segm,save_name)

    end


end