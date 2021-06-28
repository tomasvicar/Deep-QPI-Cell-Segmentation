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



    min_mass_range=[20,200];

    min_hole_range=[10,100];
    % T_bg_range=[0.05,0.06];
    T_bg=0.05;


    min_mass = optimizableVariable('min_mass',min_mass_range);
    min_hole = optimizableVariable('min_hole',min_hole_range,'Type','integer' );
    % T_bg = optimizableVariable('T_bg',T_bg_range);
    vars = [min_mass,min_hole];

    fun = @(x) segm_loewke_orig_eval(II,GTT,x.min_mass,x.min_hole,T_bg);
    results = bayesopt(fun,vars,'NumSeedPoints',5,'MaxObjectiveEvaluations',100,'UseParallel',true);


    files = subdir(['../for_standard_methods/data_train_valid_test' num2str(it) '/test/*.tif']);
    files = {files.name};

    x = results.XAtMinObjective;

    for  k=1:length(files)

        name = files{k};

        I=imread(name);
        GT=imread(replace(name,'_img.tif','_mask.png'));

        [fPath, fName, fExt] = fileparts(name);

        segm=segm_loewke_orig(I,x.min_mass,x.min_hole,T_bg);

        mkdir(['../for_standard_methods/loewke_res' num2str(it)])
        
        save_name=['../for_standard_methods/loewke_res' num2str(it) '/' fName '.png'];

        imwrite(segm,save_name)

    end


end