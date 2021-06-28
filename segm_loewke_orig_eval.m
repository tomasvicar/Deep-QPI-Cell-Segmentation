function seg=segm_loewke_orig_eval(II,GTT,min_mass,min_hole,T_bg)

segmentation=zeros(size(GTT));
for k=1:size(II,3)


    segm=segm_loewke_orig(II(:,:,k),min_mass,min_hole,T_bg);
    
    segmentation(:,:,k)=segm;
    
end
[seg]=-seg_final_segmentation(GTT,segmentation);




end