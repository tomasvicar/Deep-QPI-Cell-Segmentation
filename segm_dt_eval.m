function seg=segm_dt_eval(II,GTT,min_hole,T_bg,area_filt,min_object_size,min_hole_size,h,t)

segmentation=zeros(size(GTT));
for k=1:size(II,3)


    segm=segm_dt(II(:,:,k),min_hole,T_bg,area_filt,min_object_size,min_hole_size,h,t);
    
    segmentation(:,:,k)=segm;
    
end
[seg]=-seg_final_segmentation(GTT,segmentation);




end