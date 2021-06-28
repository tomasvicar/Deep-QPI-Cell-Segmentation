function seg=segm_frst_eval(II,GTT,init_r,max_r,t_frst,kr,alpha,min_hole,T_bg,area_filt)

segmentation=zeros(size(GTT));
for k=1:size(II,3)


    segm=segm_frst(II(:,:,k),init_r,max_r,t_frst,kr,alpha,min_hole,T_bg,area_filt);
    
    segmentation(:,:,k)=segm;
    
end
[seg]=-seg_final_segmentation(GTT,segmentation);




end