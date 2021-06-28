function seg=segm_qpi_log2_eval(II,GTT,lambda,sigmas,min_mass,min_hole,T_bg,h)

segmentation=zeros(size(GTT));
for k=1:size(II,3)

    segm=segm_qpi_log2(II(:,:,k),lambda,sigmas,min_mass,min_hole,T_bg,h);
    
    segmentation(:,:,k)=segm;
    
end
[seg]=-seg_final_segmentation(GTT,segmentation);


end