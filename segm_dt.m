function segm=segm_dt(I,min_hole,T_bg,area_filt,min_object_size,min_hole_size,h,t)

conn=4;

I_bg=I;

% I=mat2gray(I,[-1.0169 2.9386]);

fg_mask=I_bg>T_bg;
fg_mask =  ~bwareaopen(~fg_mask,min_hole);



detection = dt(I,fg_mask,min_object_size,min_hole_size,h,t);


segm=seeded_watershed(I,detection,fg_mask);

segm =  bwareaopen(segm,area_filt,conn);
