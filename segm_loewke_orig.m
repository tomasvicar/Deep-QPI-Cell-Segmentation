function segm=segm_loewke_orig(I,min_mass,min_hole,T_bg)

I0=I;

fg=I0>T_bg;
fg=mass_filt(fg,I0,min_mass,4);
I=mat2gray(I0);


binary_old=zeros(size(I));
binary_new=fg;

while ~(sum(sum(binary_new==binary_old))==numel(binary_new))

    binary_old=binary_new;
    L = bwlabel(binary_new);
    ml=max(L(:));
    for k=1:ml
        region=k==L;
        grey_blob=I0.*region;

        bb=regionprops(region,'BoundingBox');
        bb=bb(1).BoundingBox;
        bb=floor(bb);
        bb(bb==0)=1;

        if bb(2)+bb(4)>size(L,1)
            bb(4)=bb(4)-(bb(2)+bb(4)-size(L,1));
        end
        if bb(1)+bb(3)>size(L,2)
            bb(3)=bb(3)-(bb(1)+bb(3)-size(L,2));
        end

        grey_blob=grey_blob(bb(2):bb(2)+bb(4),bb(1):bb(1)+bb(3));


        increment = 0.01; 
        thresh = 0;
        bw = grey_blob > 0;

        while thresh<1
            thresh = thresh + increment;
            
            bw_blob=grey_blob>thresh;
            
            CC = bwconncomp(bw_blob);

            if CC.NumObjects>1
                remove_cc_ind=[];
                volumes=[];
                for kk=1:CC.NumObjects

                    volume=sum(sum(grey_blob(CC.PixelIdxList{kk})));
                    volumes=[volumes volume];

                    if volume<min_mass
                        remove_cc_ind=[remove_cc_ind,kk];
                    end
                end
                use_cc_ind=1:CC.NumObjects;
                use_cc_ind(remove_cc_ind)=[];
                volumes=volumes(use_cc_ind);
            else
                use_cc_ind=1;
            end
            
            if length(use_cc_ind)>1
                binary_new(region)=0;
                tmp_img=zeros(size(grey_blob));
                for kk=1:length(use_cc_ind)
                    
                    tmp_img([CC.PixelIdxList{use_cc_ind(kk)}])=kk;
                    
                end
                
                D = bwdistgeodesic(bw>0,tmp_img>0);
                D(isnan(D))=Inf;
                tmp_img2 = (watershed(D)>0).*bw;
                
                binary_new(bb(2):bb(2)+bb(4),bb(1):bb(1)+bb(3))=binary_new(bb(2):bb(2)+bb(4),bb(1):bb(1)+bb(3))+tmp_img2;
                break;
            end

        end
    end
    

    
end




segm=binary_new;

segm =  ~bwareaopen(~segm,min_hole);