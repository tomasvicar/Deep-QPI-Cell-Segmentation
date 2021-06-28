function [segm]=qpi_iterative_segmenation_egt(I,mass_threshold,foreground_threshold_2,foreground_threshold_1,max_LIT_threshold,hole_area_threshold)
%based on Loewke - automted cell segmentation for quntitative phase
%microscopy  and combined with EGT segmentation


% foreground_threshold_2 - foreground treshhold (aditional to EGT)
% foreground_threshold_1 -  foreground treshhold (aditional to EGT)
%BW = (egt and I>foreground_threshold_2)|I>foreground_threshold_1
%max_LIT_threshold - dividing lines above this treshold will not be created
%hole_area_threshold - smaler holes will be removed





a=I;
a=medfilt2(a,[3 3]);
a=imgaussfilt(a,0.2);






tmp = EGT_Segmentation(a);


b=(tmp.*(a>foreground_threshold_2))|a>foreground_threshold_1;

b=~bwareafilt(~b,[hole_area_threshold,Inf]);

b=mass_filt(b,a,mass_threshold);


% a=imerode(a,strel('sphere',1));
% a=imopen(a,strel('sphere',3));

res=b;
res_old=true(size(b));
while sum(sum(res_old==res))~=numel(res)
res_old=res;

b=res;

res=false(size(b));

l=bwlabel(b);

ml=max(l(:));
for k=1:ml
    region=k==l;
    
    grey_blob=a.*region;
    bb=regionprops(region,'BoundingBox');
    bb=bb(1).BoundingBox;
    bb=floor(bb);
    bb(bb==0)=1;

    if bb(2)+bb(4)>size(l,1)
        bb(4)=bb(4)-(bb(2)+bb(4)-size(l,1));
    end
    if bb(1)+bb(3)>size(l,2)
        bb(3)=bb(3)-(bb(1)+bb(3)-size(l,2));
    end
    
    grey_blob=grey_blob(bb(2):bb(2)+bb(4),bb(1):bb(1)+bb(3));
    
    bw=grey_blob>0;
    for t=foreground_threshold_2:0.05:max_LIT_threshold
        bw_blob=grey_blob>t;
        

        ll=bwlabel(bw_blob);
        mll=max(ll(:));
        
        for kk=1:mll
            cc=ll==kk;
            
            volume=sum(sum(grey_blob.*cc));
            if volume<mass_threshold
                bw_blob(cc)=0;

            end
        end
        tmp=bwlabel(bw_blob);
        num_css=max(tmp(:));
        if num_css>1
            break;
        end
    end
    
    tmp=bwlabel(bw_blob);
    num_blobs=max(tmp(:));
    if num_blobs>1
        dists=bwdist(bw_blob);
        basins=watershed(dists);
        cut_lines=basins==0;
        bw=bw&~cut_lines;
    end
    tmp=false(size(res));
    tmp(bb(2):bb(2)+bb(4),bb(1):bb(1)+bb(3))=bw;
    res(tmp)=1;
end

segm=res;


end



