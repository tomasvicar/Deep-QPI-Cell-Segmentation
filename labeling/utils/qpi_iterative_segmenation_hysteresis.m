function segm=qpi_iterative_segmenation_hysteresis(I,mass_threshold,tLo,tHi,max_LIT_threshold,hole_area_threshold,conn)
%based on Loewke - automted cell segmentation for quntitative phase





a=I;
a=medfilt2(a,[3 3]);
a=imgaussfilt(a,0.2);

r=3;
[x,y]=meshgrid(-r:r,-r:r);
ss=sqrt(x.^2+y.^2);
ss=ss<=r;

b=hysteresis_thresh(a,tLo,tHi);

% b=imopen(b,ss);


b=~bwareafilt(~b,[hole_area_threshold,Inf],conn);

b=mass_filt(b,a,mass_threshold,conn);




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
        for t=tLo:0.05:max_LIT_threshold
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
            dists=bwdistgeodesic(bw,bw_blob);

            dists(isnan(dists))=Inf;

            basins=watershed(dists,conn);
            cut_lines=basins==0;
            bw=bw&~cut_lines;
        end
        tmp=false(size(res));
        tmp(bb(2):bb(2)+bb(4),bb(1):bb(1)+bb(3))=bw;
        res(tmp)=1;
    end







end

res=imopen(res,ss);

res=mass_filt(res,a,mass_threshold,conn);
res=~bwareafilt(~res,[hole_area_threshold,Inf],conn);



segm=res;
