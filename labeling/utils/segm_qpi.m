function [segm,nucs]=segm_qpi(I,init_T,max_T,min_size,min_hole,T_bg,h)

bin=I>init_T;
bin =  bwareaopen(bin,min_size);
bin =  ~bwareaopen(~bin,min_hole);





L=uint16(bwlabel(bin));
max_L=max(L(:));
region_nums=1:max_L;
for T=init_T:0.03:max_T
    
    bin=I>T;   
    bin =  bwareaopen(bin,min_size);
    s = regionprops(bin,'Centroid');
    centroids = round(cat(1,s(:).Centroid));
    sz=size(I);
    centroids_bin=zeros(size(I));
    try
        centroids_bin(sub2ind(sz,centroids(:,2),centroids(:,1)))=1;
    end
    LL=bwlabel(bin);
    
    s = regionprops(L,centroids_bin,'MeanIntensity','Area');
    cents_num = cat(1,s(:).MeanIntensity).*cat(1,s(:).Area);
    cents_num_tmp=zeros(size(cents_num));
    cents_num_tmp(region_nums)=cents_num(region_nums)>1;
    region_nums_tmp=find(cents_num_tmp)';
    
    for region_num=region_nums_tmp
        u=unique(LL(L==region_num));
        u(u==0)=[];
        if length(u)>1
            region_nums(region_nums==region_num)=[];

            for k=1:length(u)
                max_L=max_L+1;
                L(LL==u(k))=max_L;
                region_nums=[region_nums,max_L];
            end
        end
    end
end

nucs=false(size(I));
for k=region_nums
    nucs(L==k)=1;
end
D=bwdist(~nucs);
D=imhmin(-D,h);
w=watershed(D);
nucs_tmp=nucs&w>0;
removed=nucs_tmp-bwareaopen(nucs_tmp,min_size);
removed=imdilate(removed,ones(3)).*nucs;
nucs=nucs_tmp|removed;

w = imimposemin(-I, nucs);
w=watershed(w)>0;
bin=I>T_bg;
bin =  bwareaopen(bin,min_size);
bin =  ~bwareaopen(~bin,min_hole);

r=3;
[x,y]=meshgrid(-r:r,-r:r);
ss=sqrt(x.^2+y.^2);
ss=ss<=r;
bin=imopen(bin,ss);

segm=w.*bin;
