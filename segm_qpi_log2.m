function segm=segm_qpi_log2(I,lambda,sigmas,min_mass,min_hole,T_bg,h)


I_bg=I;

conn=4;
gamma=2;

log_map=zeros([size(I) length(sigmas)]);
for k=1:length(sigmas)
    
    
    sig=sigmas(k);
    filter_size= 2*ceil(3*sig)+1;
    hn=(sig.^gamma)*fspecial('log', filter_size, sig);
    
    
    pom=conv2_spec_symetric(I,hn);
    log_map(:,:,k)=pom;
    
    
end
log_map=-min(log_map,[],3);

I=I+lambda*log_map;




bin=I>T_bg;
% bin =  bwareaopen(bin,min_size);
bin=mass_filt(bin,I_bg+T_bg,min_mass,conn);
bin =  ~bwareaopen(~bin,min_hole);



L=uint16(bwlabel(bin));
max_L=max(L(:));
region_nums=1:max_L;
for T=T_bg:0.03:max(I(:))
    
    bin=I>T;   
%     bin =  bwareaopen(bin,min_size);
    bin=mass_filt(bin,I_bg+T_bg,min_mass,conn);
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
% removed=nucs_tmp-bwareaopen(nucs_tmp,min_size);
removed=nucs_tmp-mass_filt(nucs_tmp,I_bg+T_bg,min_mass,conn);
removed=imdilate(removed,ones(3)).*nucs;
nucs=nucs_tmp|removed;

w = imimposemin(-I, nucs);
w=watershed(w)>0;
bin=I_bg>T_bg;
% bin =  bwareaopen(bin,min_size);
bin=mass_filt(bin,I_bg+T_bg,min_mass,conn);
bin =  ~bwareaopen(~bin,min_hole);
segm=w.*bin;
