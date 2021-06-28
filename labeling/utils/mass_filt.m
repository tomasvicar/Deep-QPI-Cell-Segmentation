function b=mass_filt(b,I,tresh,conn)

L=bwlabel(b>0,conn);
p = regionprops('table',L,I,'MeanIntensity','Area','PixelIdxList');

% mass=cat(1,p.MeanIntensity).*cat(1,p.MeanIntensity);

p=p(find((p.Area.*p.MeanIntensity)>tresh),:);

b=false(size(b));
% p=p{:,{'PixelList'}};

% p=cellfun()
p=cat(1,p.PixelIdxList{:});

b(p)=1;


end