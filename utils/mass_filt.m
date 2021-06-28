function b=mass_filt(b,I,tresh,conn)

L=bwlabel(b>0,conn);
p = regionprops('table',L,I,'MeanIntensity','Area','PixelIdxList');

% mass=cat(1,p.MeanIntensity).*cat(1,p.MeanIntensity);

tmp = strcmp(p.Properties.VariableNames,'Area');
if sum(double(tmp))>0
    p=p(find((p.Area.*p.MeanIntensity)>tresh),:);
else
    p=[];
end

b=false(size(b));
% p=p{:,{'PixelList'}};

% p=cellfun()
if ~isempty(p)
    p=cat(1,p.PixelIdxList{:});
    b(p)=1;
end




end