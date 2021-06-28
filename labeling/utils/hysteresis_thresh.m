function final = hysteresis_thresh(image,tLo,tHi)

highmask = image>tHi;
lowmask = bwlabel(~(image<tLo));
final = ismember(lowmask,unique(lowmask(highmask)));

end

