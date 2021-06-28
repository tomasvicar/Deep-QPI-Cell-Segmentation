function color_ind_img=colorize_notouchingsamecolor(BW,conn)


alowed_num_of_colors=8;



L=bwlabel(BW,conn);
N=max(L(:));


neigbours=repmat({0},[1,N]);

for k=1:N
    
    cell=L==k;
    
    cell_dilate=imdilate(cell,strel('disk',30));%%%%%%%%%%%how far can be same color....
    
    
    tmp=unique(L(cell_dilate));
    tmp(tmp==0)=[];
    tmp(tmp==k)=[];
    neigbours{k}=tmp;
    

    
end




numcolors = Inf;
all_is_not_done=1;
rounds=0;
for qqq=1:500
    all_is_not_done=0;
rounds=rounds+1;
    
% [~,I] = sort(cellfun(@length, neigbours));
I=randperm(N);

colors=zeros(1,N);
numcolors=1;
kk=1;
for k = I
    
    kk=kk+1;
    
    
    idx = neigbours{k}; % Get neighbors of the kth node

    neighborcolors = unique(colors(idx)); % Get colors used by neighbors
    neighborcolors(neighborcolors==0) =[];
    
    % Assign the smallest color value not used by the neighboring nodes

    thiscolor =setdiff(1:alowed_num_of_colors,neighborcolors);
    if length(thiscolor) ==0
        all_is_not_done=1;
        thiscolor=1:alowed_num_of_colors;
    end
    
    thiscolor =thiscolor(randi(length(thiscolor)));
    
    
%     thiscolor = min(setdiff(1:numcolors,neighborcolors));
%     if isempty(thiscolor)
%         numcolors = numcolors + 1;
%         thiscolor = numcolors;
%     end
    
    
    
    
    
    
    
    % If there isn't one, add another color to the map
    colors(k) = thiscolor;
    
    
end


if ~all_is_not_done
    break
    
end

end







color_ind_img=zeros(size(L),'uint8');

for k=1:N
    color_ind_img(L==k)=colors(k);
    
end


