function lines_data = boundarymask_custom(L)

lines_data=zeros(size(L),'uint16');

N=max(L(:));

for k=1:N
    
    cell=L==k;
    
    cell_dilate=imdilate(cell,strel('disk',1));
%     lines_data=imerode(lines_data,[1,1,1;1,1,1;1,1,1]);
    
    lines_tmp=cell_dilate;
    lines_tmp(cell)=0;

    lines_data(lines_tmp)=1;
    
    L(cell_dilate)=0;
    
end


end


