clc;clear all;close all;
addpath('utils')


data_path='../../data_split';
save_path='../../data';

mkdir([save_path '/PNT1A'])
mkdir([save_path '/PC3'])

names=subdir([data_path '/*.txt']);

names={names(:).name};

names_new={};
for k = 1:length(names)
    name=names{k};
    if ~contains(name,'-moc-spatny-')
        names_new=[names_new,name ];
    end

end
names=names_new;

Ns=[];

names_new={};
for name =names
    name=name{1};
    fid = fopen(name,'r');
    tline = fgetl(fid);
    k=1;
    while ischar(tline)
        
        if k==1
            who_labeled=tline;
        end
        
        if k==4
            tmp=replace(tline,'quality/difficulty: ','');
            qua=str2num(tmp(1));
            
            difi=str2num(tmp(3));
        end
        
        k=k+1;
        tline = fgetl(fid);
    end
           
    if (strcmp(who_labeled,'Jirka')||strcmp(who_labeled,'Vicar')||strcmp(who_labeled,'Jakubicek')) && (qua==3)
        drawnow
    else
        names_new=[names_new,name ];
    end
    
%     if strcmp(who_labeled,'Kolar')
%         who_labeled
%         qua
%         difi
%     end
%     if strcmp(who_labeled,'Kolar')&& ((qua==3)||(difi==3))
%         name_data=[ name(1:end-29) '.tif'];
%         name_mask=replace(name,'.txt','.png');
%         
%         I=imread(name_data);
%         mask=imread(name_mask);
%         
%         hold off
%         imshow(I,[])
%         hold on
%         visboundaries(mask>0)
%         drawnow;
%     end
    
    fclose(fid);    
end     
names=names_new;



who_lab={};
times=[];

pc3_num=0;
ptn_num=0;

for img_num = 1:length(names)
    name=names{img_num};

    fid = fopen(name,'r');
    tline = fgetl(fid);
    k=1;
    while ischar(tline)
        
        if k==1
            who_labeled=tline;
        end
        
        if k==5
            time=tline;
            time=replace(time,'quduration: ','');
            time=str2num(time);
        end
        
        if k==4
            tmp=replace(tline,'quality/difficulty: ','');
            qua=str2num(tmp(1));
            
            difi=str2num(tmp(3));
        end
        
        k=k+1;
        tline = fgetl(fid);
    end
    fclose(fid); 
    
    name_data=[ name(1:end-29) '.tif'];
    name_mask=replace(name,'.txt','.png');
    
    
    who_lab=[who_lab,who_labeled];
    times=[times,time];
    

    
    if contains(name,'\pc3')
    
        pc3_num=pc3_num+1;
        
        if pc3_num ==222
            drawnow;
        end

        name_data_save=[save_path '/PC3/' num2str(pc3_num,'%05.f') '_PC3_img.tif'];
        name_mask_save=[save_path '/PC3/' num2str(pc3_num,'%05.f') '_PC3_mask.png'];
    
    elseif contains(name,'\pnt')
        
        ptn_num=ptn_num+1;
        
        name_data_save=[save_path '/PNT1A/' num2str(ptn_num,'%05.f') '_PNT1A_img.tif'];
        name_mask_save=[save_path '/PNT1A/' num2str(ptn_num,'%05.f') '_PNT1A_mask.png'];
        
    else
        error('dfsdfsdf')
    end
    
    copyfile(name_data,name_data_save)
%     copyfile(name_mask,name_mask_save)
    mask=imread(name_mask);
%     u=unique(mask);
%     u=u(u>0);
%     mask_new=zeros(size(mask));
%     cumul=0;
%     for k = u
%         l=bwlabel(mask,8);
%         l=l+cumul;
%         mask_new=mask_new+l;
%         cumul=max(mask_new(:));
%     end
%     N1=cumul;
    mask = bwareaopen(mask>0,20,8);

    l=bwlabel(mask>0,8);
    
    imwrite(uint8(l),name_mask_save)
    
%     Ns=[Ns,max(l(:))];
    
%     N2=max(l(:));
%     N1
%     N2
%     disp('-------')
%     if N1~=N2
%     end
%     
%     for k =1:max(l(:))
%         cell=l==k;
%         cell2=imerode(cell,ones(5));
%         cell2=bwareaopen(cell2,50,8);
%         ll=bwlabel(cell2,8);
%         if max(ll(:))>1
%             l(l==0)=-10;
%             hold off
%             imshow(l,[])
%             cents=regionprops(l,'Centroid');
%             cents=cat(1,cents(:).Centroid);
%             hold on
%             plot(cents(:,1),cents(:,2),'b*');
%             visboundaries(cell);
%             drawnow;
%         end
% 
%     end
    
    
    
end

boxplot(times,who_lab)


