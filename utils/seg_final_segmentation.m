function [seg]=seg_final_segmentation(gt_mask,segmentation)



gt_mask=gt_mask>0;
seg_all=[];
counter=0;
for kk=1:size(gt_mask,3)
    
    
    YY=segmentation(:,:,kk);
    ll=bwlabel(YY,4);
    l=bwlabel(gt_mask(:,:,kk),4);
    

    for k=1:max(l(:))
        counter=counter+1;
        b=k==l;
        bb=ll(b);
        qq=unique(bb(find(bb)));
        
        jacard = 0; 

        for q=qq'
            cell=(ll==q);
            
            jacard_tmp=sum(sum(b & cell))/sum(sum(b | cell));
            
            if 0.5<jacard_tmp
                jacard = jacard_tmp;
                ll(cell)=0;
            end
                
        end
        
        seg_all=[seg_all jacard];   
        
    end
    
    u = unique(ll);
    u=u(u>0);
    for cell_num = 1:length(u)
        seg_all=[seg_all 0];   
    end

    drawnow;
end


seg=mean(seg_all);
