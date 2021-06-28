import numpy as np


def overlap_for_ap(gt_L,res_L):
    
    gt_L=gt_L.copy()
    res_L=res_L.copy()
    
    fp = 0
    fn = 0
    
    fp_cells=res_L.copy()
        
    jacards=[]
    N_gt=np.max(gt_L)
    for gt_cell_num_min1 in range(N_gt):
        
        gt_cell_num=gt_cell_num_min1+1
        
        jacard=0
        
        u=np.unique(res_L[gt_L==gt_cell_num])
        
        u=u[u>0]
        
        for res_cell_num in u:
            
            cell1=gt_L==gt_cell_num
            cell2=res_L==res_cell_num
            jacard_tmp=np.sum(cell1 & cell2)/np.sum(cell1 | cell2)
            
            if jacard_tmp>0.5:
                
                fp_cells[cell2]=0
                
                jacard=jacard_tmp
    
        
        if jacard>0.5:
            jacards.append(jacard)
        else:
            fn = fn +1
        
        
    jacards = np.array(jacards)
    u=np.unique(fp_cells)
    u=u[u>0]
    for res_cell_num in u:
        fp = fp +1
        

    return jacards,fp,fn
    
    
    
    



