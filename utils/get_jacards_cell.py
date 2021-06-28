import numpy as np

def get_jacards_cell(gt_L,res_L):
    gt_L=gt_L.copy()
    res_L=res_L.copy()
    
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
                
                jacard=jacard_tmp
    
        jacards.append(jacard)
    return jacards
    



def get_jacards_cell_with_fp(gt_L,res_L):
    gt_L=gt_L.copy()
    res_L=res_L.copy()
    
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
    
        jacards.append(jacard)
        
    u=np.unique(fp_cells)
    u=u[u>0]
    for res_cell_num in u:
        jacards.append(0)
    return jacards









