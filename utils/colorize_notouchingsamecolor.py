import numpy as np
from skimage.measure import label
from skimage.morphology import disk,dilation
from scipy.ndimage.morphology import binary_dilation

def colorize_notouchingsamecolor(BW,conn=1,alowed_num_of_colors=8,min_dist=15):
    
    
    L=label(BW,connectivity=conn)
    N=np.max(L)
    
    neigbours=[]
    for k in range(N):
        k=k+1
        
        cell=L==k
        
        cell_dilate=binary_dilation(cell,disk(min_dist)>0)
        
        tmp=np.unique(L[cell_dilate])
        tmp=tmp[(tmp!=0)&(tmp!=k)]
        
        neigbours.append(tmp-1)
        
        
    numcolors = np.inf
    all_is_not_done=1
    rounds=0
       
    
    maxxx=500
    for qqq in range(maxxx):
    
        all_is_not_done=0
        rounds=rounds+1
        
        I=np.random.permutation(N)
        
        
        colors=np.zeros(N);
        
        numcolors=1
        for k in I:
            
            idx = neigbours[k]
            
            neighborcolors = np.unique(colors[idx])
            
            # neighborcolors=neighborcolors[neighborcolors!=0]
            
            thiscolor = list(set(list(np.arange(alowed_num_of_colors))) - set(neighborcolors))
             
            if len(thiscolor) ==0 :
                all_is_not_done=1
                thiscolor=list(np.arange(alowed_num_of_colors))
            
            
            thiscolor = thiscolor[np.random.randint(len(thiscolor))]
            # thiscolor = thiscolor[0]
            colors[k] = thiscolor
            
            
        if ~all_is_not_done:
            break
        
        
        if qqq==(maxxx-1):
            raise NameError('colors not found')
            
            
    

    color_ind_img=np.zeros(L.shape,'uint8')
    
    for k in range(N):
        color_ind_img[L==k+1]=colors[k]+1
          
    return color_ind_img
            