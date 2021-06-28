import numpy as np 
import matplotlib.pyplot as plt
import cv2


def visboundaries(img,color='r',linewidth=2):


    img=(img>0).astype(np.uint8)*255
    
    
    if int(cv2.__version__[0])<4:
        contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    else:
        contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    for cnt in contours: 
      
        tmp1=np.concatenate((cnt[:,0,0],cnt[[0],0,0]))
        tmp2=np.concatenate((cnt[:,0,1],cnt[[0],0,1]))
    
        plt.plot(tmp1,tmp2,color=color,linewidth=linewidth)