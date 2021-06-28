import numpy as np
import matplotlib.pyplot as plt

class Log:
    def __init__(self):
        
        self.train_loss_log=[]
        self.valid_loss_log=[]
        
        self.train_loss_log_tmp=[]
        self.valid_loss_log_tmp=[]
        
    def save_tmp_log(self,loss,train_or_test):
        
        if train_or_test=='train':
            self.train_loss_log_tmp.append(loss.detach().cpu().numpy())
        elif train_or_test=='valid':
            self.valid_loss_log_tmp.append(loss.detach().cpu().numpy())
        else:
            raise NameError('invalid train or test')
 

    
    def save_log_data_and_clear_tmp(self):
        
        self.train_loss_log.append(np.mean(self.train_loss_log_tmp))
        self.train_loss_log_tmp=[]
    
        self.valid_loss_log.append(np.mean(self.valid_loss_log_tmp))
        self.valid_loss_log_tmp=[]
        
    def plot_training(self,name=None):
        plt.plot(self.train_loss_log,color=[0, 0.4470, 0.7410])
        plt.plot(self.valid_loss_log,color=[0.4660, 0.6740, 0.1880])
        plt.title('loss')
        if name:
            plt.savefig(name)
        
        plt.show()
        plt.close()
        
        