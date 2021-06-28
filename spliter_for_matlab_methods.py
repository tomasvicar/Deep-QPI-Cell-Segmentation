from config import Config
from split_train_test import split_train_test


for it in range(5):
    
    
    config = Config()
    
    config.data_train_valid_test_path = '../for_standard_methods/data_train_valid_test' + str(it)
    
    
            
    split_train_test(seed=it*100,config=config)
    
    
    
    
    
    
    

















