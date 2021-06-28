import torch
import numpy as np
import matplotlib.pyplot as plt



model = torch.load('../res_eval/semantic_66_0.00010_gpu_6.09328_train_0.06428_valid_0.04685.pt')


model.log.plot_training()