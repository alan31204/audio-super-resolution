import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
# import copy


# TODO: set up the audio net model
# learn PyTorch
# read Wavenet paper, and understand its structures


class AudioNet(nn.Module):
# Generic PyTorch model training code
	# def __init__(self, from_ckpt=False, n_dim=None, r=2, opt_params=default_opt):

	def __init__(self, block, layers):
	# 	super(AudioNet, self).__init__()
	# 	self.in_channels = 16
		

    # perform the usual initialization
    # self.r = r
    # Model.__init__(self, from_ckpt=from_ckpt, n_dim=n_dim, r=r,
    #                opt_params=opt_params, log_prefix=log_prefix)



    def create_model(self, n_dim, r):






    def predict(self, X):	