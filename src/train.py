import os, sys
os.sys.path.append(os.path.abspath('.'))
os.sys.path.append(os.path.dirname(os.path.abspath('.')))

import argparse
import numpy as np
import models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
# from models.model import default_opt
from models import AudioNet
from models.io import load_h5, upsample_wav

# TODO list for training the model
# From Kuleshov's run.py

# processing 
def make_parser():
	parser = argparse.ArgumentParser(description='ASR')
  	subparsers = parser.add_subparsers(title='Commands')

  	# train

	train_parser = subparsers.add_parser('train')
	train_parser.set_defaults(func=train)

	train_parser.add_argument('--train', required=True,
	    help='path to h5 archive of training patches')
	train_parser.add_argument('--val', required=True,
	    help='path to h5 archive of validation set patches')
	train_parser.add_argument('-e', '--epochs', type=int, default=100,
	    help='number of epochs to train')
	train_parser.add_argument('--batch-size', type=int, default=128,
	    help='training batch size')
	train_parser.add_argument('--logname', default='tmp-run',
	    help='folder where logs will be stored')
	train_parser.add_argument('--layers', default=4, type=int,
	    help='number of layers in each of the D and U halves of the network')
	train_parser.add_argument('--alg', default='adam',
	    help='optimization algorithm')
	train_parser.add_argument('--lr', default=1e-3, type=float,
	    help='learning rate')
	train_parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

	  # eval

	eval_parser = subparsers.add_parser('eval')
	eval_parser.set_defaults(func=eval)

	eval_parser.add_argument('--logname', required=True,
	    help='path to training checkpoint')
	eval_parser.add_argument('--out-label', default='',
	    help='append label to output samples')
	eval_parser.add_argument('--wav-file-list', 
	    help='list of audio files for evaluation')
	eval_parser.add_argument('--r', help='upscaling factor', type=int)
	eval_parser.add_argument('--sr', help='high-res sampling rate', 
	                                   type=int, default=16000)
	
	return parser


# training process 
def train(args):
	# get data
  	X_train, Y_train = load_h5(args.train)
 	X_val, Y_val = load_h5(args.val)


def eval(args):




# make and create model for training and evaluating
def get_model(args, num_classes, train=True):
	"""Create a model based on arguments"""  
  	if train:
    	opt_params = { 'alg' : args.alg, 'lr' : args.lr, 'b1' : 0.9, 'b2' : 0.999,
                   'batch_size': args.batch_size, 'layers': args.layers }
  	# SGD optimizer
  	# else: 
   #  	opt_params = default_opt

  	# create model

  	# model = models.AudioNet(num_classes=num_classes, r=r, 
    #                             opt_params=opt_params, log_prefix=args.logname)
  	return model






def main():
  parser = make_parser()

  args = parser.parse_args()
  use_cuda = torch.cuda.is_available()
  optimizer = optim.Adam(net_model.parameters(), lr=1e-3)
  args.func(args)


if __name__ == '__main__':
  main()