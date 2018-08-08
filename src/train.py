import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
# from models.model import default_opt
from models import *
from models.io import load_h5, upsample_wav
from data.vctk.loader import loading
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from math import log10
import librosa
# random.seed(123)


# TODO list for training the model
# From Kuleshov's run.py, PyTorch super-resolution main.py
# Train the network, finish the evaluation, finish the plotting, Think about the network
# Understand how Kuleshov prepare data, and think about network problem, and how its upsample_wav works


# parsing
parser = argparse.ArgumentParser(description='Audio Super Resolution')
# subparsers = parser.add_subparsers(title='Commands')

# train

# train_parser = subparsers.add_parser('train')
# parser.set_defaults(func=train)

# parser.add_argument('--model_name', type=str, default='',
					# help='model name')
# parser.add_argument('--train', required=False, 
# 	help='path to h5 archive of training patches')
# parser.add_argument('--val', required=False,
# 	help='path to h5 archive of validation set patches')
parser.add_argument('-e', '--epochs', type=int, default=100,
	help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128,
	help='training batch size')
parser.add_argument('--res_block', type=int, default=20,
	help='number of res_block')
parser.add_argument('--feats', type=int, default=64,
	help='number of feat')
parser.add_argument('--kernel_size', type=int, default=13,
	help='kernel size')
# parser.add_argument('--logname', default='tmp-run',
# 	help='folder where logs will be stored')
parser.add_argument('--layers', default=4, type=int,
	help='number of layers in each of the D and U halves of the network')
# parser.add_argument('--wav-file-list', default="../data/vctk/speaker1/speaker1-val-files.txt",
# 	help='list of audio files for evaluation')
parser.add_argument('--path',
	help='path to previous training epoch')
# parser.add_argument('--alg', default='adam',
# 	help='optimization algorithm')
# parser.add_argument('--lr', default=1e-3, type=float,
# 	help='learning rate')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#	 help='momentum')

# eval

# eval_parser = subparsers.add_parser('eval')
# eval_parser.set_defaults(func=eval)

# eval_parser.add_argument('--logname', required=Trues,
# 	help='path to training checkpoint')
# eval_parser.add_argument('--out-label', default='',
# 	help='append label to output samples')
# eval_parser.add_argument('--wav-file-list', 
# 	help='list of audio files for evaluation')
# eval_parser.add_argument('--r', help='upscaling factor', type=int)
# eval_parser.add_argument('--sr', help='high-res sampling rate', 
# 	type=int, default=16000)

args = parser.parse_args()
print(args)


# model_name = args.model_name

# model building
if args.path: 
	model = torch.load(args.path)
	model.cuda()
	loss_function = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
else:
	model = AudioSRNet(args)
	model.cuda()
	loss_function = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-3)




# training process 
def train(args):
	# get data
	root_dir = '../data/vctk/vctk-speaker1-train.4.16000.8192.4096.h5'
	# val_dir = '../data/vctk/vctk-speaker1-val.4.16000.8192.4096.h5'
	dataset1 = loading(root_dir, transform=None)
	# valset1 = loading(val_dir, transform=None)

	dataset = DataLoader(dataset1, batch_size=64, shuffle=True, num_workers=4)
	# valset = DataLoader(valset1, batch_size=4, shuffle=True, num_workers=4)
	#dataset = dataset1
	# valset = valset1
	nb_batch = dataset.__len__()
	epoch_l = []
	iter_num = []
 	# start training process
	i = 0
	for epoch in range(args.epochs):
		epoch_loss = 0
		n = 0
		start = time.time()
		for i_batch, data in enumerate(dataset):
			X_train, Y_train = data['lr'], data['hr']
		#for batch in range(nb_batch):
		# for batch in enumerate(dataset, 1):
			X_train = X_train.float()
			Y_train = Y_train.float()
			X_train = Variable(X_train.cuda(), requires_grad=False).permute(0, 2, 1)
			Y_train = Variable(Y_train.cuda(), requires_grad=False).permute(0, 2, 1)
			model.zero_grad()
			optimizer.zero_grad()
			loss = loss_function((model(X_train)), Y_train) # not sure yet
			#epoch_loss += loss.item()
			epoch_loss += loss.cpu().data.numpy()
			loss.backward()
			optimizer.step()
			n = n + 1

		i = i + 1
		end = time.time()
		epoch_l.append(epoch_loss/n)
		iter_num.append(i)
		# print(i)
		print("== Epoch {%s}   Loss: {%.4f}  Running time: {%4f}" % (str(epoch), (epoch_loss) / n, end - start))
		checkpoint(epoch) # store checkpoint

	fig = plt.figure()
	plt.plot(iter_num, epoch_l)
	plt.xlabel('number iteration')
	plt.ylabel('Loss')
	plt.savefig('epoch/loss.png')


def eval(args):
	# load model
	# model = get_model(args, 0, args.r, from_ckpt=True, train=False)
 # 	model.load(args.logname) # from default checkpoint
	num = 49
	model = torch.load('epochs/' + "model_epoch_"+str(num)+".pth")
	avg_psnr = 0
	avg_snr = 0
	sum_x = 0
	sum_y = 0
	val_dir = '../data/vctk/vctk-speaker1-val.4.16000.8192.4096.h5'
	file_list = '../data/vctk/speaker1/speaker1-val-files.txt'
 	# X_val, Y_val = load_h5(args.val)
 	# dataset = loading(root_dir, transform=None)
	valset1 = loading(val_dir, transform=None)
	valset = DataLoader(valset1, batch_size=1, shuffle=False, num_workers=4)
	nb_batch = valset.__len__()
	with torch.no_grad():
		for i_batch, val in enumerate(valset):
		# for batch in range(nb_batch):
			# input, target = batch[0].to(device), batch[1].to(device)
			X_val, Y_val = val['lr'], val['hr']
			# print(X_val.numpy()[0].shape)
			# x_temp = X_val.numpy()[0]
			# y_temp = Y_val.numpy()[0]
			# x_S = computeSNR(x_temp,2048)
			# y_S = computeSNR(y_temp,2048)
			# sum_x += x_S
			# sum_y += y_S
			X_val = X_val.float()
			Y_val = Y_val.float()
			X_val = Variable(X_val.cuda(), requires_grad=False).permute(0, 2, 1) # compute N, C L 
			Y_val = Variable(Y_val.cuda(), requires_grad=False).permute(0, 2, 1)
			# print(X_val.size())
			# print(Y_val.size())
			prediction = model(X_val)
			mse = loss_function(prediction, Y_val)
			psnr = 10 * log10(1 / mse.item())
			snr = 10 * log10(1 / mse.cpu().data.numpy())
			avg_psnr += psnr
			avg_snr += snr

		print("===> Avg. SNR: {:.4f} dB".format(avg_snr / len(valset)))
		# print("===> X. SNR: {:.4f} dB".format(sum_x / len(valset)))
		# print("===> Y. SNR: {:.4f} dB".format(sum_y / len(valset)))


	with open(file_list) as f:
		for line in f:
			try:
				print(line.strip())
				upsample_wav(line.strip(), model)
			except EOFError:
				print('WARNING: Error reading file:', line.strip())



# model

# Checkpoint for storing training info from superresolution/main.py by Soumith and Alykhan
def checkpoint(epoch):
	model_out_path = "model_epoch_{}.pth".format(epoch)
	# model_out_path = 'model/' + model_name + ".pth"
	torch.save(model, 'epoch/' + model_out_path)
	print("Checkpoint saved to {}".format(model_out_path))

def computeSNR(x, n_fft=2048):
	snr = librosa.stft(x, n_fft)
	p = np.angle(snr)
	snr = np.log1p(np.abs(snr))
	return snr


# train(args)
eval(args)


# make and create model for training and evaluating
# def get_model(args, num_classes, train=True):
  	# if train:
   #  	opt_params = { 'alg' : args.alg, 'lr' : args.lr, 'b1' : 0.9, 'b2' : 0.999,
   #				 'batch_size': args.batch_size, 'layers': args.layers }
  	# SGD optimizer

  	# model = models.AudioNet(num_classes=num_classes, r=r, 
	#							 opt_params=opt_params, log_prefix=args.logname)
  	# return model



# def main():
#   	args.func(args)

# if __name__ == '__main__':
#   	main()
