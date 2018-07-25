import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from models import AudioNet
from models.io import load_h5, upsample_wav
from data.vctk.loader import loading
from torch.utils.data import Dataset, DataLoader
import time
# random.seed(123)


# TODO list for training the model
# From Kuleshov's run.py, PyTorch super-resolution main.py

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
# parser.add_argument('--logname', default='tmp-run',
# 	help='folder where logs will be stored')
parser.add_argument('--layers', default=4, type=int,
	help='number of layers in each of the D and U halves of the network')
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
model = AudioNet(num_classes=1000)
model.cuda()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)




# training process 
def train(args):
	# get data
	root_dir = '../data/vctk/vctk-speaker1-train.4.16000.8192.4096.h5'
	val_dir = '../data/vctk/vctk-speaker1-val.4.16000.8192.4096.h5'
	dataset1 = loading(root_dir, transform=None)
	valset1 = loading(val_dir, transform=None)

	# dataset = DataLoader(dataset1, batch_size=4, shuffle=True, num_workers=4)
	# valset = DataLoader(valset1, batch_size=4, shuffle=True, num_workers=4)
	dataset = dataset1
	valset = valset1
	nb_batch = dataset.__len__()
	epoch_l = []
 	# start training process
	for epoch in range(args.epochs):
		epoch_loss = 0
		n = 0
		start = time.time()
		for batch in range(nb_batch):
		# for batch in enumerate(dataset, 1):
			X_train = dataset.data[batch]
			Y_train = dataset.label[batch]
			X_train = torch.from_numpy(X_train)
			Y_train = torch.from_numpy(Y_train)
			X_train = Variable(X_train.cuda(), requires_grad=False)
			Y_train = Variable(Y_train.cuda(), requires_grad=False)
			X_train = X_train.transpose(0,1)
			Y_train = Y_train.transpose(0,1)
			print("X_train size: " , X_train.size())
			print("X_train dimension: ", X_train[1])
			print("Y_train size: " , Y_train.size())
			print("Y_train dimension: ", Y_train[1])
			model.zero_grad()
			optimizer.zero_grad()
			loss = loss_function((model(X_train)), Y_train) # not sure yet
			epoch_loss += loss.item()
			# epoch_loss += loss.cpu().data.numpy()
			loss.backward()
			optimizer.step()
			n = n + 1
		
		end = time.time()
		epoch_l.append(epoch_loss)
		print("== Epoch {%s}   Loss: {%.4f}  Running time: {%4f}" % (str(epoch), (epoch_loss) / n, end - start))
		checkpoint(epoch) # store checkpoint




def eval(args):
	# load model
	# model = get_model(args, 0, args.r, from_ckpt=True, train=False)
 # 	model.load(args.logname) # from default checkpoint
	num = 5
	model = torch.load('epoch/' + "model_epoch_"+num+".pth")
	avg_psnr = 0
	val_dir = '../data/vctk/vctk-speaker1-val.4.16000.8192.4096.h5'
 	# X_val, Y_val = load_h5(args.val)
 	# dataset = loading(root_dir, transform=None)
	valset1 = loading(val_dir, transform=None)
	valset = valset1
	# valset = DataLoader(valset1, batch_size=4, shuffle=True, num_workers=4)
	# valset = DataLoader(valset1, batch_size=4, shuffle=True, num_workers=4)
	nb_batch = valset.__len__()
	with torch.no_grad():
		for batch in range(nb_batch):
			# input, target = batch[0].to(device), batch[1].to(device)
			X_val = valset.data(batch)
			Y_val = valset.label(batch)
			X_val = torch.from_numpy(X_val)
			Y_val = torch.from_numpy(Y_val)
			X_val = Variable(X_val.cuda(), requires_grad=False)
			Y_val = Variable(Y_val.cuda(), requires_grad=False)
			X_val = X_val.transpose(0,1)
			Y_val = Y_val.transpose(0,1)
			prediction = model(X_val)
			mse = loss_function(prediction, Y_val)
			psnr = 10 * log10(1 / mse.item())
			avg_psnr += psnr
		print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(valset)))


	with open(args.wav_file_list) as f:
		for line in f:
			try:
				print(line.strip())
				upsample_wav(line.strip(), args, model)
			except EOFError:
				print('WARNING: Error reading file:', line.strip())



# model

# Checkpoint for storing training info from superresolution/main.py by Soumith and Alykhan
def checkpoint(epoch):
	model_out_path = "model_epoch_{}.pth".format(epoch)
	# model_out_path = 'model/' + model_name + ".pth"
	torch.save(model, 'epoch/' + model_out_path)
	print("Checkpoint saved to {}".format(model_out_path))



train(args)
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
