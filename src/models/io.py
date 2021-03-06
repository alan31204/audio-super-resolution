import os
import numpy as np
import h5py
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from scipy.signal import decimate

from matplotlib import pyplot as plt

# ----------------------------------------------------------------------------
# from Kuleshov's work

def load_h5(h5_path):
	# load training data
	with h5py.File(h5_path, 'r') as hf:
		print ('List of arrays in input file:', hf.keys())
		X = np.array(hf.get('data'))
		Y = np.array(hf.get('label'))
		print ('Shape of X:', X.shape)
		print ('Shape of Y:', Y.shape)

	return X, Y

def upsample_wav(wav, model):
	# load signal
	sr = 16000
	x_hr, fs = librosa.load(wav, sr)

	# downscale signal
	# x_lr = np.array(x_hr[0::args.r])
	x_lr = decimate(x_hr, 4)
	# x_lr = decimate(x_hr, args.r, ftype='fir', zero_phase=True)
	# x_lr = downsample_bt(x_hr, args.r)

	# upscale the low-res version
	# P = model.predict(x_lr.reshape((1,len(x_lr),1)))
	# x_pr = P.flatten()
	
	temp_lr = x_lr.reshape((1,1,len(x_lr)))
	temp_lr = torch.from_numpy(temp_lr.copy())
	temp_lr = temp_lr.float()
	temp_lr = Variable(temp_lr.cuda(), requires_grad=False).permute(0,1,2)
	P = model(temp_lr)
	# x_pr = P.flatten()
	x_pr = P.cpu().data.numpy()
	x_pr = x_pr.flatten()


	# crop so that it works with scaling ratio
	x_hr = x_hr[:len(x_pr)]
	x_lr = x_lr[:len(x_pr)]

	# save the file
	out_label = "singlespeaker-out"
	outname = wav + '.' + out_label
	rand = 4
	librosa.output.write_wav(outname + '.hr.wav', x_hr, fs)	
	librosa.output.write_wav(outname + '.lr.wav', x_lr, int(fs / rand))	
	librosa.output.write_wav(outname + '.pr.wav', x_pr, fs)	

	# save the spectrum
	S = get_spectrum(x_pr, n_fft=2048)
	# print("PR val: %.4f" % S)
	save_spectrum(S, outfile=outname + '.pr.png')
	S = get_spectrum(x_hr, n_fft=2048)
	# print("HR val: %.4f" % S)
	save_spectrum(S, outfile=outname + '.hr.png')
	S = get_spectrum(x_lr, n_fft=2048/2)
	# print("LR val: %.4f" % S)
	save_spectrum(S, outfile=outname + '.lr.png')

# ----------------------------------------------------------------------------

def get_spectrum(x, n_fft=2048):
	S = librosa.stft(x, int(n_fft))
	p = np.angle(S)
	S = np.log1p(np.abs(S))
	return S

def save_spectrum(S, lim=800, outfile='spectrogram.png'):
	plt.imshow(S.T, aspect=10)
	# plt.xlim([0,lim])
	plt.tight_layout()
	plt.savefig(outfile)