import os
os.sys.path.append(os.path.abspath('.'))
os.sys.path.append(os.path.dirname(os.path.abspath('.')))

import argparse
import numpy as np
import models
from models.model import default_opt
from models.io import load_h5, upsample_wav

# TODO list for training the model

# processing 
def make_parser():


# training process 
def train(args):


# make and create model for training and evaluating
def get_model(args, n_dim, r, from_ckpt=False, train=True):



def main():
  parser = make_parser()
  args = parser.parse_args()
  args.func(args)


if __name__ == '__main__':
  main()