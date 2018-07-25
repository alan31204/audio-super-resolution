from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py
import time


# data loading from PyTorch tutorial

class loading(Dataset):
    # filename = 'vctk-speaker1-train.4.16000.8192.4096.h5'
    # root_dir = 'vctk-speaker1-train.4.16000.8192.4096.h5'

    def __init__(self, root_dir, transform=None):
        
        f = h5py.File(root_dir, 'r')

	# List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
        b_group_key = list(f.keys())[1]

        # Get the data
        data = list(f[a_group_key])
        label = list(f[b_group_key])

	# self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #img_name = os.path.join(self.root_dir,
         #                       self.landmarks_frame.iloc[idx, 0])
        #image = io.imread(img_name)
        #landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'lr': self.data[idx], 'hr': self.label[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

'''
transformed_dataset = loading(root_dir = 'vctk-speaker1-train.4.16000.8192.4096.h5',transform=None)
#loading(transform=transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()]))

# print(len(transformed_dataset))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

   # print(i, sample['lr'].size(), sample['hr'].size())

    if i == 2:
        break

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
'''

# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['lr'], sample_batched['hr']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')


'''
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['lr'].size(),
          sample_batched['hr'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        #plt.figure()
        #show_landmarks_batch(sample_batched)
        #plt.axis('off')
        #plt.ioff()
        #plt.show()
        break

    try:
        FileNotFoundError
    except NameError:
        FileNotFoundError = IOError
'''
