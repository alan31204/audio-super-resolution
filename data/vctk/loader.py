import h5py
import torch.utils.data.Dataset

# single speaker training dataset 
filename = 'vctk-speaker1-train.4.16000.8192.4096.h5'
# single speaker validation dataset
#filename = 'vctk-speaker1-val.4.16000.8192.4096.h5'

f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]
b_group_key = list(f.keys())[1]

# Get the data
data = list(f[a_group_key])
label = list(f[b_group_key])

class loading(Dataset):
    root_dir = 'vctk-speaker1-train.4.16000.8192.4096.h5'

    def __init__(self, csv_file, root_dir, transform=None):
       """
	vctk-speaker1-train.4.16000.8192.4096.h5
	vctk-speaker1-val.4.16000.8192.4096.h5
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        "if self.transform:
            sample = self.transform(sample)"

        return sample
