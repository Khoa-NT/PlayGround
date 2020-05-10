import torch
from torch.backends import cudnn
from torch.utils import data

# from my_classes import Dataset


def get_path_files(fdir: "path to folder" = "", ftype: "file type" = "*", do_sort = False ) -> "Paths[file_path]":
    """ Glob all file in fdir and return a sorted list"""
    list_files_path = [str(x) for x in Path(fdir).glob(ftype)]
    if do_sort:
        list_files_path = natsorted(list_files_path)
    return list_files_path

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs):
        'Initialization'
#         self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = ID
#         y = self.labels[ID]

        return X

# # CUDA for PyTorch
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
# cudnn.benchmark = True

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

# Datasets
partition = get_path_files("data")
# labels = # Labels

# Generators
training_set = Dataset(partition)
training_generator = data.DataLoader(training_set, **params)


# Loop over epochs
for epoch in range(max_epochs):
    # Training
    for local_batch in training_generator:
        # Transfer to GPU
        print(local_batch)
        print(len(local_batch))
        break
    break