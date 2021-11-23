import torch
from torch.utils.data import Dataset


class CustomSubset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the 
                           same length as indices
    """

    def __init__(self, dataset, indices, labels):
        self.dataset = dataset
        self.indices = indices
        # ( some number not present in the #labels just to make sure
        labels_hold = torch.ones(len(indices)).type(torch.long) * 300
        labels_hold[:] = labels
        self.targets = labels_hold

    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]][0]
        label = self.targets[idx]
        return (item, label)

    def __getlabel__(self, idx):

        label = self.labels[idx]
        return label

    def __len__(self):
        return len(self.indices)
