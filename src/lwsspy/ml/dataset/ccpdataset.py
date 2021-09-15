"""
This file shows how to get from a CCP labeled volume to a Pytorch datta set

"""
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

from lwsspy.ml import dataset


class CCPDataset(Dataset):

    def __init__(self, filename, sq_size: int = 33):

        if sq_size % 2 == 0:
            raise ValueError("The square image size must be odd.")

        # Load NPZ file
        vardict = np.load(filename, allow_pickle=True)

        # Assign variables
        self.x = vardict["x"]
        self.y = vardict["y"]
        self.z = vardict["z"]
        self.V = vardict["V"]

        # Labeled
        self.labeled = dict()
        self.labeled["lx"] = vardict["lx"]
        self.labeled["ly"] = vardict["ly"]
        self.labeled["lz"] = vardict["lz"]
        self.labeled["lV"] = vardict["lV"]

        # Label dictionary
        self.labeldict = vardict["labeldict"].item()

        # Padding value
        self.pad = int((sq_size - 1)/2)
        self.padval = 0
        self.padmask = self.labeldict['none']

        # Number of labeled slices in each dimension l? is a boolean array
        self.nlx = np.sum(self.labeled['lx'])
        self.nly = np.sum(self.labeled['ly'])
        self.nlz = np.sum(self.labeled['lz'])

        # Total number of items is
        # n labeled slice in one direction
        #   X dimension of the other two dimensions
        self.nx = self.nlx * self.y.size * self.z.size
        self.ny = self.nly * self.x.size * self.z.size
        self.nz = self.nlz * self.x.size * self.y.size
        self.NT = self.nx + self.ny + self.nz

        # # Pad the arrays so it's easy to grab he masked values.
        # self.V = np.pad(
        #     V, self.padval, mode='constant',
        #     constant_values=self.labeldict['none'])
        # self.labeled['lV'] = np.pad(
        #     self.labeled['lV'], mode='constant',
        #     constant_values=self.labeldict['none'])

        """
        Here I have to figure out mathematically, how many images of 33x33 
        I can extract:
        1. I have to calculate how many labeled slices I have in each direction 
           using the self.labeled[{lx,ly,lz}] variables
        2. Then for each direction figure out how many data points I'd have?
        3. Take every pixel? What to do with pixels on the edges? 'none'
           them?
        """

    def __getitem__(self, index):

        if index == self.NT:
            raise ValueError('Limit reached')

        if index < self.nx and self.nx != 0:
            dim = (self.nlx, len(self.y), len(self.z))

            # index of the image center in the subvolume
            idx, idy, idz = np.unravel_index(index, dim)

            # Position of slice
            isl = np.where(self.labeled['lx'])[0][idx]

            # Pad the slice
            slc = np.pad(
                self.V[isl, :, :], self.pad, mode='constant',
                constant_values=self.labeldict['none'])

            # Get image form padded slice and the label from the labeled volume
            # print([idy, idy + 2 * self.pad + 1])
            # print([idz, idz + 2 * self.pad + 1])

            image = slc[
                idy: idy + 2 * self.pad + 1,
                idz: idz + 2 * self.pad + 1]
            label = self.labeled['lV'][isl, idy, idz]

        elif index >= self.nx and self.ny != 0:
            dim = (len(self.x), self.nly, len(self.z))

            # Correct the index to accoutn for the first dimension
            index -= int(self.nx)

            # index of the image center in the subvolume
            idx, idy, idz = np.unravel_index(index, dim)

            # Position of slice
            isl = np.where(self.labeled['ly'])[0][idy]

            # Pad the slice
            slc = np.pad(
                self.V[:, isl, :], self.pad, mode='constant',
                constant_values=self.labeldict['none'])

            # print([idx, idx + 2 * self.pad + 1])
            # print([idz, idz + 2 * self.pad + 1])

            # Get image form padded slice and the label from the labeled volume
            image = slc[
                idx: idx + 2 * self.pad + 1,
                idz: idz + 2 * self.pad + 1]
            label = self.labeled['lV'][idx, isl, idz]

            # print(slc.shape)

        elif index >= self.nx + self.ny:
            dim = (len(self.x), len(self.y), self.nlz)

            # Correct the index to accoutn for the first dimension
            index -= int((self.nx + self.ny))

            # index of the image center in the subvolume
            idx, idy, idz = np.unravel_index(index, dim)

            # Position of slice
            isl = np.where(self.labeled['lz'])[0][idz]

            # Pad the slice
            slc = np.pad(
                self.V[:, :, isl], self.pad, mode='constant',
                constant_values=self.labeldict['none'])

            # Get image form padded slice and the label from the labeled volume
            image = slc[
                idx: idx + 2 * self.pad + 1,
                idy: idy + 2 * self.pad + 1]
            label = self.labeled['lV'][idx, idy, isl]

        return torch.from_numpy(image).reshape((1, *image.shape)).float(), torch.tensor(label)

    def __len__(self):
        return self.NT

    def plot_samples(self):

        labels_map = {v: k for (k, v) in self.labeldict.items()}

        print("Starting loop to go through dataset")
        print("Press:")
        print("    Enter - for the next set of images")
        print("    Ctrl + C, then Enter - to end the loop")

        while True:
            figure = plt.figure(figsize=(8, 8))
            cols, rows = 3, 3

            for i in range(1, cols * rows + 1):

                label = self.labeldict['none']

                while label == self.labeldict['none']:

                    sample_idx = torch.randint(len(self), size=(1,)).item()
                    img, label = self[sample_idx]

                figure.add_subplot(rows, cols, i)
                plt.title(labels_map[int(label)])
                plt.axis("off")
                plt.imshow(img.squeeze().numpy().T,
                           cmap="rainbow", aspect='auto')

            plt.show(block=False)

            input("Press Enter to continue...")
            plt.close()