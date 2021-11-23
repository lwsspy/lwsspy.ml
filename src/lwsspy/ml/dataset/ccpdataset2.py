"""
This file shows how to get from a CCP labeled volume to a Pytorch datta set

"""
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm
# from ...plot import plot_label


class CCPDataset(Dataset):

    def __init__(self, filename, sq_size: int = 33):
        super().__init__()
        if sq_size % 2 == 0:
            raise ValueError("The square image size must be odd.")

        # Load NPZ file
        vardict = np.load(filename, allow_pickle=True)

        # Assign variables
        self.x = vardict["x"]
        self.y = vardict["y"]
        self.z = vardict["z"]
        self.V = vardict["V"]
        self.V = self.V - np.min(self.V)
        self.V = self.V/np.max(self.V)

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

        # Subvolume dimennsions Slice dimensions
        self.dimx = (self.nlx, len(self.y), len(self.z))
        self.dimy = (len(self.x), self.nly, len(self.z))
        self.dimz = (len(self.x), len(self.y), self.nlz)

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
        self.__set_img_cmap_norm__()
        self.__set_boundary_cmap_norm__()

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
                constant_values=0)

            # Get image form padded slice and the label from the labeled volume
            # print([idy, idy + 2 * self.pad + 1])
            # print([idz, idz + 2 * self.pad + 1])

            image = slc[
                idy: idy + 2 * self.pad + 1,
                idz: idz + 2 * self.pad + 1].T
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
                constant_values=0)

            # print([idx, idx + 2 * self.pad + 1])
            # print([idz, idz + 2 * self.pad + 1])

            # Get image form padded slice and the label from the labeled volume
            image = slc[
                idx: idx + 2 * self.pad + 1,
                idz: idz + 2 * self.pad + 1].T
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
                constant_values=0)

            # Get image form padded slice and the label from the labeled volume
            image = slc[
                idx: idx + 2 * self.pad + 1,
                idy: idy + 2 * self.pad + 1]
            label = self.labeled['lV'][idx, idy, isl]

        return torch.from_numpy(image).reshape((1, *image.shape)).float(), torch.tensor(label)

    def __get_label__(self, index):

        if index == self.NT:
            raise ValueError('Limit reached')

        if index < self.nx and self.nx != 0:

            # index of the image center in the subvolume
            idx, idy, idz = np.unravel_index(index, self.dimx)

            # Position of slice
            isl = np.where(self.labeled['lx'])[0][idx]

            label = self.labeled['lV'][isl, idy, idz]

        elif index >= self.nx and self.ny != 0:

            # Correct the index to accoutn for the first dimension
            index -= int(self.nx)

            # index of the image center in the subvolume
            idx, idy, idz = np.unravel_index(index, self.dimx)

            # Position of slice
            isl = np.where(self.labeled['ly'])[0][idy]

            label = self.labeled['lV'][idx, isl, idz]

            # print(slc.shape)

        elif index >= self.nx + self.ny:

            # Correct the index to accoutn for the first dimension
            index -= int((self.nx + self.ny))

            # index of the image center in the subvolume
            idx, idy, idz = np.unravel_index(index, self.dimz)

            # Position of slice
            isl = np.where(self.labeled['lz'])[0][idz]

            label = self.labeled['lV'][idx, idy, isl]

        return torch.tensor(label)

    def __get_labels__(self):

        targets = 9999 * np.ones(self.NT)

        for i in range(self.NT):
            if np.mod(i, 100000) == 0:
                print(f"Got {i:{len(str(self.NT))}d} of {self.NT} labels")

            targets[i] = self.__get_label__(i)

        return targets

    def __len__(self):
        return self.NT

    # def get_labels(self):
    #     # Labels and counts
    #     ilabels = [i for i in self.labeldict.values()]
    #     nlabels = [0] + [np.sum(self.labeled['lV'] == i)
    #                      for i in ilabels]
    #     print(len(self), np.sum(nlabels))
    #     labels = torch.zeros(len(self))

    #     for i, n in zip(ilabels, nlabels):
    #         labels[n:n+1] = i

    #     return ilabels, nlabels, labels

    def plot_labeled_slices(self, dir='x'):

        print("Starting loop to go through dataset")
        print("Press:")
        print("    Enter - for the next set of images")
        print("    Ctrl + C, then Enter - to end the loop")

        if dir not in 'xy':
            raise ValueError('Only x and y supported.')

        # if dir == 'x':
        #     if i >= self.nlx:
        #         raise ValueError('index too large')
        if dir == 'x':
            n = self.nlx
        else:
            n = self.nly

        for i in range(n):

            if dir == 'x':
                pos = np.where(self.labeled['lx'])[0][i]
                segmentation = self.labeled['lV'][pos, :, :].T
                img = self.V[pos, :, :].T

            else:
                pos = np.where(self.labeled['ly'])[0][i]
                segmentation = self.labeled['lV'][:, pos, :].T
                img = self.V[:, pos, :].T

            fig = plt.figure(figsize=(10, 6))

            img_ax = plt.subplot(121)

            img_ax.imshow(
                img, cmap=self.imgcmap, norm=self.imgnorm, aspect='auto',
                alpha=1.0
            )

            segment_ax = plt.subplot(122)

            labeled_m = np.ma.masked_values(
                segmentation, self.labeldict['none']
            )
            segment_ax.imshow(
                labeled_m, cmap=self.bcmap, norm=self.bnorm, aspect='auto',
                alpha=0.5
            )

            plt.show(block=False)

            input("Press Enter to continue...")
            plt.close(fig)

    def __set_boundary_cmap_norm__(self):

        picknumber = []
        for label, number in self.labeldict.items():
            if label != 'none':
                picknumber.append(number)

        # Get boundary color norm based on label numbers
        pickarray = np.array(picknumber)
        dpickarray = np.diff(pickarray)/2

        # Very artificial create color bounds
        bounds = list(pickarray[:-1] + dpickarray)
        bounds = [pickarray[0] - dpickarray[0]] + bounds
        bounds = bounds + [pickarray[-1] + dpickarray[-1]]

        # Create adhoc cmap and norm
        self.bcmap = plt.get_cmap('rainbow').copy()
        self.bcmap.set_bad('lightgray', alpha=0.0)

        self.bnorm = BoundaryNorm(bounds, self.bcmap.N)

    def __set_img_cmap_norm__(self):

        # Create adhoc cmap and norm
        self.imgcmap = plt.get_cmap('rainbow').copy()
        self.imgcmap.set_bad('lightgray', alpha=0.0)

        self.imgnorm = Normalize(-0.02, 0.04)

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
                im = plt.imshow(img.squeeze().numpy(),
                                cmap="rainbow", aspect='auto')
            # figure.colorbar(im)
            plt.show(block=False)

            input("Press Enter to continue...")
            plt.close()
