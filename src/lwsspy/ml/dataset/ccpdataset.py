"""
This file shows how to get from a CCP labeled volume to a Pytorch datta set

"""
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm
from copy import copy
# from ...plot import plot_label


class CCPDataset(Dataset):

    def __init__(self, filename, sq_size: int = 33):
        super().__init__()
        if sq_size % 2 == 0:
            raise ValueError("The square image size must be odd.")

        # Load NPZ file
        vardict = np.load(filename, allow_pickle=True)

        # Assign variables
        self.x = torch.from_numpy(vardict["x"])
        self.y = torch.from_numpy(vardict["y"])
        self.z = torch.from_numpy(vardict["z"])
        self.V = torch.from_numpy(vardict["V"])
        self.V = self.V - self.V.min()
        self.V = self.V/self.V.max()
        self.V = self.V.float()

        # Labeled
        self.labeled = dict()
        self.labeled["lx"] = torch.from_numpy(vardict["lx"])
        self.labeled["ly"] = torch.from_numpy(vardict["ly"])
        self.labeled["lz"] = torch.from_numpy(vardict["lz"])
        self.labeled["lV"] = torch.from_numpy(vardict["lV"])

        # Making sure we don't
        self.fix_labeling()
        print('get where')
        self.positions = torch.stack(list(torch.where(
            ~torch.eq(self.labeled["lV"], -999))))
        self.npositions = torch.stack(list(torch.where(
            torch.eq(self.labeled["lV"], -999))))

        print('set nt')
        self.NT = torch.tensor(self.positions[0].size()[0])
        self.targets = self.labeled["lV"][
            self.positions[0], self.positions[1], self.positions[2]]

        # Label dictionary
        self.labeldict = vardict["labeldict"].item()

        # Padding value
        self.pad = int((sq_size - 1)/2)
        self.padtuple = torch.tensor([self.pad, self.pad, self.pad, self.pad])
        self.padval = 0
        self.padmask = self.labeldict['none']

        # Number of labeled slices in each dimension l? is a boolean array
        self.nlx = torch.sum(self.labeled['lx'])
        self.nly = torch.sum(self.labeled['ly'])
        self.nlz = torch.sum(self.labeled['lz'])

        # Total number of items is
        # n labeled slice in one direction
        #   X dimension of the other two dimensions
        self.nx = torch.tensor(self.nlx * self.y.size()[0] * self.z.size()[0])
        self.ny = torch.tensor(self.nly * self.x.size()[0] * self.z.size()[0])
        self.nz = torch.tensor(self.nlz * self.x.size()[0] * self.y.size()[0])
        # self.NT = self.nx + self.ny + self.nz

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

    def fix_labeling(self):

        # kjlskj
        x = torch.clone(self.labeled["lV"])
        x[~self.labeled['lx'], :, :] = -999
        y = torch.clone(self.labeled["lV"])
        y[:, ~self.labeled['ly'], :] = -999
        z = torch.clone(self.labeled["lV"])
        z[:, :, ~self.labeled['lz']] = -999

        # lkjdsf
        xs = ~torch.eq(x, -999)
        ys = ~torch.eq(y, -999)
        zs = ~torch.eq(z, -999)

        pos = (xs | ys | zs)

        # xpos = np.hstack((xs[0], ys[0], zs[0]))
        # ypos = np.hstack((xs[1], ys[1], zs[1]))
        # zpos = np.hstack((xs[2], ys[2], zs[2]))

        # pos = np.vstack((xpos, ypos, zpos))
        # pos = np.unique(pos, axis=1)
        # pos = pos[0], pos[1], pos[2]

        # mask = np.ones_like(self.labeled['lV'], np.bool)
        # mask[pos] = 0
        self.labeled['lV'][~pos] = -999

    def to(self, device):

        self.NT.to(device)
        self.positions.to(device)
        self.V.to(device)
        self.padtuple.to(device)

        for k, v in self.labeled.items():
            v.to(device)

    def __getitem__(self, index):

        if index >= self.NT:
            raise ValueError('Limit reached')

        # first get position
        idx = self.positions[0][index]
        idy = self.positions[1][index]
        idz = self.positions[2][index]

        # Get slices
        xslc = torch.nn.functional.pad(
            self.V[idx, :, :], pad=list(self.padtuple), mode='constant', value=0)
        yslc = torch.nn.functional.pad(
            self.V[:, idy, :], pad=list(self.padtuple), mode='constant', value=0)
        zslc = torch.nn.functional.pad(
            self.V[:, :, idz], pad=list(self.padtuple), mode='constant', value=0)

        # Get excerpts
        ximage = xslc[
            idy: idy + 2 * self.pad + 1,
            idz: idz + 2 * self.pad + 1].T

        yimage = yslc[
            idx: idx + 2 * self.pad + 1,
            idz: idz + 2 * self.pad + 1].T

        zimage = zslc[
            idx: idx + 2 * self.pad + 1,
            idy: idy + 2 * self.pad + 1]

        label = self.labeled['lV'][idx, idy, idz]

        image = torch.stack((ximage, yimage, zimage), axis=0)

        return image, label

    def __get_label__(self, index):

        return self.targets[index]

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
        self.bcmap = copy(plt.get_cmap('rainbow'))
        self.bcmap.set_bad('lightgray', alpha=0.0)

        self.bnorm = BoundaryNorm(bounds, self.bcmap.N)

    def __set_img_cmap_norm__(self):

        # Create adhoc cmap and norm
        self.imgcmap = plt.get_cmap('rainbow').copy()
        self.imgcmap.set_bad('lightgray', alpha=0.0)

        self.imgnorm = Normalize(
            np.quantile(self.V, 0.01), np.quantile(self.V, 0.99))

    def plot_samples(self):

        labels_map = {v: k for (k, v) in self.labeldict.items()}

        print("Starting loop to go through dataset")
        print("Press:")
        print("    Enter - for the next set of images")
        print("    Ctrl + C, then Enter - to end the loop")

        while True:
            figure = plt.figure(figsize=(8, 8))
            cols, rows = 3, 3

            for i in range(0, 3):

                label = self.labeldict['none']

                while label == self.labeldict['none']:

                    sample_idx = torch.randint(self.NT, size=(1,)).item()
                    img, label = self[sample_idx]

                for _j, _l in enumerate(['X', 'Y', 'Z']):
                    figure.add_subplot(rows, cols, 3*i+1+_j)
                    plt.title(f"{_l}: {labels_map[int(label)]}")
                    plt.axis("off")
                    im = plt.imshow(img[:, _j, :, :].squeeze().numpy(),
                                    cmap="rainbow", aspect='auto')
            # figure.colorbar(im)
            plt.show(block=False)

            input("Press Enter to continue...")
            plt.close()
