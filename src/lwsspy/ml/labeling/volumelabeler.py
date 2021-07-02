
# %%
from typing import Optional
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from pyglimer.ccp.plot_utils.midpointcolornorm import MidpointNormalize

import matplotlib.pyplot as plt
import numpy as np

from skimage.segmentation import slic

from .segmentlabeler import SegmentLabeler


class VolumeLabeler:

    def __init__(
            self, x, y, z, V,
            labeldict={'moho': 1, '410': 2, '660': 3, 'none': 9999},
            labeled: Optional[dict] = None):
        """Labeled should contain 4 variables lx, ly, lz, lV, where
        l{x,y,z} are boolean arrays that where defined as locations for 
        labeled layers"""

        # Variables
        self.x = x
        self.y = y
        self.z = z
        self.V = V

        # Dimensions
        self.nx = len(x)
        self.ny = len(y)
        self.nz = len(z)

        # Label info
        self.labeldict = labeldict

        # Load labeled volume, to continue labeling
        if labeled is not None:
            self.labeled = labeled
        else:
            self.labeled = dict()
            self.labeled['lx'] = np.zeros(self.nx, dtype=bool)
            self.labeled['ly'] = np.zeros(self.ny, dtype=bool)
            self.labeled['lz'] = np.zeros(self.ny, dtype=bool)
            self.labeled['lV'] = self.labeldict['none'] * \
                np.ones(self.V.shape, dtype=int)

        # Colormap choices
        self.cmap = plt.get_cmap('seismic')
        self.norm = MidpointNormalize(midpoint=0, vmin=np.min(
            self.V), vmax=0.25 * np.max(self.V))

    @classmethod
    def from_CCPVolume(self, filename, *args, **kwargs):

        # Load NPZ file

        vardict = np.load(filename)

        # Assign variables
        y = vardict["y"]
        x = vardict["x"]
        z = vardict["z"]
        V = vardict["data"]

        return self(x, y, z, V, *args, **kwargs)

    @classmethod
    def from_labeled_volume(self, filename, *args, **kwargs):

        # Load NPZ file
        vardict = np.load(filename, allow_pickle=True)

        # Assign variables
        x = vardict["x"]
        y = vardict["y"]
        z = vardict["z"]
        V = vardict["V"]

        # Labeled
        labeled = dict()
        labeled["lx"] = vardict["lx"]
        labeled["ly"] = vardict["ly"]
        labeled["lz"] = vardict["lz"]
        labeled["lV"] = vardict["lV"]

        # Label dictionary
        labeldict = vardict["labeldict"]

        return self(
            x, y, z, V, *args,
            labeled=labeled, labeldict=labeldict, **kwargs)

    def save(self, outfile):

        np.savez(
            outfile,
            x=self.x, y=self.y, z=self.z, V=self.V,
            lx=self.labeled['lx'],
            ly=self.labeled['ly'],
            lz=self.labeled['lz'],
            lV=self.labeled['lV'],
            labeldict=self.labeldict
        )

    def label(self, direction='x', n=20, n_segments=700):

        # Get slice to plot
        if direction == 'x':
            tobelabeled = self.__get_randarray__(self.labeled["lx"], n)
        elif direction == 'y':
            tobelabeled = self.__get_randarray__(self.labeled["ly"], n)
        elif direction == 'z':
            tobelabeled = self.__get_randarray__(self.labeled["lz"], n)
        else:
            raise ValueError(f'Direction {direction} not implemented.')

        for _label_idx in tobelabeled:
            try:
                # Get slice to plot
                if direction == 'x':
                    array = self.V[_label_idx, :, :].T
                elif direction == 'y':
                    array = self.V[:, _label_idx, :].T
                elif direction == 'z':
                    array = self.V[:, :, _label_idx]

                img2 = self.cmap(self.norm(array))
                img = img2[:, :, :-1]

                # Super Pixel segmentation
                segments = slic(
                    img, n_segments=n_segments, compactness=20, sigma=0,
                    start_label=0)

                # Label
                sl = SegmentLabeler(img, segments, labeldict=self.labeldict)

                # Get slice to plot
                if direction == 'x':
                    self.labeled["lV"][_label_idx,
                                       :, :] = sl.start_labeling().T
                    self.labeled["lx"][_label_idx] = True
                elif direction == 'y':
                    self.labeled["lV"][:, _label_idx,
                                       :] = sl.start_labeling().T
                    self.labeled["ly"][_label_idx] = True
                elif direction == 'z':
                    self.labeled["lV"][:, :, _label_idx] = sl.start_labeling()
                    self.labeled["lz"][_label_idx] = True
            except KeyboardInterrupt:
                break

    @staticmethod
    def __get_randarray__(labeled, N):

        # Get Unlabeled indeces
        unlabeled_indeces = np.where(~labeled)[0]

        # Check whether all labeled
        if len(unlabeled_indeces) == 0:
            raise ValueError('Nothing to be labeled')
        elif len(unlabeled_indeces) < N:
            print('Less to be labeled than indicated.')
            N = len(unlabeled_indeces)

        # Choose N times from unlabeled indeces
        tobelabeled = np.random.choice(
            unlabeled_indeces, size=N, replace=False)

        return tobelabeled


if __name__ == "__main__":

    infile = '/Users/lucassawade/OneDrive/Research/RF/DATA/GLImER/ccps/US_P_0.58_minrad_3D_it_f2_volume.pkl.npz'

    outfile = 'labeled_volume.npz'

    S = Segmenter.from_CCPVolume(infile)

    S.label(direction='y', n=20, n_segments=700)

    S.save(outfile)
