

# External
from typing import Optional
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from skimage.segmentation import slic

# Internal
from .segmentlabeler import SegmentLabeler


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class VolumeLabeler:

    def __init__(
            self, x, y, z, V,
            labeldict={'moho': 1, '410': 2, '660': 3, 'none': 9999},
            labeled: Optional[dict] = None):
        """This class uses the 
        :class:``lwsspy.ml.labeling.segementlabeler.SegmentLabeler`` to label
        slices in a volume. After instantiation, use the label 
        method to label a Volumen of single Channel data.

        Parameters
        ----------
        x : arraylike
            x vector
        y : arralike
            y vector
        z : arraylike
            z vector
        V : arraylike
            3D array with data corresponding to vectors X x Y x Z direction.
        labeldict : dict, optional
            dictionary with labels, by default {'moho': 1, '410': 2, '660': 3, 'none': 9999}
        labeled : Optional[dict], optional
            Optional dictionary with labeled volume. Labeled should contain 
            4 variables lx, ly, lz, lV, where l{x,y,z} are boolean arrays that
            where defined as locations for labeled layers. 
            Useful if you need to take a labeling break. Right now there is no 
            way of resuming to an image make sure you finisshe the ones you
            have started, by default None

        Notes
        -----

        Things that I have to fix:

        1. The volume should be allowed to be 3-channel data
        2. Also allow for custom normalizations and colormaps

        :Authors:
            Lucas Sawade (lsawade@princeton.edu)

        :Last Modified:
            2021.07.02 00.00 (Lucas Sawade)

        """

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
    def from_volume(self, filename, *args, **kwargs):

        # Load NPZ file

        vardict = np.load(filename)

        # Assign variables
        x = vardict["x"]
        y = vardict["y"]
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
        """The labeling method enables you to label a ``n`` slices
        The labeler will automatically pick ``n`` unlabeled
        slices randomly. The direction let's you decide the normal to the 
        slices.

        Parameters
        ----------
        direction : str, optional
            normal to slices, by default 'x'
        n : int, optional
            number of slices tto label, by default 20
        n_segments : int, optional
            number of segments to split the image into, by default 700

        Raises
        ------
        ValueError
            [description]
        """

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
