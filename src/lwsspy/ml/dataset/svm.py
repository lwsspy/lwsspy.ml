import numpy as np
from scipy.ndimage import convolve, uniform_filter
from scipy.signal import fftconvolve
from copy import copy
import pandas as pd
import matplotlib.pyplot as plt


def fix_labeling(lx, ly, lz, lV):

    # kjlskj
    x = copy(lV)
    x[~lx, :, :] = -999
    y = copy(lV)
    y[:, ~ly, :] = -999
    z = copy(lV)
    z[:, :, ~lz] = -999

    # lkjdsf
    xs = ~np.equal(x, -999)
    ys = ~np.equal(y, -999)
    zs = ~np.equal(z, -999)

    pos = (xs | ys | zs)

    lV[~pos] = -999

    return lV


def CCP2DF():
    filename = \
        '/Users/lucassawade/OneDrive/Research/RF/DATA/GLImER/' \
        'ccps/US_P_0.58_minrad_3D_it_f2_volume_labeled.npz'

    # Load NPZ file
    vardict = np.load(filename, allow_pickle=True)

    # Assign variables
    x = vardict["x"]
    y = vardict["y"]
    z = vardict["z"]
    V = vardict["V"]
    V = V - np.min(V)
    V = V/np.max(V)

    # Labeled
    labeled = dict()
    labeled["lx"] = vardict["lx"]
    labeled["ly"] = vardict["ly"]
    labeled["lz"] = vardict["lz"]
    labeled["lV"] = vardict["lV"]

    # fix labeling
    labeled["lV"] = fix_labeling(
        labeled["lx"], labeled["ly"], labeled["lz"], labeled["lV"])

    # All positions that are labeled.
    positions = np.where(~np.equal(labeled["lV"], -999))

    # Create filter with size
    size = 75

    # Slice mean kernels
    kxmean = np.ones((size, size))/size**2
    kymean = np.ones((size, size))/size**2
    kzmean = np.ones((size, size))/size**2
    kxmean = kxmean.reshape((1, size, size))
    kymean = kymean.reshape((size, 1, size))
    kzmean = kzmean.reshape((size, size, 1))

    # %
    print("Computing local mean, std ...")
    VV = V*V
    lmean = uniform_filter(V, size=size, mode='constant', cval=np.mean(V))
    lsq = uniform_filter(V*V, size=size, mode='constant', cval=np.mean(VV))

    # % this creates the matrix
    print("Computing slice means ...")
    xmean = fftconvolve(V, kxmean, mode='same')
    ymean = fftconvolve(V, kymean, mode='same')
    zmean = fftconvolve(V, kzmean, mode='same')

    print("Computing slice vars ...")
    xsq = fftconvolve(VV, kxmean, mode='same')
    ysq = fftconvolve(VV, kymean, mode='same')
    zsq = fftconvolve(VV, kzmean, mode='same')

    del VV

    # Compute Standard deviations
    print("Computing slice std")
    lstd = np.sqrt(lsq - lmean ** 2)
    xstd = np.sqrt(xsq - xmean ** 2)
    ystd = np.sqrt(ysq - ymean ** 2)
    zstd = np.sqrt(zsq - zmean ** 2)

    del xsq
    del ysq
    del zsq

    # Get coordinates for the table
    xx, yy, zz = np.meshgrid(x, y, z)
    xx = xx.reshape(V.shape)
    yy = yy.reshape(V.shape)
    zz = zz.reshape(V.shape)

    print("Summarizing in Dataframe")
    df = pd.DataFrame(
        dict(
            data=V[positions],
            x=xx[positions],
            y=yy[positions],
            z=zz[positions],
            lmean=lmean[positions],
            lstd=lstd[positions],
            xmean=xmean[positions],
            xstd=xstd[positions],
            ymean=ymean[positions],
            ystd=ystd[positions],
            zmean=zmean[positions],
            zstd=zstd[positions],
            target=labeled['lV'][positions]
        )
    )

    return df


def plot_df(df, labels=None):

    df1 = df.drop(['x', 'y', 'z'], axis=1)
    fig = plt.figure(figsize=(10, 8))
    N = len(df1.columns[:-1])

    if hasattr(df1, 'labels'):
        pos = np.where(df1['labels'] != 5)[0]
        labels = df1['labels'].iloc[pos]
    else:
        pos = np.where(labels != 5)[0]
        labels = labels[pos]
    for _i, iname in enumerate(df1.columns[:-1]):
        for _j, jname in enumerate(df1.columns[_i+1:-1]):

            ax = plt.subplot(N, N, _i * N + _j + (_i+1) + 1)

            # counts, xbin, ybin = np.histogram2d(
            #     df1[jname].iloc[pos], df1[iname].iloc[pos], 100, density=True)
            # ax.imshow(counts, aspect='auto', extent=[
            #           xbin.min(), xbin.max(), ybin.min(), ybin.max()],
            #           cmap='hot_r')
            ax.scatter(df1[jname].iloc[pos], df1[iname].iloc[pos], c=labels, s=1,
                       cmap='rainbow', alpha=0.1)

            # ax.set_ylabel(iname)
            # ax.set_yticklabels([])
            # ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)
            # print(_j, N-_i)
            if _i == 0:
                ax.set_xlabel(jname)
                ax.xaxis.set_label_position('top')
            if _j == N-_i-2:
                ax.set_ylabel(iname)
                ax.yaxis.set_label_position('right')
            # ax.scatter(df[iname].iloc[pos], df[jname].iloc[pos],
            #            c=df['labels'].iloc[pos], cmap='rainbow')

    plt.show()
