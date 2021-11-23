
from imblearn.under_sampling import RandomUnderSampler
from sklearn import svm
import matplotlib.pyplot as plt
from lwsspy.ml.dataset.svm import CCP2DF, plot_df
import hvplot.pandas  # noqa
# %% Support vector machine testing
"""Here image processing that I'm going to do to compute local statistics,
That I then use for pairplots. """


df = CCP2DF()

# %%
plot_df(df)

# %%

xslice = 100
plt.figure()
plt.subplot(2, 5, 1)
plt.imshow(x[xslice, :, :].T, aspect='auto')

# Local mean
plt.subplot(2, 5, 2)
plt.imshow(lmean[xslice, :, :].T, aspect='auto')
plt.subplot(2, 5, 7)
plt.imshow(lstd[xslice, :, :].T, aspect='auto')

# X Slice mean
plt.subplot(2, 5, 3)
plt.imshow(xmean[xslice, :, :].T, aspect='auto')
plt.subplot(2, 5, 8)
plt.imshow(xstd[xslice, :, :].T, aspect='auto')

plt.subplot(2, 5, 4)
plt.imshow(ymean[xslice, :, :].T, aspect='auto')
plt.subplot(2, 5, 9)
plt.imshow(ystd[xslice, :, :].T, aspect='auto')

plt.subplot(2, 5, 5)
plt.imshow(zmean[xslice, :, :].T, aspect='auto')
plt.subplot(2, 5, 10)
plt.imshow(zstd[xslice, :, :].T, aspect='auto')

# %%

# %%
svm.SVC
class_weight = 'balanced'

# %%


def plot_2d_space(X, y, label='Classes'):
    colors = ['r', 'g', 'b', 'y']
    markers = ['o', 's', '+', 'd']
    for l, c, m in zip(np.sort(np.unique(y)), colors, markers):
        plt.scatter(
            X[y == l, 0],
            X[y == l, 1], s=3,
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


def plot_3d_space(X, y, label='Classes'):
    colors = ['r', 'g', 'b', 'y']
    markers = ['o', 's', '+', 'd']
    for l, c, m in zip(np.sort(np.unique(y)), colors, markers):
        plt.scatter(
            X[y == l, 0],
            X[y == l, 1],
            X[y == l, 2], s=1,
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


# %% Drop rows with NaNs
df = df.dropna()

# %% Undersampling


rus = RandomUnderSampler(sampling_strategy='all')
Xr, yr = rus.fit_resample(df[df.columns[-1]], df['target'])
