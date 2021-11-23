# %%
from skimage.segmentation import random_walker
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from numpy.lib.utils import safe_eval
from lwsspy.ml.labeling.volumelabeler import VolumeLabeler
from lwsspy.ml.labeling.segmentlabeler import SegmentLabeler
from skimage.segmentation import slic

# %% First round of labeling
infile = '/Users/lucassawade/OneDrive/Research/RF/DATA/GLImER/ccps/US_P_0.58_minrad_3D_it_f2_volume.pkl.npz'
outfile = '/Users/lucassawade/OneDrive/Research/RF/DATA/GLImER/ccps/US_P_0.58_minrad_3D_it_f2_volume_labeled.npz'

# %%
vl = VolumeLabeler.from_volume(infile)
vl.label(direction='x')

# %%
# Label the thing
vl.save(outfile=outfile)

# %% Subsequent labeling

vl = VolumeLabeler.from_labeled_volume(outfile)

vl.save(outfile=outfile)

# %% label just a segment

vl = VolumeLabeler.from_volume(infile)

# %% spacing make it kilometres

dx = np.diff(vl.x)[0]*111.11
dy = np.diff(vl.y)[0]*111.11
dz = np.diff(vl.y)[0]
# %%
array = vl.V[:, 50, :].T

cmap = plt.get_cmap('rainbow')
# norm = Normalize(vmin=-0.025, vmax=0.075)
norm = Normalize(vmin=-0.01, vmax=0.025)  # 410/660
img2 = cmap(norm(array))
img = img2[:, :, :-1]

# %% Denoise image
im_denoised = restoration.denoise_nl_means(array, h=0.0025)
plt.figure()
plt.imshow(im_denoised, cmap=cmap, norm=norm)
plt.show()

# %%

# 660/410
slicdict = dict(
    n_segments=1000,
    compactness=0.0001,
    sigma=0.01,
    start_label=0,
    spacing=(1, dx, dy)
)

# Moho
# slicdict = dict(
#     n_segments=1000,
#     compactness=0.025,
#     sigma=0.001,
#     start_label=0
#     spacing=(dx, dy)
# )

# # Super Pixel segmentation
# segments = slic(array, **slicdict, enforce_connectivity=True)
# # segments = slic(array, **slicdict, enforce_connectivity=True)

# seg = SegmentLabeler(img, segments)
# seg.start_labeling()

# %% random walker segmentation
data = im_denoised

# Generate noisy synthetic data
# data = skimage.img_as_float(binary_blobs(length=128, seed=1))
# sigma = 0.35
# data += rng.normal(loc=0, scale=sigma, size=data.shape)
# data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
#                          out_range=(-1, 1))

# The range of the binary image spans over (-1, 1).
# We choose the hottest and the coldest pixels as markers.

# markers[np.abs(data) < 0.001] = 1

# %%

# Moho -- Use some threshold as starting marker
threshold = np.ones(data.shape)

threshold[300:, :][data[300:, :] > 0.0025] = 2
# threshold[300:, :][data[300:, :] < 0.0025] = 1
threshold[:200, :][data[:200, :] > 0.025] = 2
# threshold[:200, :][data[:200, :] < 0.05] = 1

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
axes[0].imshow(array, cmap=cmap, norm=norm, aspect='auto')
axes[0].set_title('Raw')
axes[1].imshow(threshold, aspect='auto')
axes[1].set_title('thresholded')


# %%


slicdict = dict(
    n_segments=1000,
    compactness=0.01,
    sigma=1.0,
    start_label=1,
    # spacing=(1, dx, dy)
)

mask = np.zeros(data.shape, dtype=bool)
mask[:150, :] = True
mask[300:, ]
# Super Pixel segmentation
segments = slic(threshold, **slicdict, enforce_connectivity=True,)
# segments = slic(array, **slicdict, enforce_connectivity=True)

img2 = cmap(norm(im_denoised[:200, :]))
img = img2[:, :, :-1]
seg = SegmentLabeler(img, segments)
seg.start_labeling()


# %%
fig, axes = plt.subplots(2, 3, figsize=(10, 4), sharex=True, sharey=True)
axes[0][0].imshow(array, aspect='auto', cmap=cmap, norm=norm)
axes[0][0].set_title('RF section')
axes[0][1].imshow(im_denoised, aspect='auto', cmap=cmap, norm=norm)
axes[0][1].set_title('denoised')
axes[0][2].imshow(markers, aspect='auto')
axes[0][2].set_title('markers for random walker')
axes[1][0].imshow(labels, aspect='auto')
axes[1][0].set_title('labels after random walk')
axes[1][1].imshow(array, aspect='auto', cmap=cmap, norm=norm)
axes[1][1].imshow(mark_boundaries(img, segments, color=(0, 0, 0), mode='inner'),
                  aspect='auto')
axes[1][1].set_title('superpixels with rw input')
axes[1][2].axis('off')
