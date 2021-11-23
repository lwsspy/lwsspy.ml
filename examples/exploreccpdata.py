# %%
from matplotlib.gridspec import GridSpec
from lwsspy.ml.dataset.ccpdataset import CCPDataset
import matplotlib.pyplot as plt
import numpy as np

filename = \
    '/Users/lucassawade/OneDrive/Research/RF/DATA/GLImER/' \
    'ccps/US_P_0.58_minrad_3D_it_f2_volume_labeled.npz'
sq_size = 75
dataset = CCPDataset(filename, sq_size=sq_size)

print("Labels: ")
print(dataset.labeldict)

# %% Checkout the implemented sample plotting tool (nothing crazy)
dataset.plot_samples()

# %% Subsample the arrays
V = dataset.V.flatten()
V = V[V != 0]
Vmoho = dataset.V[dataset.labeled['lV'] == 0].flatten()
Vmoho = Vmoho[Vmoho != 0]
V410 = dataset.V[dataset.labeled['lV'] == 1].flatten()
V410 = V410[V410 != 0]
V660 = dataset.V[dataset.labeled['lV'] == 2].flatten()
V660 = V660[V660 != 0]

# %%

fig = plt.figure(figsize=(10, 8))
gs = GridSpec(ncols=2, nrows=4, hspace=0.4, height_ratios=[2, 1, 1, 1])


bins1 = np.linspace(-0.5, 0.5, 8000)
bins = np.linspace(-0.25, 0.25, 2000)


lim = 0.125
ax0 = fig.add_subplot(gs[:, 0])
plt.hist(V, bins=bins1, density=True)

ax1 = fig.add_subplot(gs[1, 1])
plt.title(f'Moho: $\mu$={np.mean(Vmoho):0.4f}, $\sigma$={np.std(Vmoho):0.4f}')
plt.hist(Vmoho,
         bins=bins, density=True, label="Moho")
plt.xlim(-lim, lim)

ax2 = fig.add_subplot(gs[2, 1])
plt.title(f'410: $\mu$={np.mean(V410):0.4f}, $\sigma$={np.std(V410):0.4f}')
plt.hist(V410,
         bins=bins, density=True, label="410")
plt.xlim(-lim, lim)

ax3 = fig.add_subplot(gs[3, 1])
plt.title(f'660: $\mu$={np.mean(V660):0.4f}, $\sigma$={np.std(V660):0.4f}')
plt.hist(V660[V660 != 0],
         bins=bins, density=True, label="660")
plt.xlim(-lim, lim)

ax4 = fig.add_subplot(gs[0, 1])
plt.title()
plt.hist(Vmoho,
         bins=bins, density=True, label="Moho", histtype='step')
plt.hist(V410,
         bins=bins, density=True, label="410", histtype='step')
plt.hist(V660,
         bins=bins, density=True, label="660", histtype='step')
plt.xlim(-lim, lim)
plt.legend()

plt.show()
