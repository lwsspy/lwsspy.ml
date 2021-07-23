# %%
from lwsspy.ml.labeling.volumelabeler import VolumeLabeler

# %% First round of labeling
infile = '/Users/lucassawade/OneDrive/Research/RF/DATA/GLImER/ccps/US_P_0.58_minrad_3D_it_f2_volume.pkl.npz'
outfile = '/Users/lucassawade/OneDrive/Research/RF/DATA/GLImER/ccps/US_P_0.58_minrad_3D_it_f2_volume_labeled.npz'

vl = VolumeLabeler.from_volume(infile)
vl.label(direction='x')

# %%
# Label the thing
vl.save(outfile=outfile)

# %% Subsequent labeling

vl = VolumeLabeler.from_labeled_volume(outfile)

vl.save(outfile=outfile)
