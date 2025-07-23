import numpy as np, pandas as pd, nibabel as nib
from nilearn.datasets import fetch_atlas_harvard_oxford, load_mni152_brain_mask, load_mni152_template
from nilearn.image import resample_to_img, iter_img, load_img, new_img_like
from nilearn.regions import connected_regions
from scipy.ndimage import center_of_mass
from scipy.stats import ttest_ind
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests

# -------------- SETTINGS --------------
T_MAP     = 'vlsm_tstat.nii.gz'
P_MASK    = 'vlsm_pval_fdr05.nii.gz'
OUT_CSV   = 'vlsm_clusters_summary.csv'
MIN_VOX   = 10            # drop clusters smaller than this
# --------------------------------------

print('1/6 • Load t-map & FDR mask …')
t_img = load_img(T_MAP)
p_img = load_img(P_MASK)

print('2/6 • Connected components …')
clusters_4d, sizes = connected_regions(p_img, min_region_size=MIN_VOX)
print(f'   → {len(sizes)} clusters extracted')

print('3/6 • Prepare brain mask & voxel geometry …')
brain_mask = load_mni152_brain_mask(resolution=2)
template   = load_mni152_template(resolution=2)
voxel_vol  = abs(np.linalg.det(template.affine))

print('4/6 • Fetch & resample Harvard–Oxford atlas …')
ho = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
ho_img = load_img(ho.maps)
# resample atlas into t-map/template space
ho2 = resample_to_img(ho_img, template, interpolation='nearest')
ho_data   = ho2.get_fdata().astype(int)
ho_labels = ho.labels

print('5/6 • Iterate clusters & gather stats …')
rows = []
for idx, (clust_img, size) in enumerate(zip(iter_img(clusters_4d), sizes), start=1):
    mask = clust_img.get_fdata().astype(bool)

    # t-values within cluster
    tvals = t_img.get_fdata()[mask]
    peak_t, mean_t = float(tvals.max()), float(tvals.mean())

    # peak coordinate in vox then mm
    vox_idx = np.unravel_index(np.argmax(np.abs(t_img.get_fdata()) * mask), mask.shape)
    peak_xyz = nib.affines.apply_affine(template.affine, vox_idx)

    # center-of-mass in mm
    com_vox = center_of_mass(mask.astype(float))
    com_xyz = nib.affines.apply_affine(template.affine, com_vox)

    # atlas label: most-frequent nonzero code in cluster
    codes = ho_data[mask]
    codes = codes[codes > 0]
    label = ho_labels[int(np.bincount(codes).argmax())] if codes.size else 'unlabelled'

    rows.append({
        'cluster_id': idx,
        'voxels': int(size),
        'volume_mm3': float(size * voxel_vol),
        'peak_t': peak_t,
        'mean_t': mean_t,
        'peak_x': peak_xyz[0],
        'peak_y': peak_xyz[1],
        'peak_z': peak_xyz[2],
        'com_x': com_xyz[0],
        'com_y': com_xyz[1],
        'com_z': com_xyz[2],
        'label': label
    })

print('6/6 • Writing CSV …')
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f'✓ Written {OUT_CSV} with {len(rows)} clusters')
