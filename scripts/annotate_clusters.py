#!/usr/bin/env python3
import numpy as np, pandas as pd, nibabel as nib
from nilearn.regions import connected_regions
from nilearn.image    import load_img, iter_img, resample_to_img, new_img_like
from nilearn.datasets import fetch_atlas_harvard_oxford, load_mni152_template
from scipy.ndimage    import center_of_mass
from tqdm             import tqdm

# Settings
T_MAP       = 'vlsm_tstat.nii.gz'
P_MASK      = 'vlsm_pval_fdr05.nii.gz'
MIN_CLUSTER = 10
OUT_CSV     = 'vlsm_clusters_summary.csv'

print('1/6 • Load images …')
t_img = load_img(T_MAP)
p_img = load_img(P_MASK)

print('2/6 • Connected components (min', MIN_CLUSTER, 'vox) …')
clusters_4d, _ = connected_regions(p_img, min_region_size=MIN_CLUSTER)

print('3/6 • Template & voxel volume …')
template  = load_mni152_template(resolution=2)
vox_vol   = abs(np.linalg.det(template.affine))

print('4/6 • Fetch & resample HO atlas …')
ho        = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
ho_img    = resample_to_img(load_img(ho.maps), template, interpolation='nearest')
ho_data   = ho_img.get_fdata().astype(int)
ho_labels = ho.labels

print('5/6 • Iterate clusters & compute stats …')
rows = []
for cid, cl_img in enumerate(iter_img(clusters_4d), start=1):
    mask     = cl_img.get_fdata().astype(bool)
    voxels   = int(mask.sum())                # <-- recalc size here
    volume   = float(voxels * vox_vol)
    tvals    = t_img.get_fdata()[mask]
    peak_t   = float(tvals.max()) if voxels else np.nan
    mean_t   = float(tvals.mean()) if voxels else np.nan

    # peak coordinate
    idx_peak = np.unravel_index(np.nanargmax(np.abs(t_img.get_fdata())*mask), mask.shape)
    peak_xyz = nib.affines.apply_affine(template.affine, idx_peak)

    # center‐of‐mass
    com_vox  = center_of_mass(mask.astype(float))
    com_xyz  = nib.affines.apply_affine(template.affine, com_vox)

    # HO label
    codes    = ho_data[mask]
    codes    = codes[codes>0]
    label    = ho_labels[int(np.bincount(codes).argmax())] if codes.size else 'unlabelled'

    # save NIfTI
    nii_name = f'cluster_{cid:02d}.nii.gz'
    new_img_like(template, mask.astype(np.int32)).to_filename(nii_name)

    rows.append({
      'cluster_id': cid,
      'voxels': voxels,
      'volume_mm3': volume,
      'peak_t': peak_t,
      'mean_t': mean_t,
      'peak_x': peak_xyz[0], 'peak_y': peak_xyz[1], 'peak_z': peak_xyz[2],
      'com_x': com_xyz[0],   'com_y': com_xyz[1],   'com_z': com_xyz[2],
      'label': label,
      'nifti': nii_name
    })

print('6/6 • Writing CSV …')
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f'✓ Wrote {OUT_CSV} with {len(rows)} clusters')
