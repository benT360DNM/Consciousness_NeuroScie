import glob, os
import pandas as pd, numpy as np
import nibabel as nib
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img

# Fetch the 2 mm Harvard–Oxford cortical atlas
atlas_res = fetch_atlas_harvard_oxford('cort-prob-2mm')

# Load the image (could be str or Nifti1Image)
atlas_img = nib.load(atlas_res.maps) if isinstance(atlas_res.maps, str) else atlas_res.maps
atlas_data = atlas_img.get_fdata()

# Find index for Angular Gyrus
ag_idx = atlas_res.labels.index('Angular Gyrus')

# Extract a 3D probability map for AG
if atlas_data.ndim == 4:
    ag_prob = atlas_data[..., ag_idx]        # shape (X,Y,Z)
else:
    # some atlases store labels as integer codes in 3D
    ag_prob = (atlas_data == ag_idx).astype(float)

# Make a binary ROI (threshold >0); for >50% use ag_prob > 0.5
ag_roi = ag_prob > 0

rows = []
for mask_path in glob.glob('derivatives/lesion_masks/**/*_mask.nii.gz', recursive=True):
    try:
        mask_img = nib.load(mask_path)
    except FileNotFoundError:
        print(f"⚠️ Missing file, skipping: {mask_path}")
        continue

    # Resample lesion mask into atlas space
    mask_resamp = resample_to_img(mask_img, atlas_img, interpolation='nearest')
    mask_data   = mask_resamp.get_fdata() > 0

    # Compute overlap
    overlap_vox = int(np.sum(mask_data & ag_roi))
    if overlap_vox:
        subj = os.path.basename(mask_path).split('_')[0]
        rows.append({'subject': subj, 'ag_voxels': overlap_vox})

# Save results
pd.DataFrame(rows).to_csv('ag_overlap_summary.csv', index=False)
print(f"✓ ag_overlap_summary.csv written; {len(rows)} subjects hit the AG")
