import glob, os
import numpy as np
import pandas as pd
import nibabel as nib

# 2-A. Load and collapse AG-overlap to one row / subject
df_raw = pd.read_csv('ag_overlap_summary.csv')
# here we take the MAX per subject, but you could use sum(): .sum()
df_ag  = df_raw.groupby('subject', as_index=True)['ag_voxels'].max()

rows = []
for mask in glob.glob('derivatives/lesion_masks/**/*_mask.nii.gz', recursive=True):
    subj = os.path.basename(mask).split('_')[0]
    data = nib.load(mask).get_fdata() > 0
    lesion_vox = int(data.sum())
    ag_vox     = int(df_ag.loc[subj]) if subj in df_ag.index else 0
    rows.append({'subject': subj,
                 'lesion_vox': lesion_vox,
                 'ag_voxels': ag_vox})

pd.DataFrame(rows).to_csv('lesion_volumes.csv', index=False)
print(f"✓ lesion_volumes.csv written; {len(rows)} subjects total")
