import glob, os, numpy as np, pandas as pd
import nibabel as nib
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiMasker
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# -------- INPUT PARAMETERS --------
CSV      = 'participants_lesion_merged.csv'
MASKS    = 'derivatives/lesion_masks/**/*_mask.nii.gz'
OUT_T    = 'vlsm_tstat.nii.gz'
OUT_P    = 'vlsm_pval_fdr05.nii.gz'
FDR_ALPHA= 0.05
# ----------------------------------

print('1/7 Loading clinical table …')
df = pd.read_csv(CSV)

print('2/7 Loading MNI152 2 mm template …')
template_img = load_mni152_template(resolution=2)

print('3/7 Gathering lesion masks …')
mask_paths = sorted(glob.glob(MASKS, recursive=True))
imgs, subjects = [], []
for path in tqdm(mask_paths):
    subj = os.path.basename(path).split('_')[0]
    if subj not in df.subject.values:
        continue
    img = nib.load(path)
    # resample to template voxel grid
    img2 = resample_to_img(img, template_img, interpolation='nearest')
    imgs.append(img2)
    subjects.append(subj)
print(f'   → Kept {len(imgs)} lesion maps')

print('4/7 Building 4D image & masker …')
stack = nib.concat_images(imgs)                    # 4D image (x,y,z,n)
masker = NiftiMasker(mask_strategy='background')  
mat = masker.fit_transform(stack).astype(bool)      # shape: (n_subjects, n_voxels)

print('5/7 Aligning NIHSS scores …')
scores = df.set_index('subject').loc[subjects, 'nihss'].values

print('6/7 Running voxel-wise t-tests …')
# ttest_ind across subjects for each voxel
t_vals, p_vals = ttest_ind(
    scores[mat], 
    scores[~mat], 
    axis=0, 
    equal_var=False, 
    nan_policy='omit'
)

print('7/7 FDR correction & saving maps …')
reject, p_fdr, _, _ = multipletests(p_vals, alpha=FDR_ALPHA, method='fdr_bh')

# convert back to NIfTI
t_map = masker.inverse_transform(t_vals)
p_map = masker.inverse_transform(reject.astype(int))  # binary mask

t_map.to_filename(OUT_T)
p_map.to_filename(OUT_P)

print(f'✓ Saved t-stat map:       {OUT_T}')
print(f'✓ Saved FDR mask (α={FDR_ALPHA}): {OUT_P}')
