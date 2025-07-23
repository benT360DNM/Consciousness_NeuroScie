import glob, os, numpy as np, pandas as pd
import nibabel as nib
from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiMasker
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# ----- PARAMETERS -----
CSV       = 'participants_lesion_merged.csv'
MASK_GLOB = 'derivatives/lesion_masks/**/*_mask.nii.gz'
OUT_T     = 'vlsm_tstat.nii.gz'
OUT_P     = 'vlsm_pval_fdr05.nii.gz'
FDR_ALPHA = 0.05
# ----------------------

print('1/8 • Loading clinical table …')
df = pd.read_csv(CSV)

print('2/8 • Loading MNI152 template & brain mask …')
template = load_mni152_template(resolution=2)
brain_m  = load_mni152_brain_mask(resolution=2)

print('3/8 • Gathering all lesion paths …')
all_paths = sorted(glob.glob(MASK_GLOB, recursive=True))
print(f'   → Found {len(all_paths)} mask files')

print('4/8 • Filtering to participants in clinical table …')
subjects, paths = [], []
for p in all_paths:
    subj = os.path.basename(p).split('_')[0]
    if subj in df.subject.values:
        subjects.append(subj)
        paths.append(p)
print(f'   → Kept {len(paths)} masks matching subjects')

print('5/8 • Deduplicating so one mask per subject …')
seen = set()
uniq_subj, uniq_paths = [], []
for subj, p in zip(subjects, paths):
    if subj not in seen:
        seen.add(subj)
        uniq_subj.append(subj)
        uniq_paths.append(p)
subjects, paths = uniq_subj, uniq_paths
n_subj = len(subjects)
print(f'   → {n_subj} unique subjects')

print('6/8 • Setting up masker …')
masker = NiftiMasker(mask_img=brain_m)
masker.fit()
n_vox = int(masker.mask_img_.get_fdata().sum())
print(f'   → Brain mask = {n_vox} voxels')

print('7/8 • Streaming lesion masks into design matrix …')
X = np.zeros((n_subj, n_vox), dtype=bool)
for i, p in enumerate(tqdm(paths)):
    img = nib.load(p)
    img2 = resample_to_img(img, template, interpolation='nearest')
    X[i] = masker.transform(img2).ravel() > 0

print('8/8 • Aligning NIHSS & running t-tests …')
# Build outcome vector, one per subject
y = df.drop_duplicates('subject').set_index('subject').loc[subjects, 'nihss'].values

# Pre-allocate results
t_vals = np.full(n_vox, np.nan)
p_vals = np.full(n_vox, np.nan)

for j in tqdm(range(n_vox)):
    g1 = y[X[:, j]]       # lesion at voxel j
    g2 = y[~X[:, j]]      # no lesion at voxel j
    if len(g1) < 2 or len(g2) < 2:
        continue
    t_vals[j], p_vals[j] = ttest_ind(g1, g2, equal_var=False, nan_policy='omit')

# FDR correction
reject, _, _, _ = multipletests(p_vals, alpha=FDR_ALPHA, method='fdr_bh')

# Save NIfTIs
t_img = masker.inverse_transform(t_vals)
p_img = masker.inverse_transform(reject.astype(int))
t_img.to_filename(OUT_T)
p_img.to_filename(OUT_P)

print(f'✓ t-stat map:      {OUT_T}')
print(f'✓ FDR mask (α={FDR_ALPHA}): {OUT_P}')
