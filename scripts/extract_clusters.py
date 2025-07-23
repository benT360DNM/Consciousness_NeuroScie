import numpy as np, pandas as pd
import nibabel as nib
from nilearn.regions import connected_regions

# Input files
tmap_file = 'vlsm_tstat.nii.gz'
pmask_file = 'vlsm_pval_fdr05.nii.gz'
out_csv = 'vlsm_clusters_summary.csv'

# Load images
timg = nib.load(tmap_file)
pimg = nib.load(pmask_file)

# Extract connected regions from the binary mask
# connected_regions returns (cluster_imgs, labels), where labels[i]=voxel count
clusters, labels = connected_regions(
    pimg, min_region_size=10  # drop tiny speckles <10 voxels
)

rows = []
for idx, (clust_img, size) in enumerate(zip(clusters, labels), start=1):
    data = clust_img.get_fdata().astype(bool)
    # mask the t-map by this cluster
    tvals = timg.get_fdata()[data]
    peak_t = np.nanmax(np.abs(tvals))
    rows.append({
        'cluster_id': idx,
        'voxel_count': int(size),
        'peak_t': float(peak_t)
    })

# Save summary
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f'✓ {out_csv} written; {len(rows)} clusters')
