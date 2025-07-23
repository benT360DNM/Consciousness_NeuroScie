import pandas as pd, numpy as np, json

THRESH      = 50          # AG voxel cutoff
OUTCOME_COL = 'nihss'     # <-- matches participants.tsv

df = pd.read_csv('participants_lesion_merged.csv').dropna(subset=[OUTCOME_COL])

ag   = df[df.ag_voxels >= THRESH][OUTCOME_COL].values
ctrl = df[df.ag_voxels == 0][OUTCOME_COL].values

def cohen_d(a, b):
    return (a.mean() - b.mean()) / np.sqrt(((a.var(ddof=1) + b.var(ddof=1)) / 2))

rng  = np.random.default_rng(42)
boots = [cohen_d(rng.choice(ag,   size=ag.size,   replace=True),
                 rng.choice(ctrl, size=ctrl.size, replace=True))
         for _ in range(10_000)]

res = dict(
    cohen_d_mean = float(np.mean(boots)),
    ci_lower     = float(np.percentile(boots,  2.5)),
    ci_upper     = float(np.percentile(boots, 97.5)),
    n_ag   = int(ag.size),
    n_ctrl = int(ctrl.size),
    ag_cut = THRESH,
    outcome = OUTCOME_COL
)

with open('bootstrap_results.json','w') as f: json.dump(res, f, indent=2)
print("✓ bootstrap_results.json written:")
print(json.dumps(res, indent=2))
