import pandas as pd

# 1. Load the clinical metadata
clin = pd.read_csv('participants.tsv', sep='\t')

# 2. Load your lesion+AG data
les = pd.read_csv('lesion_volumes.csv')

# 3. Rename for a common key
if 'participant_id' in clin.columns:
    clin = clin.rename(columns={'participant_id':'subject'})

# 4. Merge on 'subject' (inner join keeps only stroke subjects with lesion data)
merged = pd.merge(clin, les, on='subject', how='inner')

# 5. Write out the combined table
merged.to_csv('participants_lesion_merged.csv', index=False)
print(f"✓ participants_lesion_merged.csv written; {len(merged)} rows")
