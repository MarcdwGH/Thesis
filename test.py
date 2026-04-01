import pandas as pd
import numpy as np

# 1. Load the pieces
df_master = pd.read_csv('6DATASET_CLEAN_SLATE.csv')
df_sentiment = pd.read_csv('leg_sentiment_scores_with_total.csv')

# 2. Left Merge
# Legs not in df_sentiment will naturally become NaN in all sentiment columns
df_final = pd.merge(df_master, df_sentiment, on='legid', how='left')

# 3. Validation Audit (Re-running your check)
sentiment_vars = ['ACT', 'CP', 'GT', 'WYR', 'total_sentiment']
stats = []

for var in sentiment_vars:
    total = len(df_final)
    missing_count = df_final[var].isna().sum()
    valid_zero_count = (df_final[var] == 0).sum()
    active_count = ((df_final[var] != 0) & (df_final[var].notna())).sum()
    
    stats.append({
        'Variable': var,
        'Missing % (Skipped)': round(missing_count / total * 100, 2),
        'Valid 0 % (Neutral)': round(valid_zero_count / total * 100, 2),
        'Active Mention %': round(active_count / total * 100, 2)
    })

# 4. Save the corrected file
df_final.to_csv('7FINAL_DATASET.csv', index=False)

print("--- CORRECTED FEATURE AUDIT ---")
print(pd.DataFrame(stats))