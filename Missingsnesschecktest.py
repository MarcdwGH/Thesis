import pandas as pd
import numpy as np

# 1. Load the finalized training data
df = pd.read_csv('7Dataset_complete_merge.csv')

# 2. Define our engineered variables
sentiment_vars = ['ACT', 'CP', 'GT', 'WYR', 'total_sentiment']

def check_missing(series):
    # In our engineering, NaN represents a skipped survey
    # 0 represents a valid, neutral interaction
    total = len(series)
    missing_count = series.isna().sum()
    return round((missing_count / total) * 100, 2)

# 3. Generate the Report
stats = []
for var in sentiment_vars:
    stats.append({
        'Variable': var,
        'Missing % (Skipped)': check_missing(df[var]),
        'Valid 0 % (Neutral)': round(((df[var] == 0).sum() / len(df)) * 100, 2),
        'Active Mention %': round(((df[var] != 0) & (df[var].notna())).sum() / len(df) * 100, 2)
    })

audit_results = pd.DataFrame(stats)

print("--- NEW FEATURE MISSINGNESS & ACTIVITY AUDIT ---")
print(audit_results)