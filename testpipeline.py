import pandas as pd
import numpy as np

# 1. Load the data
df = pd.read_csv('8FINAL_DATASET.csv')

activity_cols = [col for col in df.columns if col.startswith('act_')]

# 2. Map the MoTiV Logic
# True -> 1
# False -> 0
# 0 -> NaN (Missing)
def translate_activity(val):
    v = str(val).lower().strip()
    if v == 'true':
        return 1
    elif v == 'false':
        return 0
    elif v == '0' or v == '0.0':
        return np.nan
    else:
        return np.nan # Catch-all for any other weirdness

print("Translating Activity codes (True/False/0)...")
for col in activity_cols:
    df[col] = df[col].apply(translate_activity)
    # Cast to nullable Int64 so NaNs stay as <NA>
    df[col] = df[col].astype('Int64')

# 3. Quality Report
total_rows = len(df)
print(f"\n{'Feature':<20} | {'Missing (%)':<15} | {'Confirmed No (0)':<20}")
print("-" * 65)

for col in ['act_Walking', 'act_Talking', 'act_Browsing']:
    if col in df.columns:
        null_pct = (df[col].isna().sum() / total_rows) * 100
        zero_count = (df[col] == 0).sum()
        one_count = (df[col] == 1).sum()
        print(f"{col:<20} | {null_pct:>10.2f}% | {zero_count:>15} (Yes: {one_count})")

# 4. Save
df.to_csv('9FINAL_DATASET_TRANSLATED.csv', index=False)