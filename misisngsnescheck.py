import pandas as pd
import numpy as np

# 1. Load the data with 'keep_default_na=True' to catch standard blanks
df = pd.read_csv('9FINAL_DATASET_TRANSLATED.csv', skipinitialspace=True)

# Clean up column names immediately to remove hidden spaces/newlines
df.columns = df.columns.str.strip()

# 2. Define our "Hidden Missing" indicators
categorical_cols = [
    'gender', 'age_range', 'city', 'country', 'education_level', 
    'marital_status_household', 'labour_status_household', 
    'years_of_residence_household', 'temperature_category', 'weather_group'
]
unknown_strings = ['unknown', 'none', 'nan', 'null', '', ' ']
rating_cols = ['mood_rating', 'did_you_have_to_arrive']

# 3. Comprehensive Audit
report = []
total_rows = len(df)

for col in df.columns:
    # A: Count actual NaNs AND empty strings
    # This is the primary check for sentiment pillars and activities
    is_blank = df[col].isna() | (df[col].astype(str).str.strip() == '')
    missing_count = is_blank.sum()
    
    # B: Check for demographic "Unknown" labels
    if col in categorical_cols:
        is_unknown = df[col].astype(str).str.lower().str.strip().isin(unknown_strings)
        # Add them only if they weren't already counted as blank
        missing_count += (is_unknown & ~is_blank).sum()

    # C: Check for numeric "-1" labels
    elif col in rating_cols:
        is_minus_one = (df[col] == -1)
        missing_count += (is_minus_one & ~is_blank).sum()

    missing_pct = (missing_count / total_rows) * 100
    
    report.append({
        'Feature': col,
        'Missing_Count': int(missing_count),
        'Missing_%': round(missing_pct, 2)
    })

# 4. Display EVERYTHING
quality_report = pd.DataFrame(report).sort_values(by='Missing_%', ascending=False)

# Force Pandas to show every single row in the printout
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("--- COMPLETE DATA QUALITY AUDIT (ALL FEATURES) ---")
print(quality_report.to_string(index=False))