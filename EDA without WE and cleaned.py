import pandas as pd

# 1. Load raw data
df = pd.read_csv('legs.csv')

# 2. Apply the two "Gold Standard" filters
# Filter A: Only actual movement (Legs)
# Filter B: Only the clear 1-5 integer ratings
valid_ints = [1.0, 2.0, 3.0, 4.0, 5.0]
df_final = df[(df['class'] == 'Leg') & (df['worthwhileness_rating'].isin(valid_ints))].copy()

# 3. Clean the type
df_final['worthwhileness_rating'] = df_final['worthwhileness_rating'].astype(int)

# 4. Save this as your "Analytical Dataset"
df_final.to_csv('legs_cleaned_final.csv', index=False)

print(f"Dataset ready! Final count: {len(df_final)} legs.")