import pandas as pd

# 1. Load the training data (which already has integrated purposes)
df_train = pd.read_csv('7FINAL_DATASET.csv')

# 2. Load the original raw purpose file (the "long" format one)
# I am assuming the file name here based on our previous logic
df_purpose_raw = pd.read_csv('purposes.csv') 

# 3. Identify unique leg_ids in both
ids_in_train = set(df_train['tripid'].unique())
ids_in_purpose = set(df_purpose_raw['tripid'].unique())

# 4. Calculate Overlap
only_in_train = ids_in_train - ids_in_purpose
only_in_purpose = ids_in_purpose - ids_in_train
both = ids_in_train.intersection(ids_in_purpose)

print("--- DATASET INTEGRITY CROSS-CHECK ---")
print(f"Legs in Master Training Data: {len(ids_in_train)}")
print(f"Legs in Raw Purpose File:      {len(ids_in_purpose)}")
print("-" * 35)
print(f"✅ Legs present in BOTH:       {len(both)}")
print(f"❌ Legs in Master but NO Purpose found: {len(only_in_train)}")
print(f"⚠️ Legs in Purpose but NOT in Master:   {len(only_in_purpose)}")

# 5. Percentage of coverage
coverage = (len(both) / len(ids_in_train)) * 100
print(f"\nPurpose Coverage: {coverage:.2f}%")
