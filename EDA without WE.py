import pandas as pd
import matplotlib.pyplot as plt
import os

# 2. Load the data
df = pd.read_csv('legs.csv')

# 3. APPLY THE CLASS FILTER FIRST
# We only want actual travel (Legs), not stationary time (Waiting)
df_legs_only = df[df['class'] == 'Leg'].copy()

# 4. Categorization Logic
def categorize(val):
    if val == -1.0: return "-1.0"
    if val == 1.0:  return "1.0"
    if val == 2.0:  return "2.0"
    if val == 3.0:  return "3.0"
    if val == 4.0:  return "4.0"
    if val == 5.0:  return "5.0"
    return "Other"

df_legs_only['cat'] = df_legs_only['worthwhileness_rating'].apply(categorize)

# 5. Calculate Counts
order = ["-1.0", "1.0", "2.0", "3.0", "4.0", "5.0", "Other"]
counts = df_legs_only['cat'].value_counts().reindex(order)
percentages = (counts / len(df_legs_only)) * 100

# 6. Create the Plot
plt.figure(figsize=(12, 7))
colors = ['#95a5a6', '#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60', '#3498db']
bars = plt.bar(counts.index, counts.values, color=colors, edgecolor='black', alpha=0.8)

# Add counts and percentages on top
for i, bar in enumerate(bars):
    yval = bar.get_height()
    if yval > 0: # Only label bars that have data
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(counts.values)*0.01), 
                 f'{int(yval):,}\n({percentages[i]:.1f}%)', 
                 ha='center', va='bottom', fontweight='bold')

plt.title(f'Worthwhileness Distribution: "Leg" Class Only (N={len(df_legs_only):,})', fontsize=15, pad=20)
plt.ylabel('Number of Trip Legs', fontsize=12)
plt.xlabel('User Rating Category', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.ylim(0, max(counts.values) * 1.15)
plt.tight_layout()

plt.savefig('worthwhileness_legs_only.png', dpi=300)
plt.show()

print(f"Filtered out {len(df) - len(df_legs_only)} 'Waiting' events.")
print(f"New total rows: {len(df_legs_only)}")