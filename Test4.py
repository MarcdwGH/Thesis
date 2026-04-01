import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_sample_weight

# 1. Load your data
df = pd.read_csv('9FINAL_DATASET_TRANSLATED.csv')
target = df['worthwhileness_rating'].dropna()

# 2. Setup the Plot
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid", {'axes.spines.top': False, 'axes.spines.right': False})

# Calculate counts and percentages
counts = target.value_counts().sort_index()
percentages = (counts / len(target)) * 100

# 3. Create Bar Chart
palette = sns.color_palette("viridis", len(counts))
ax = sns.barplot(x=counts.index.astype(int), y=counts.values, palette=palette, edgecolor='black', linewidth=1)

# 4. Add "Balanced Impact" Annotations
# We show how many 'extra' importance points the minority classes get
weights = compute_sample_weight(class_weight='balanced', y=target)
unique_weights = pd.Series(weights, index=target).drop_duplicates().sort_index()

# 5. Styling & Labels
plt.title('Frequency Distribution of Worthwhileness Ratings\n(MoTiV Dataset)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Worthwhileness Rating (Stars)', fontsize=13, labelpad=10)
plt.ylabel('Number of Trip Samples', fontsize=13, labelpad=10)

# Add Data Labels on top of bars
for i, p in enumerate(ax.patches):
    height = p.get_height()
    rating = counts.index[i]
    weight_factor = unique_weights.iloc[i]
    
    # Label: Count + Percentage
    ax.text(p.get_x() + p.get_width()/2., height + (max(counts)*0.02),
            f'{int(height)}\n({percentages.iloc[i]:.1f}%)',
            ha="center", fontsize=11, fontweight='bold')
    
    # Optional: Small label showing the "Balanced Weight" impact
    ax.text(p.get_x() + p.get_width()/2., height / 2,
            f'Weight:\n{weight_factor:.2f}x',
            ha="center", color='white', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('target_distribution_pro.png', dpi=300)
plt.show()

print("Visualization saved as 'target_distribution_pro.png'")