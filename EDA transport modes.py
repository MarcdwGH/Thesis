import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your clean data
df = pd.read_csv('legs_cleaned_final.csv')

# 2. Count the observations per mode
mode_counts = df['transport_category'].value_counts()
print("--- Observations per Transport Mode ---")
print(mode_counts)

# 3. Visualization: Mode Frequency and Average Rating
fig, ax1 = plt.subplots(figsize=(14, 7))

# Bar Plot for Frequency
sns.barplot(x=mode_counts.index, y=mode_counts.values, alpha=0.6, ax=ax1, color='grey')
ax1.set_ylabel('Number of Legs (Frequency)', fontsize=12)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

# Line Plot for Average Worthwhileness (Dual Axis)
ax2 = ax1.twinx()
sns.pointplot(x=mode_counts.index, y=df.groupby('transport_category')['worthwhileness_rating'].mean().reindex(mode_counts.index), 
              color='darkred', ax=ax2)
ax2.set_ylabel('Average Worthwhileness (1-5)', color='darkred', fontsize=12)
ax2.set_ylim(1, 5)

plt.title('Transport Mode Frequency vs. Mean Worthwhileness', fontsize=15)
plt.tight_layout()
plt.show()