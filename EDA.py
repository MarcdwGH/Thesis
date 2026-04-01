import pandas as pd
import matplotlib.pyplot as plt
import os

# 2. Load the data
df = pd.read_csv('legs_cleaned_final.csv')

# 3. Categorization Logic
def categorize(val):
    if val == 1.0:  return "1.0"
    if val == 2.0:  return "2.0"
    if val == 3.0:  return "3.0"
    if val == 4.0:  return "4.0"
    if val == 5.0:  return "5.0"
    
df['cat'] = df['worthwhileness_rating'].apply(categorize)

# 4. Calculate Counts and Percentages
order = ["1.0", "2.0", "3.0", "4.0", "5.0"]
counts = df['cat'].value_counts().reindex(order)
percentages = (counts / len(df)) * 100

# 5. Create the Plot
plt.figure(figsize=(12, 7))
colors = ['#95a5a6', '#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60', '#3498db']
bars = plt.bar(counts.index, counts.values, color=colors, edgecolor='black', alpha=0.8)

# Add counts and percentages on top of bars
for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1000, 
             f'{int(yval):,}\n({percentages[i]:.1f}%)', 
             ha='center', va='bottom', fontweight='bold')

plt.title('Overview of Worthwhileness Rating Distribution', fontsize=15, pad=20)
plt.ylabel('Number of Trip Legs', fontsize=12)
plt.xlabel('User Rating Category', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.ylim(0, max(counts.values) * 1.15) # Leave space for labels
plt.tight_layout()

# Save for thesis
plt.savefig('worthwhileness_overview_cleaned.png', dpi=300)
plt.show()

print("Graph saved as 'worthwhileness_overview_cleaned.png'")
