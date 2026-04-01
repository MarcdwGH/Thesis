import pandas as pd

# 1. Load the raw experience factors
# Assumes columns: legid, factor, type, minus, plus
df_factors = pd.read_csv('experience_factors.csv')

# 2. Calculate the Net Contribution per row
# If plus=1 and minus=0, value is 1. If plus=0 and minus=1, value is -1.
df_factors['net_score'] = df_factors['plus'] - df_factors['minus']

# 3. Pivot to "Wide" format
# We sum the net_scores for each category (type) per leg
df_sentiment_matrix = df_factors.pivot_table(
    index='legid', 
    columns='type', 
    values='net_score', 
    aggfunc='sum'
).fillna(0)

# 4. Convert to whole integers (since counts are whole numbers)
df_sentiment_matrix = df_sentiment_matrix.astype(int)

# 5. Reset index to make legid a column again
df_sentiment_matrix = df_sentiment_matrix.reset_index()

# 6. Save this as your "Sentiment Gold Standard" file
df_sentiment_matrix.to_csv('leg_sentiment_scores.csv', index=False)

print("--- STANDALONE SENTIMENT MATRIX CREATED ---")
print(f"Processed {len(df_sentiment_matrix)} unique legs that provided feedback.")
print("\nFirst 5 rows of Net Category Scores:")
print(df_sentiment_matrix.head())