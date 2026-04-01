import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
print("Loading data...")

# Load Datasets
worth_df = pd.read_csv('worthwhileness_elements_from_trips.csv')
legs_df = pd.read_csv('legs.csv')
trips_df = pd.read_csv('trips.csv')
mots_df = pd.read_csv('mots.csv')
purposes_df = pd.read_csv('purposes.csv')

# --- A. Prepare Target Variables (Worthwhileness) ---
# Pivot to wide format so we have columns: Enjoyment, Fitness, etc.
worth_wide_df = worth_df.pivot_table(
    index=['legid', 'tripid'],
    columns='worthwhileness_element',
    values='value',
    aggfunc='first'
).reset_index()

# Filter for the main targets
target_cols = ['Enjoyment', 'Fitness', 'Paid_work', 'Personal_tasks']
worth_targets = worth_wide_df[['legid', 'tripid'] + target_cols]

# --- B. Prepare Features (Legs & Trips) ---
# Merge Legs with Transport Mode text
legs_merged = legs_df.merge(mots_df[['motid', 'mot_text']], on='motid', how='left')

# Select relevant leg columns
legs_features = legs_merged[['legid', 'tripid', 'start_date', 'end_date', 
                             'true_distance', 'leg_distance', 'weekday', 'weekday_class', 'mot_text']]

# Merge Targets with Leg Features
analytic_df = worth_targets.merge(legs_features, on=['legid', 'tripid'], how='inner')

# Drop rows where targets are missing
analytic_df.dropna(subset=target_cols, inplace=True)

# --- C. Feature Engineering ---
# Convert dates to datetime
analytic_df['start_date'] = pd.to_datetime(analytic_df['start_date'])
analytic_df['end_date'] = pd.to_datetime(analytic_df['end_date'])

# Calculate Duration (in minutes)
analytic_df['duration_minutes'] = (analytic_df['end_date'] - analytic_df['start_date']).dt.total_seconds() / 60

# Log Transformations (to handle skewed data)
analytic_df['log_duration_minutes'] = np.log1p(analytic_df['duration_minutes'])

# Impute missing distances with median
median_dist = analytic_df['true_distance'].median()
analytic_df['true_distance'].fillna(median_dist, inplace=True)
analytic_df['log_true_distance'] = np.log1p(analytic_df['true_distance'])

# Extract Time Features
analytic_df['start_hour'] = analytic_df['start_date'].dt.hour
analytic_df['is_weekend'] = analytic_df['start_date'].dt.day_name().isin(['Saturday', 'Sunday']).astype(int)

# --- D. Merge Trip & Purpose Context ---
# Get Trip level features (Speed, Mood)
trip_features = trips_df[['tripid', 'average_speed', 'max_speed', 'mood_rating']]
analytic_df = analytic_df.merge(trip_features, on='tripid', how='left')

# Get Purpose (Taking the first purpose listed for the trip)
purposes_agg = purposes_df.groupby('tripid')['purpose'].first().reset_index()
analytic_df = analytic_df.merge(purposes_agg, on='tripid', how='left')
analytic_df['purpose'].fillna('Unknown', inplace=True)

# ==========================================
# 2. PREPARE FOR MACHINE LEARNING
# ==========================================
print("Encoding features...")

# Define Features (X) and Targets (Y)
feature_columns = [
    'log_duration_minutes', 'log_true_distance', 'average_speed', 'max_speed', 
    'mood_rating', 'start_hour', 'is_weekend',
    'mot_text', 'weekday', 'purpose' # Categorical columns to be encoded
]

X = analytic_df[feature_columns]
Y = analytic_df[target_cols]

# One-Hot Encoding (Convert text categories to numbers)
X_encoded = pd.get_dummies(X, columns=['mot_text', 'weekday', 'purpose'], drop_first=True)

# Handle any remaining missing values in features (simple fill with 0 or median)
X_encoded.fillna(0, inplace=True)

# Train/Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=42)

# ==========================================
# 3. TRAIN MODEL
# ==========================================
print("Training Multi-Target Model...")

# We use a MultiOutputRegressor which fits one Regressor per target variable
# You can swap GradientBoostingRegressor for XGBRegressor if you install xgboost
model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
model.fit(X_train, Y_train)

# ==========================================
# 4. EVALUATION
# ==========================================
print("Evaluating Model...")
Y_pred = model.predict(X_test)

print("\n--- Results ---")
for i, col in enumerate(target_cols):
    rmse = np.sqrt(mean_squared_error(Y_test.iloc[:, i], Y_pred[:, i]))
    r2 = r2_score(Y_test.iloc[:, i], Y_pred[:, i])
    print(f"Target: {col:15} | RMSE: {rmse:.4f} | R2 Score: {r2:.4f}")

# ==========================================
# 5. PLOT FEATURE IMPORTANCE
# ==========================================
print("\nGenerating Feature Importance Plot...")
feature_names = X_encoded.columns
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, target in enumerate(target_cols):
    # Extract importance from the specific estimator for this target
    importances = model.estimators_[i].feature_importances_
    indices = np.argsort(importances)[-10:] # Top 10 features
    
    ax = axes[i]
    ax.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[j] for j in indices])
    ax.set_title(f'Top Predictors for {target}')

plt.tight_layout()
plt.show()