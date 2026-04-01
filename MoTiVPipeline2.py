import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. LOAD DATA
df = pd.read_csv('9FINAL_DATASET_TRANSLATED.csv')
df = df.dropna(subset=['worthwhileness_rating'])

# 2. FEATURE ENGINEERING & SELECTION. TAKE OUT NON RELEVANT FEATURES

X = df.drop(columns=[
    'legid', 'userid', 'tripid', 'motid', 'class', 
    'sentiment_skipped', 'worthwhileness_rating',
    'mood_rating', 'total_sentiment'
], errors='ignore')
y = df['worthwhileness_rating']

act_features = [col for col in X.columns if col.startswith('act_')]
purp_features = [col for col in X.columns if col.startswith('purp_')]

numerical_features = [
    'leg_distance', 'leg_duration', 'ACT', 'CP', 'GT', 'WYR',
    'activity_count', 'purpose_count', 'did_you_have_to_arrive'
] + act_features + purp_features

categorical_features = [
    'transport_category', 'weekday', 'gender', 'age_range', 'city', 'country', 
    'education_level', 'marital_status_household', 'labour_status_household', 
    'temperature_category', 'weather_group', 'number_people_household', 
    'years_of_residence_household'
]

# Ensure types are correct 
for col in categorical_features:
    if col in X.columns:
        X[col] = X[col].astype('category')

# 3. SPLIT & WEIGHTING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate weights to balance the classes (fixes the "always predict 4" bias)
weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 4. TRAIN XGBOOST MODEL. (POISSONG was reccomended)
print("Training Balanced XGBoost Model...")
xgb_model = xgb.XGBRegressor(
    objective='count:poisson', 
    tree_method="hist", 
    enable_categorical=True,
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6
)
xgb_model.fit(X_train, y_train, sample_weight=weights)

# 5. TRAIN LINEAR BASELINE (RIDGE)
print("Training Linear Baseline (Ridge)...")
valid_num = [c for c in numerical_features if c in X_train.columns]
valid_cat = [c for c in categorical_features if c in X_train.columns]

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), valid_num),
    ('cat', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), valid_cat)
])

linear_pipe = Pipeline([('pre', preprocessor), ('reg', Ridge(alpha=1.0))])
linear_pipe.fit(X_train, y_train, reg__sample_weight=weights)

# 6. EVALUATION
xgb_preds = xgb_model.predict(X_test)
lin_preds = linear_pipe.predict(X_test)
dummy_preds = np.full_like(y_test, y_train.mean())

print("\n" + "="*40)
print("       FINAL MODEL COMPARISON")
print("="*40)
print(f"{'Model':<18} | {'MAE':<8} | {'R2':<8}")
print("-" * 40)
print(f"{'Dummy (Mean)':<18} | {mean_absolute_error(y_test, dummy_preds):<8.4f} | {0.0:<8.4f}")
print(f"{'Linear (Ridge)':<18} | {mean_absolute_error(y_test, lin_preds):<8.4f} | {r2_score(y_test, lin_preds):<8.4f}")
print(f"{'XGBoost (Final)':<18} | {mean_absolute_error(y_test, xgb_preds):<8.4f} | {r2_score(y_test, xgb_preds):<8.4f}")

# 7. FEATURE IMPORTANCE (XGBOOST)
importance = xgb_model.get_booster().get_score(importance_type='gain')
sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

print("\n--- TOP 15 DRIVERS (XGBOOST GAIN) ---")
for feature, score in sorted_imp[:15]:
    print(f"{feature:<30} | {score:>10.4f}")

# 8. VISUALIZATION (CONFUSION MATRIX)
y_pred_rounded = np.clip(np.round(xgb_preds), 1, 5)
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rounded, cmap='Greens', normalize='true', ax=ax)
plt.title('Balanced XGBoost Confusion Matrix (Sensitivity Improved)')
plt.show()

#Hi