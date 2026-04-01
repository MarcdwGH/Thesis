
import logging
import pandas as pd
import numpy as np
import json
import zipfile
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



#load Data
df = pd.read_csv('9FINAL_DATASET_TRANSLATED.csv')

#Get rid of identifiers that arent needed to train the model. Y=target
X = df.drop(columns=['legid', 'userid', 'tripid', 'motid', 'class', 
                     'sentiment_skipped', 'worthwhileness_rating','mood_rating','total_sentiment'], errors='ignore')
Y = df['worthwhileness_rating']


# 3. Numerical Features 
act_features = [col for col in X.columns if col.startswith('act_')]
purp_features = [col for col in X.columns if col.startswith('purp_')]

# Update your Numerical Features list
numerical_features = [
    'leg_distance', 'leg_duration',
    'ACT', 'CP', 'GT', 'WYR',
    'activity_count', 'purpose_count', 'did_you_have_to_arrive', 'weekday_class'
] + act_features + purp_features

categorical_features = [
    'transport_category', 'weekday', 'gender', 
    'age_range', 'city', 'country', 'education_level', 
    'marital_status_household', 'labour_status_household', 
    'temperature_category', 'weather_group', 'number_people_household', 'years_of_residence_household'
]
      
for col in categorical_features:
    X[col] = X[col].astype('category')     

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 5. Model (No weights, focused on MAE)
model = xgb.XGBRegressor(
    objective='reg:absoluteerror', 
    tree_method="hist", 
    enable_categorical=True,
    n_estimators=300,
    learning_rate=0.05
)

model.fit(X_train, y_train)

# 6. Evaluation
preds = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, preds):.4f}")
print(f"R2: {r2_score(y_test, preds):.4f}")

baseline_mae = mean_absolute_error(y_test, [y_train.mean()] * len(y_test))
print(f"Baseline MAE (Predicting the Average): {baseline_mae:.4f}")
# Run this to see your final rankings


# 7. Print Feature Importance to Terminal
importance = model.get_booster().get_score(importance_type='gain')

# Sort importance dictionary by value (descending)
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

print("\n--- FEATURE IMPORTANCE (Ranked by Gain) ---")
print(f"{'Feature':<30} | {'Gain Score':<10}")
print("-" * 45)
for feature, score in sorted_importance:
    print(f"{feature:<30} | {score:>10.4f}")


# 1. Feature Correlation (Top 10 most important + Sentiment)
# We focus on the big hitters to keep the heatmap readable
# 1. Feature Correlation (Numerical Only)
# We filter top_features to ensure only numbers are passed to .corr()
numerical_top_features = [
    'worthwhileness_rating', 'leg_distance', 
    'leg_duration', 'activity_count', 'ACT', 'CP', 'GT', 'WYR'
]

# Ensure only columns that exist in df and are numeric are used
valid_numeric_cols = [col for col in numerical_top_features if col in df.columns]

plt.figure(figsize=(12, 8))
# Setting numeric_only=True is the key safety switch here
sns.heatmap(df[valid_numeric_cols].corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Numerical Feature Correlation Heatmap')
plt.show()

# 2. Confusion Matrix (This part was likely fine, but here it is for context)
y_pred_rounded = np.clip(np.round(preds), 1, 5)
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rounded, 
                                        cmap='Blues', 
                                        normalize='true', 
                                        ax=ax)
plt.title('Confusion Matrix: Predicted vs Actual Worthwhileness')
plt.show()

# 2. Confusion Matrix
# Since Worthwhileness is 1, 2, 3, 4, 5, we round the predictions
y_pred_rounded = np.clip(np.round(preds), 1, 5)

fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rounded, 
                                        cmap='Blues', 
                                        normalize='true', # Shows percentages
                                        ax=ax)
plt.title('Confusion Matrix: Predicted vs Actual Worthwhileness')
plt.show()
