import pandas as pd
import numpy as np

def preprocess_motiv_data():
    print("--- Starting Data Preprocessing (With Demographics) ---")

    # 1. Load Raw Data
    print("Loading raw CSV files...")
    try:
        worth_df = pd.read_csv('worthwhileness_elements_from_trips.csv')
        legs_df = pd.read_csv('legs.csv')
        trips_df = pd.read_csv('trips.csv')
        mots_df = pd.read_csv('mots.csv')
        purposes_df = pd.read_csv('purposes.csv')
        # Load User Details
        users_df = pd.read_csv('user_details.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}. Please ensure all CSVs (including user_details.csv) are in the folder.")
        return

    # 2. Process Target Variables
    print("Processing target variables...")
    worth_wide_df = worth_df.pivot_table(
        index=['legid', 'tripid'],
        columns='worthwhileness_element',
        values='value',
        aggfunc='first'
    ).reset_index()

    target_cols = ['Enjoyment', 'Fitness', 'Paid_work', 'Personal_tasks']
    # Filter to keep only rows that have these targets
    worth_targets = worth_wide_df[['legid', 'tripid'] + target_cols]

    # 3. Process Features (Legs & Modes)
    print("Merging leg data...")
    legs_merged = legs_df.merge(mots_df[['motid', 'mot_text']], on='motid', how='left')

    # 4. Merge Targets with Leg Features
    analytic_df = worth_targets.merge(legs_merged, on=['legid', 'tripid'], how='inner')
    analytic_df.dropna(subset=target_cols, inplace=True)

    # 5. Feature Engineering
    print("Engineering features...")
    analytic_df['start_date'] = pd.to_datetime(analytic_df['start_date'])
    analytic_df['end_date'] = pd.to_datetime(analytic_df['end_date'])
    
    # Duration & Distance
    analytic_df['duration_minutes'] = (analytic_df['end_date'] - analytic_df['start_date']).dt.total_seconds() / 60
    analytic_df['log_duration_minutes'] = np.log1p(analytic_df['duration_minutes'])
    
    median_dist = analytic_df['true_distance'].median()
    analytic_df['true_distance'].fillna(median_dist, inplace=True)
    analytic_df['log_true_distance'] = np.log1p(analytic_df['true_distance'])

    # Time Features
    analytic_df['start_hour'] = analytic_df['start_date'].dt.hour
    analytic_df['is_weekend'] = analytic_df['start_date'].dt.day_name().isin(['Saturday', 'Sunday']).astype(int)

    # 6. Add Context & USER DETAILS
    print("Adding trip context and user demographics...")
    
    # Merge Trip Context
    trip_cols = ['tripid', 'average_speed', 'max_speed', 'mood_rating']
    analytic_df = analytic_df.merge(trips_df[trip_cols], on='tripid', how='left')

    # Merge Purpose
    purposes_agg = purposes_df.groupby('tripid')['purpose'].first().reset_index()
    analytic_df = analytic_df.merge(purposes_agg, on='tripid', how='left')
    analytic_df['purpose'].fillna('Unknown', inplace=True)

    # NEW: Merge User Details using the CORRECT column names
    # We use 'age_range' instead of 'age'
    user_cols_to_keep = ['userid', 'gender', 'age_range', 'education_level']
    
    # Check if columns exist before merging
    if all(col in users_df.columns for col in user_cols_to_keep):
        analytic_df = analytic_df.merge(users_df[user_cols_to_keep], on='userid', how='left')
        
        # Fill missing demographics with 'Unknown'
        for col in ['gender', 'age_range', 'education_level']:
            analytic_df[col].fillna('Unknown', inplace=True)
    else:
        print("Warning: One or more user columns not found. Skipping user merge.")

    # 7. Final Save
    print("Saving processed files...")
    
    # Define Features (X)
    feature_cols = [
        'log_duration_minutes', 'log_true_distance', 'average_speed', 'max_speed', 
        'mood_rating', 'start_hour', 'is_weekend',
        'mot_text', 'weekday', 'purpose'
    ]
    
    # Define Categorical Columns for One-Hot Encoding
    categorical_cols = ['mot_text', 'weekday', 'purpose']

    # Add user demographics if they exist in the dataframe
    if 'age_range' in analytic_df.columns:
        feature_cols += ['gender', 'age_range', 'education_level']
        # age_range is text (e.g. "25-34"), so we add it to categorical_cols
        categorical_cols += ['gender', 'age_range', 'education_level']

    X = analytic_df[feature_cols]
    
    # One-Hot Encode
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X_encoded.fillna(0, inplace=True)

    Y = analytic_df[target_cols]

    # Save to CSV
    X_encoded.to_csv('final_features.csv', index=False)
    Y.to_csv('final_targets.csv', index=False)

    print(f"--- Success! Saved features with {X_encoded.shape[1]} columns. ---")

if __name__ == "__main__":
    preprocess_motiv_data()