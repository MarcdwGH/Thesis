import logging
import pandas as pd
import numpy as np
import json
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import uniform, randint




def boosted_regressor_model():
    logging.info("Reading train and test files")
    train = pd.read_json("train.json", orient='records')
    test = pd.read_json("test.json", orient='records')

    

    train['rooms_x_rating'] = train['rooms'] * train['rating']
    train['num_reviews_x_rating'] = train['num_reviews'] * train['rating']
    train['lat_x_lon'] = train['lat'] * train['lon']

    test['rooms_x_rating'] = test['rooms'] * test['rating']
    test['num_reviews_x_rating'] = test['num_reviews'] * test['rating']
    test['lat_x_lon'] = test['lat'] * test['lon']

    train, valid = train_test_split(train, test_size=1/3, random_state=123)

    
    numeric_features = [
        'lat', 'lon', 'rooms', 'bathrooms', 'beds',
        'min_nights', 'num_reviews', 'rating', 'guests',
        'rooms_x_rating', 'num_reviews_x_rating', 'lat_x_lon'
    ]
    categorical_features = ['room_type']

    # Preprocessing steps
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency = 10))
    ])

    preprocess = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    label = 'revenue'

    dummy = make_pipeline(preprocess, DummyRegressor())

    base_boosted = make_pipeline(preprocess, GradientBoostingRegressor(random_state=123))
   

    param_dist = {
        'gradientboostingregressor__n_estimators': randint(100, 800),
        'gradientboostingregressor__max_depth': randint(3, 10),
        'gradientboostingregressor__learning_rate': uniform(0.01, 0.3),
        'gradientboostingregressor__subsample': uniform(0.7, 0.3),
    }

    search = RandomizedSearchCV(
        base_boosted,
        param_distributions=param_dist,
        n_iter=20,
        scoring='neg_mean_absolute_error',
        cv=3,
        verbose=1,
        random_state=123,
        n_jobs=-1
    )

    for model_name, model in [("mean", dummy), ("boosted", search)]:
        logging.info(f"Fitting model {model_name}")
        model.fit(train.drop([label], axis=1), np.log1p(train[label].values))

        # If boosted model, pull best estimator
        if model_name == "boosted":
            model = model.best_estimator_
            logging.info(f"Best hyperparameters: {search.best_params_}")

        for split_name, split in [("train", train), ("valid", valid)]:
            pred = np.expm1(model.predict(split.drop([label], axis=1)))
            mae = mean_absolute_error(split[label], pred)
            logging.info(f"{model_name} {split_name} {mae:.3f}")

    # Predict on test set using best model
    pred_test = np.expm1(search.best_estimator_.predict(test))
    test[label] = pred_test

    predicted = test[['revenue']].to_dict(orient='records')
    with zipfile.ZipFile("boosted2.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("predicted.json", json.dumps(predicted, indent=2))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    boosted_regressor_model()




