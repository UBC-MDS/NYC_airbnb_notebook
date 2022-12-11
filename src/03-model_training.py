# Date 2022-12-10

'''This script takes the training data set and returns the optimized models to be tested in .joblib formats.
The models will be exported in the directory where this script is found.
Tables for cross-validation results will be created in the specified directory.

Usage:
03-model_training.py --data_path=<data_path> --output_path_cv=<output_path_cv>

Options:
--data_path=<data_path>     Path to the data file (including the file name).
--output_path_cv=<output_path_cv> Desired path for the performance results returned.

Example:
python src/03-model_training.py --data_path='data/processed/train.csv' --output_path_cv='results'
'''

from docopt import docopt
import numpy as np
import pandas as pd
import os
from joblib import dump, load
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning

import altair as alt
alt.renderers.enable('mimetype')
import vl_convert as vlc
# alt.data_transformers.enable('data_server')

import shutup
shutup.please()

opt = docopt(__doc__)
def main( data_path, output_path):
    data_full = pd.read_csv( data_path)

    numerical_feats = ['latitude', 'longitude', 'minimum_nights', 'days_since_last_review', 'availability_365']
    log_numerical_feats = ['price', 'number_of_reviews', 'calculated_host_listings_count']
    categorical_feats = ['neighbourhood_group', 'room_type']
    drop_feats = ['neighbourhood', 'host_id', 'id', 'name', 'host_name', 'last_review']

    preprocessor = make_column_transformer(
        (FunctionTransformer(helper_functions.log_transformer), log_numerical_feats),
        (StandardScaler(), numerical_feats + log_numerical_feats),
        (OneHotEncoder(handle_unknown='ignore'), categorical_feats),
        ('drop', drop_feats)
    )
    cross_val_results = {}

    pipe_linear = make_pipeline(
        preprocessor,
        LinearRegression()
    )

    pipe_decisiontree = make_pipeline(
        preprocessor,
        DecisionTreeRegressor()
    )

    cross_val_results["Base-Linear"] = pd.DataFrame(cross_validate(
        pipe_linear,
        X_train,
        y_train,
        cv=10,
        scoring="r2",
        return_train_score=True,
        error_score='raise'
    )).agg(['mean','std']).round(3).T

    cross_val_results["Base-Tree"] = pd.DataFrame(cross_validate(
        pipe_decisiontree,
        X_train,
        y_train,
        cv=10,
        scoring="r2",
        return_train_score=True,
        error_score='raise'
    )).agg(['mean','std']).round(3).T

    pd.concat(cross_val_results, axis=1)
    
    pipe_ridge = make_pipeline(
    preprocessor,
    RidgeCV()
    )

    cross_val_results["Ridge"] = pd.DataFrame(cross_validate(
        pipe_ridge,
        X_train,
        y_train,
        cv=10,
        scoring="r2",
        return_train_score=True,
        error_score='raise'
    )).agg(['mean','std']).round(3).T

    pd.concat(cross_val_results, axis=1)
    
    # create pipeline for Random forest regressor
    pipe_rf = make_pipeline(
        preprocessor,
        RandomForestRegressor(random_state=123)
    )

    # create pipeline for SVC
    pipe_svr = make_pipeline(
        preprocessor,
        SVR()
    )

    # create pipeline for KNN regressor
    pipe_knn = make_pipeline(
        preprocessor,
        KNeighborsRegressor(n_jobs=-1)
    )

    # cross validation for Randomforest
    cross_val_results["Randomforest"] = pd.DataFrame(cross_validate(
        pipe_rf,
        X_train,
        y_train,
        cv=10,
        n_jobs=-1,
        scoring="r2",
        return_train_score=True,
    )).agg(['mean','std']).round(3).T

    # cross validation for KNN
    cross_val_results["KNN"] = pd.DataFrame(cross_validate(
        pipe_knn,
        X_train,
        y_train,
        cv=10,
        n_jobs=-1,
        scoring="r2",
        return_train_score=True,
    )).agg(['mean','std']).round(3).T

    # cross validation for SVR
    cross_val_results["SVR"] = pd.DataFrame(cross_validate(
        pipe_svr,
        X_train,
        y_train,
        cv=10,
        n_jobs=-1,
        scoring="r2",
        return_train_score=True,
    )).agg(['mean','std']).round(3).T

    pd.concat(cross_val_results, axis=1)


if __name__ == "__main__":
  main(opt["--data_path"], opt["--output_path_cv"])
