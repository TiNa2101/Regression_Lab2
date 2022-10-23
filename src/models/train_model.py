# -*- coding: utf-8 -*-
import sys
import click
import logging
import pandas as pd
import config as cfg

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.utils import save_as_pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

from catboost import CatBoostRegressor
from category_encoders.count import CountEncoder

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

real_pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
    ])

# sklearn classical linear regressors

sk_model = LinearRegression(positive=True)

sk_preprocess_pipeline = ColumnTransformer(transformers=[
    ('real_cols', real_pipeline, cfg.REAL_COLS),
    ('cat_cols', cat_pipeline, cfg.CAT_COLS),
    ])

sk_linear_regression_model = Pipeline([
    ('preprocess', sk_preprocess_pipeline),
    ('model', sk_model)
    ])

# catboost

catboost_model = CatBoostRegressor(iterations=1000,
                                   learning_rate=1,
                                   depth=2)

catboost_preprocess_pipeline = ColumnTransformer(transformers=[
    ('real_cols', real_pipeline, cfg.REAL_COLS),
    ('cat_cols', cat_pipeline, cfg.CAT_COLS),
    ('cat_bost_cols', CountEncoder(), cfg.CAT_COLS)
    ])

rscv = GridSearchCV(
    estimator=catboost_model,
    param_grid={'learning_rate': [0.03, 0.1],
                'depth': [2, 4],
                'l2_leaf_reg': [0.2, 0.5],
                'model_size_reg': [0.5, 1]},
    scoring='explained_variance',
    cv=5,
    refit=True
)

catboost_regression_model = Pipeline([
    ('preprocess', catboost_preprocess_pipeline),
    ('model', rscv)
    ])

# train

@click.command()
@click.argument('input_train_data_filepath', type=click.Path(exists=True))
@click.argument('input_train_target_filepath', type=click.Path(exists=True))
@click.argument('output_model_catboost_filepath', type=click.Path())
@click.argument('output_model_sk_linear_filepath', type=click.Path())
@click.argument('output_data_test_filepath', type=click.Path())
@click.argument('output_target_test_filepath', type=click.Path())

def main(input_train_data_filepath, input_train_target_filepath, output_model_catboost_filepath, output_model_sk_linear_filepath, output_data_test_filepath, output_target_test_filepath):
    logger = logging.getLogger(__name__)
    logger.info('training model...')

    train = pd.read_pickle(input_train_data_filepath)
    target = pd.read_pickle(input_train_target_filepath)

    train_data, val_data, train_target, val_target = train_test_split(train, target, test_size=0.4, random_state=7)

    sk_linear_regression_model.fit(train_data, train_target)
    catboost_regression_model.fit(train_data, train_target)

    save_as_pickle(sk_linear_regression_model, output_model_sk_linear_filepath)
    save_as_pickle(catboost_regression_model, output_model_catboost_filepath)

    save_as_pickle(val_data, output_data_test_filepath)
    save_as_pickle(val_target, output_target_test_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
