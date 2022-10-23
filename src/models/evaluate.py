# -*- coding: utf-8 -*-
import logging
import click
import json
import pandas as pd
import pickle
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, explained_variance_score, max_error


@click.command()
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('input_target_filepath', type=click.Path(exists=True))
@click.argument('model_catboost_filepath', type=click.Path())
@click.argument('model_sk_linear_filepath', type=click.Path())
@click.argument('output_metrics_filepath', type=click.Path())

def main(input_data_filepath, input_target_filepath, model_catboost_filepath, model_sk_linear_filepath, output_metrics_filepath):

    logger = logging.getLogger(__name__)
    logger.info('model evaluation...')

    val_data = pd.read_pickle(input_data_filepath)
    val_target = pd.read_pickle(input_target_filepath)

    sk_linear_regression_model = pickle.load(open(model_sk_linear_filepath, 'rb'))
    catboost_regression_model = pickle.load(open(model_catboost_filepath, 'rb'))

    predict_linear_regression = sk_linear_regression_model.predict(val_data)
    predict_catboost_regression = catboost_regression_model.predict(val_data)

    linear_mae = mean_absolute_error(val_target, predict_linear_regression)
    linear_mse = mean_squared_error(val_target, predict_linear_regression)
    linear_r2 = r2_score(val_target, predict_linear_regression)
    linear_evs = explained_variance_score(val_target, predict_linear_regression)
    linear_me = max_error(val_target, predict_linear_regression)
    
    catboost_mae = mean_absolute_error(val_target, predict_catboost_regression)
    catboost_mse = mean_squared_error(val_target, predict_catboost_regression)
    catboost_r2 = r2_score(val_target, predict_catboost_regression)
    catboost_evs = explained_variance_score(val_target, predict_catboost_regression)
    catboost_me = max_error(val_target, predict_catboost_regression)

    metrics = {
        "Model 1 Name": "CatBoostRegression",
        "MAE": catboost_mae,
        "MSE": catboost_mse,
        "R2": catboost_r2,
        "Explained Varience Score": catboost_evs,
        "Max Error": catboost_me,
        "Model 2 Name": "LinearRegression",
        "MAE ": linear_mae,
        "MSE ": linear_mse,
        "R2 ": linear_r2,
        "Explained Varience Score ": linear_evs,
        "Max Error ": linear_me
    }
    with open(output_metrics_filepath, "w") as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
