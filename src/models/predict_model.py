# -*- coding: utf-8 -*-
import click
import logging
import pickle
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.utils import save_as_pickle
from src.data.preprocess import preprocess_data


@click.command()
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('input_model_filepath', type=click.Path(exists=True))
@click.argument('output_predict_filepath', type=click.Path())

def main(input_data_filepath, input_model_filepath, output_predict_filepath):

    logger = logging.getLogger(__name__)
    logger.info('model inference...')

    df = pd.read_csv(input_data_filepath)
    df = preprocess_data(df)

    catboost_regression_model = pickle.load(open(input_model_filepath, 'rb'))
    predict_catboost_regression = catboost_regression_model.predict(df)

    save_as_pickle(predict_catboost_regression, output_predict_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
