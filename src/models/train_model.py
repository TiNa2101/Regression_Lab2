# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import GridSearchCV
from src.utils import save_as_pickle
from src.models.sklearn_model import model
import pandas as pd


@click.command()
@click.argument('input_train_data_filepath', type=click.Path(exists=True))
@click.argument('input_train_target_filepath', type=click.Path(exists=True))
@click.argument('output_model_filepath', type=click.Path())
def main(input_train_data_filepath, input_train_target_filepath, output_model_filepath):

    logger = logging.getLogger(__name__)
    logger.info('training model...')

    train = pd.read_pickle(input_train_data_filepath)
    target = pd.read_pickle(input_train_target_filepath)

    rscv = GridSearchCV(
        estimator=model,
        param_grid={'estimator__model__C': [0.5, 1.0, 1.7]},
        scoring='f1_samples',
        cv=5,
        refit=True
    )

    rscv.fit(train, target)
    save_as_pickle(rscv, output_model_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
