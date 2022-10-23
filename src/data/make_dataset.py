# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from src.utils import save_as_pickle
from preprocess import preprocess_data, preprocess_target, extract_target
from src.config import RS
import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.option('--output_data_filepath', type=click.Path(), default=None)
@click.option('--output_train_data_filepath', type=click.Path(), default=None)
@click.option('--output_train_target_filepath', type=click.Path(), default=None)
@click.option('--output_val_data_filepath', type=click.Path(), default=None)
@click.option('--output_val_target_filepath', type=click.Path(), default=None)
def main(
    input_filepath, 
    output_data_filepath=None, 
    output_train_target_filepath=None,
    output_train_data_filepath=None,
    output_val_target_filepath=None,
    output_val_data_filepath=None
    ):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making interim data set from raw data')

    df = pd.read_csv(input_filepath)
    df = preprocess_data(df)
    if output_train_target_filepath:
        train, val = train_test_split(df, test_size=0.2, random_state=RS)
        
        train_data, train_target =  extract_target(train)
        val_data, val_target =  extract_target(val)
        
        train_target = preprocess_target(train_target)
        val_target = preprocess_target(val_target)

        save_as_pickle(train_data, output_train_data_filepath)
        save_as_pickle(train_target, output_train_target_filepath)
        save_as_pickle(val_data, output_val_data_filepath)
        save_as_pickle(val_target, output_val_target_filepath)

    if output_data_filepath:
        save_as_pickle(df, output_data_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
