# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import f1_score
from src.utils import load_pickle
import json


@click.command()
@click.argument('input_pred_filepath', type=click.Path(exists=True))
@click.argument('input_true_filepath', type=click.Path(exists=True))
@click.argument('out_metrics_filepath', type=click.Path())
def main(input_pred_filepath, input_true_filepath, out_metrics_filepath):

    logger = logging.getLogger(__name__)
    logger.info('model evaluation...')

    pred = load_pickle(input_pred_filepath)
    true = load_pickle(input_true_filepath)

    metrics = {'f1_score_samples': f1_score(true, pred, average='samples', zero_division=0)}
    with open(out_metrics_filepath, "w") as f:
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
