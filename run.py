"""
The orchestration script.
"""

import argparse
import logging.config
# logging has to be configured here to show loggers from other modules
logging.config.fileConfig('config/local.conf')
logger = logging.getLogger('chatbot-pipeline')

import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.prepare import prepare
from src.train import main

if __name__ == '__main__':

    # Add parsers
    parser = argparse.ArgumentParser(description="chatbot pipeline")
    subparsers = parser.add_subparsers(dest='subparser_name')

    # Prepare data
    sb_prepare = subparsers.add_parser("prepare", description="Prepare and preprocess data")
    sb_prepare.add_argument("--input_path", default='data/message_1.json', help="original json file from FB")
    sb_prepare.add_argument("--output_path", default='data/contexts.csv', help="local path to store processed datasets")

    # Model
    sb_train = subparsers.add_parser("train", description="Finetune DialoGPT")
    sb_train.add_argument("--input_path", default='data/contexts.csv')

    args = parser.parse_args()
    sp_used = args.subparser_name

    if sp_used == 'prepare': # prepare data
        prepare(args.input_path, args.output_path)
    elif sp_used == 'train': # finetuning chatbot
        df = pd.read_csv(args.input_path)
        trn_df, val_df = train_test_split(df, test_size = 0.1, random_state=42)
        main(trn_df, val_df)
    else:
        parser.print_help()