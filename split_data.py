import os
import argparse
import numpy as np      # type: ignore
import pandas as pd     # type: ignore
from common_utils import load_file
from sklearn.model_selection import train_test_split    # type: ignore


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_txt_file',
                        help='Text file containing polynomials and expansions',
                        type=str, default='./train.txt')
    parser.add_argument('--output_dir',
                        help='Dirctory to store output',
                        type=str, default='./output')
    parser.add_argument('--random_state',
                        help='Random state for initialization',
                        type=int, default=42)
    parser.add_argument('--val_size',
                        help='Size of validation partition',
                        type=float, default=0.2)
    return parser.parse_args()


def split_data(args):

    data_txt_file = args.data_txt_file
    output_dir = args.output_dir
    random_state = args.random_state
    val_size = args.val_size

    os.makedirs(output_dir, exist_ok=True)

    factors, expansions = load_file(data_txt_file)

    X_train, X_val, y_train, y_val = train_test_split(
        factors, expansions, test_size=val_size, random_state=random_state)

    df_train = pd.DataFrame()
    df_train['factor'] = np.array(X_train).T
    df_train['expansion'] = np.array(y_train).T

    training_filepath = os.path.join(output_dir, 'training.csv')
    df_train.to_csv(training_filepath, index=False)

    df_val = pd.DataFrame()
    df_val['factor'] = np.array(X_val).T
    df_val['expansion'] = np.array(y_val).T

    validation_filepath = os.path.join(output_dir, 'validation.csv')
    df_val.to_csv(validation_filepath, index=False)


def main():
    args = get_arguments()
    split_data(args)


if __name__ == '__main__':
    main()
