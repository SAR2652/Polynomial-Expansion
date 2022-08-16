import numpy as np
import pandas as pd
from main import load_file
from sklearn.model_selection import train_test_split

factors, expansions = load_file('./train.txt')

X_train, X_val, y_train, y_val = train_test_split(factors, expansions, test_size = 0.2, random_state = 42)

df_train = pd.DataFrame()
df_train['factor'] = np.array(X_train).T
df_train['expansion'] = np.array(y_train).T

df_train.to_csv('./data/training.csv', index = False)

df_val = pd.DataFrame()
df_val['factor'] = np.array(X_val).T
df_val['expansion'] = np.array(y_val).T

df_val.to_csv('./data/validation.csv', index = False)
