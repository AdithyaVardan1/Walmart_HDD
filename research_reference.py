import os
import numpy as np
import pandas as pd
from interpretableai import iai

def get_X(df, drop_age, drop_eng):
  cols_to_drop = [col for col in df.columns if 'smart' not in col]
  X = df.drop(cols_to_drop, axis=1)

  if drop_age:
    inds = [4, 9, 12, 192, 193, 240, 241, 242]
    cols_age = [col for col in X.columns for i in inds
                if ('_' + str(i) + '_') in col]
    X = X.drop(cols_age, axis=1)

  if drop_eng:
    cols_engineered = [col for col in X.columns
                       if ('raw_' in col) or ('normalized_' in col)]
    X = X.drop(cols_engineered, axis=1)

  return(X)

def train_OST(df, file_name, drop_age=True, drop_eng=True):
  X = get_X(df, drop_age=drop_age, drop_eng=drop_eng)
  died = df['failed']
  times = df['remaining_useful_life']

  grid = iai.GridSearch(
      iai.OptimalTreeSurvivor(
          random_seed=1,
          missingdatamode='separate_class',
          criterion='localfulllikelihood',
          minbucket=int(0.01 * len(X))
      ),
      max_depth=range(2, 5)
  )
  grid.fit(X, died.tolist(), times, validation_criterion='harrell_c_statistic')

  grid.get_learner().write_html('models/' + file_name + '.html')
  grid.write_json('models/' + file_name + '.json')

def train_OCT(df, file_name, drop_age=False, drop_eng=True):
  machines_failures = list(df[df['failure'] == 1]['serial_number'].unique())
  machines_no_failures = list(df['serial_number'].drop_duplicates() \
                             .replace(machines_failures, np.NaN).dropna())

  np.random.seed(1)
  machines_train = list(np.random.choice(machines_failures,
                                         int(len(machines_failures) * 0.8),
                                         replace=False)) + \
                   list(np.random.choice(machines_no_failures,
                                         int(len(machines_no_failures) * 0.8),
                                         replace=False))
  machines_test = list(set(machines_failures + machines_no_failures).difference(
                       set(machines_train)))

  df = df.sort_values(['serial_number', 'datetime'])
  df = df.set_index(['serial_number', 'datetime'])

  df_train = df.loc[machines_train]
  df_test = df.loc[machines_test]

  X_train = get_X(df_train, drop_age=drop_age, drop_eng=drop_eng)
  X_test = get_X(df_test, drop_age=drop_age, drop_eng=drop_eng)
  y_train = df_train['failure_next_30'].apply(
                lambda x: 'Failure' if x else 'Operational')
  y_test = df_test['failure_next_30'].apply(
                lambda x: 'Failure' if x else 'Operational')

  grid = iai.GridSearch(
      iai.OptimalTreeClassifier(
          random_seed=1,
          missingdatamode='separate_class',
          criterion='gini',
          minbucket=int(0.005 * len(X_train))
      ),
      max_depth=range(2, 7)
  )
  grid.fit_cv(X_train, y_train)

  grid.get_learner().write_html('models/' + file_name + '.html')
  grid.write_json('models/' + file_name + '.json')


df_failures = pd.read_csv('dataset_1.csv')
df_failures['datetime'] = pd.to_datetime(df_failures['datetime'])
df_failures_rs = df_failures.sample(50000, random_state=1)
train_OST(df_failures_rs, '1_OST_17_20', drop_age=True, drop_eng=True)


df = pd.read_csv('dataset_2.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df_rs = df.sample(50000, random_state=1)
train_OST(df_rs, '2_OST_19_20', drop_age=True, drop_eng=True)


df = pd.read_csv('dataset_3.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
train_OCT(df, '3_OCT_19_20', drop_age=True, drop_eng=True)


df = pd.read_csv('dataset_4.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
train_OST(df, '4_OST_Q1_20_no_age', drop_age=True, drop_eng=True)
