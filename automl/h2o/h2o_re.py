#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import json
import pandas as pd
import subprocess as subp
import time
import sys
from sklearn.preprocessing import OrdinalEncoder
import sys
import os
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import sklearn.metrics as skmet
import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import accuracy_score


this_dir = Path(os.getcwd()).resolve()
bert_sort_root = this_dir.parent.parent


DATA_DIR = bert_sort_root / 'benchmarks'

test_info_path = bert_sort_root / 'automl' / 'task_spec.json'
output_dir = Path('output')

if not output_dir.exists():
    output_dir.mkdir()

with open(test_info_path) as f:
    test_info = json.load(f)
    
    
def drop_columns(dataset, project_key, ignore):
    '''drop columns for ignored fields such as IDs'''
    if ignore:
        for c in ignore:
            if c in dataset.columns:
                print('dropping column %s' % (c))
                dataset = dataset.drop(c, axis=1)
    return dataset


def split(X, y, project_key, random_seed):
    '''split data sets with 75%/25% ratios for training and testing, respectively'''
    X_training, X_test, y_training, y_test = train_test_split(X, 
                                                              y, 
                                                              test_size=0.25, 
                                                              random_state=int(random_seed))
    return X_training, X_test, y_training, y_test

train_time=5 #set maximum allowance time for each experiment (per data set per Method) in minutes  
datasets = ['audiology', 
            'bank', 
            'car_eval', 
            'labor-relations', 
            'Nursery', 
            'Post-Operative', 
            'Pittsburgh_Bridges', 
            'cat-in-the-dat-ii'
            #'Coil_1999', 
            #'uci-automobile',
           ]
file_extensions = [
                   '_Raw.csv',
                   '_bs__roberta.csv', 
                   '_EncodedBERT.csv' , 
                   '_OrdinalEncoder.csv', 
                   '_GroundTruth.csv'
                  ]
seeds=['108',
       '180', 
       '234',
       '309', 
      # '533' used for 5 seeds
      ]

max_seeds=None

for seed in seeds[:max_seeds]: 
    for dataname in datasets:
        for tail in file_extensions:
            subfolder_name = dataname + '_' + str(seed) + '_m' + str(train_time) + tail
            print(f'Processing:{dataname}, Seed:{seed}, Encoded Method:{tail}')
            try:
                result_output_path = 'output/' + subfolder_name
                Path(result_output_path).mkdir(exist_ok=True)

                ignore = None
                target_column_name = test_info[dataname]['target_feature']
                print('target_columns:', target_column_name)

                target_dataset = test_info[dataname]['target_dataset']
                task_type = test_info[dataname]['task']

                if 'ignore' in test_info[dataname]:
                    ignore_column_name = test_info[dataname]['ignore']
                    if isinstance(ignore_column_name, list):
                        ignore_column_name = ','.join(ignore_column_name)
                else:
                    ignore_column_name = None

                filepath = DATA_DIR / (('%s' % (target_dataset)) + tail)
                if ignore_column_name:
                    ignore = ignore_column_name.split(',')

                main_df = pd.read_csv(filepath)
                print(filepath, main_df.shape)
                main_df = drop_columns(main_df, dataname, ignore)
                if isinstance(target_column_name[0],str):
                    target_cols = target_column_name
                else:
                    target_cols = main_df.columns[target_column_name]
                y_all = main_df[target_cols]
                X_all = main_df.drop(target_cols, axis=1)

                if len(target_column_name) == 1:
                    y_all = y_all.iloc[:, 0]

                X_train, X_test, y_train, y_test = split(X_all, y_all, dataname, seed)

                result_outputmodel_path = 'output/' + subfolder_name + '/model'
                Path(result_outputmodel_path).mkdir(exist_ok=True)

                # Specify automl goal and constraint
                if task_type == 'classification':
                    metric = 'accuracy'
                else:
                    metric = 'r2' 

                # Train with labeled input data
                train_data = pd.concat([X_train, y_train], axis=1)
                save_path = 'model/' + dataname + '_' + str(seed) + '_' + tail # where to store trained models
                if len(target_column_name) == 1:
                    target_column_name = target_column_name[0]
                    #y = target_cols
                    if isinstance(target_cols, pd.core.indexes.base.Index):
                        y = list(target_cols)[-1]
                        print("****", type(target_cols), target_cols, type(y), y) 
                    else:
                        y = target_cols
                else:
                    y = target_column_name
                print('target col:', target_cols, type(target_cols), y)
                print(train_data.columns)
                
                time_limit = train_time * 60
                params = {
                    'max_models': 20,
                    'max_runtime_secs': time_limit
                }
                h2o.init(nthreads=-1)
                train_df = pd.concat([X_train, y_train], axis=1)
                test_df = pd.concat([X_test, y_test], axis=1)
                os.makedirs('input', exist_ok=True)
                train_inputpath = 'input/train_%s_%s_%s.csv' % (dataname, seed, tail)
                test_inputpath = 'input/test_%s_%s_%s.csv' % (dataname, seed, tail)
                train_df.to_csv(path_or_buf=train_inputpath, index=False)
                test_df.to_csv(path_or_buf=test_inputpath, index=False)
                train = h2o.import_file(train_inputpath, header=1, col_names=list(train_df.columns))
                test = h2o.import_file(test_inputpath, header=1, col_names=list(test_df.columns))
                # Identify predictors and response
                print(target_column_name, type(target_column_name), train.columns)
                x = train.columns
                print('type of x: ', type(x), x, type(y), y)
                x.remove(y)

                # For binary classification, response should be a factor
                if task_type == 'classification':
                    train[y] = train[y].asfactor()
                    test[y] = test[y].asfactor()

                t0 = time.time()
                # Run AutoML for 20 base models (limited to 1 hour max runtime by default)
                aml = H2OAutoML(**params)
                aml.train(x=x, y=y, training_frame=train)
                t1 = time.time()

                # To generate predictions on a test set, you can make predictions
                # directly on the `"H2OAutoML"` object or on the leader model
                # object directly
                t2 = time.time()
                y_pred = aml.predict(test)
                #y_pred_df = h2o.as_list(y_pred)
                print(y_pred, type(y_pred))
                #print(y_pred_df, type(y_pred_df))
                y_pred_value = h2o.as_list(y_pred['predict'])
                print(y_pred_value, type(y_pred_value))
                # or:
                #preds = aml.leader.predict(test)
                t3 = time.time()

                if task_type == 'classification':
                    score = skmet.f1_score(y_test, y_pred_value, average='macro')
                    acc_score = skmet.accuracy_score(y_test, y_pred_value)
                elif task_type == 'regression':
                    score = skmet.r2_score(y_test, y_pred_value)
                    acc_score=-1 #since we excluded regression tasks, this value is not used
                
                outputpath = ('%s/%s_%s_m%s%s.txt' % (str(output_dir), dataname, seed, train_time, tail))
                with open(outputpath, 'w') as f:
                    output_log =  '%s %s %f %f %f %f \n' % (dataname, 
                                                        seed, 
                                                        t1-t0, 
                                                        t3-t2, 
                                                        score, 
                                                        #roc_auc_score, 
                                                        acc_score, 
                                                       )
                    f.write(output_log)
                    print(output_log)

                X_testpath = ('%s/%s/X_test%s.csv' % (str(output_dir), subfolder_name, tail))
                X_test.to_csv(X_testpath, index=False)

            except Exception as e:
                logfile = ('%s/%s/%serror.log' % (str(output_dir), subfolder_name, tail))
                with open(logfile, 'w') as f:
                    f.write(traceback.format_exc())
                continue

                
                


