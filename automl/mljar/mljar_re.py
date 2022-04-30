#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import json
import pandas as pd
import subprocess as subp
import time
import sys
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import sklearn.metrics as skmet

from supervised.automl import AutoML
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
def convert_to_mljar_metric_type(task_type, metric_type):
    metric_map = {
        'classification': {
            'LogLoss' : 'logloss',
            'ROC_AUC' : 'auc',
            'AUC' : 'auc',
            'F1' : 'f1',
            'Accuracy' : 'accuracy'
        },
        'regression' : {
            'RMSE' : 'rmse',
            'MSE' : 'mse',
            'MAE' : 'mae',
            'R2' : 'r2',
            'MAPE' : 'mape',
        }
    }
    default_metric = {
        'classification': 'logloss',
        'regression' : 'rmse'
    }
    if metric_type in metric_map[task_type]:
        print('convert metric type %s to %s' % (metric_type, 
                                                metric_map[task_type][metric_type]))
        
        return metric_map[task_type][metric_type]
    else:
        print('cannot find metric %s. Use default metric %s' % (metric_type, 
                                                                default_metric[task_type]))
        return default_metric[task_type]

for seed in seeds[:max_seeds]: 
    for dataname in datasets:
        for tail in file_extensions:
            print(f'Processing:{dataname}, Seed:{seed}, Encoded Method:{tail}')    
            subfolder_name = dataname + '_' + str(seed) + '_m' + str(train_time) + tail
            try:
                result_output_path = 'output/' + subfolder_name
                Path(result_output_path).mkdir(exist_ok=True)

                ignore = None
                target_column_name = test_info[dataname]['target_feature']
#                print('target_columns:', target_column_name)

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
                main_df = drop_columns(main_df, dataname, ignore)

                if isinstance(target_column_name[0],str):
                    target_cols = target_column_name
                else:
                    target_cols = main_df.columns[target_column_name]
                y_all = main_df[target_cols]
                X_all = main_df.drop(target_cols, axis=1)

                if len(target_column_name) == 1:
                    y_all = y_all.iloc[:, 0]

                X_train, X_test, y_train, y_test = split(X_all, 
                                                         y_all, 
                                                         dataname, seed)

                result_outputmodel_path = 'output/' + subfolder_name + '/model'
                Path(result_outputmodel_path).mkdir(exist_ok=True)

                # Specify automl goal and constraint
                if task_type == 'classification':
                    metric = 'accuracy'
                else:
                    metric = 'r2'

                time_limit = 60 * train_time
                params = {
                    'total_time_limit': time_limit,
                    'mode': 'Compete',
                    'results_path': result_outputmodel_path
                }

                metric_type = convert_to_mljar_metric_type(task_type, metric)
                params['eval_metric'] = metric_type 

                if task_type == 'classification':
                    if len(y_train.unique()) <= 2:
                        mljar_task_type = 'binary_classification'
                    else:
                        mljar_task_type = 'multiclass_classification'
                else:
                    mljar_task_type = task_type

                model = AutoML(ml_task = mljar_task_type, **params)
                t0 = time.time()
                model.fit(X_train, y_train)
                t1 = time.time()

                # Predict
                t2 = time.time()
                y_pred = model.predict(X_test)
                t3 = time.time()

                if task_type == 'classification':
                    score = skmet.f1_score(y_test, y_pred, average='macro')
                    acc_score = skmet.accuracy_score(y_test, y_pred)
                elif task_type == 'regression':
                    score = skmet.r2_score(y_test, y_pred)
                    acc_score=-1 #since we excluded regression tasks after manual invastigation, this value is not used
                

                outputpath = ('%s/%s_%s_m%s%s.txt' % (str(output_dir), 
                                                      dataname, 
                                                      seed, 
                                                      train_time, 
                                                      tail))
                
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

                X_testpath = ('%s/%s/X_test%s.csv' % (str(output_dir), 
                                                      subfolder_name, 
                                                      tail))
                
                X_test.to_csv(X_testpath, index=False)

            except Exception as e:
                logfile = ('%s/%s/%serror.log' % (str(output_dir), 
                                                  subfolder_name, tail))
                with open(logfile, 'w') as f:
                    f.write(traceback.format_exc())
                raise
                