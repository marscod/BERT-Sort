#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess as subp
import traceback
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import sklearn.metrics as skmet
from autogluon.tabular import TabularPredictor
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
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.25, random_state=int(random_seed))
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
            try:
                print(f'Processing:{dataname}, Seed:{seed}, Encoded Method:{tail}')
                # construct the path of benchmark
                subfolder_name = dataname + '_' + str(seed) + '_m' + str(train_time) + tail
                result_output_path = 'output/' + subfolder_name
                Path(result_output_path).mkdir(exist_ok=True)

                ignore = None
                target_column_name = test_info[dataname]['target_feature']
                #print('target_columns:', target_column_name)

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

                if dataname == 'bank' and tail == '':
                        main_df = pd.read_csv(filepath, sep=';', header='infer')
                else:
                    main_df = pd.read_csv(filepath)
                #print(filepath, main_df.shape)

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
                save_path = 'model/' + dataname + '_' + str(seed) + '_' + tail # where to store trained models #TODO: use path for cross-platforms
                if len(target_column_name) == 1:
                    target_column_name = target_column_name[0]
                #print('target col:', target_cols)
                #print(train_data.columns)
                t0 = time.time()
                if len(target_cols)>1:
                    #out of scope
                    continue
                else:
                    predictor = TabularPredictor(label=list(target_cols)[-1], # all datasets used only the last column except coil whihc is excluded
                                                 eval_metric=metric, 
                                                 path=save_path, 
                                                 verbosity=3).fit(
                                                                  train_data, 
                                                                  presets='best_quality',
                                                                  time_limit=train_time*60,
                                                                 )
                
                t1 = time.time()
                results = predictor.fit_summary()
                

                # Predict
                test_data = pd.concat([X_test, y_test], axis=1)
                t2 = time.time()
                y_pred = predictor.predict(X_test)
                t3 = time.time()

                performance = predictor.evaluate(test_data)
                #print(performance)

                if task_type == 'classification':
                    score = skmet.f1_score(y_test,
                                           y_pred,
                                           average='macro')
                    
                    acc_score = skmet.accuracy_score(y_test, y_pred)
                elif task_type == 'regression':
                    score = skmet.r2_score(y_test, y_pred)
                    acc_score=-1 #since we excluded regression tasks, this value is not used
                

                outputpath = ('%s/%s_%s_h%s%s.txt' % (str(output_dir), 
                                                      dataname, seed, 
                                                      train_time*60, 
                                                      tail))
                with open(outputpath, 'w') as f:
                    output_log =  '%s %s %f %f %f %f \n' % (dataname, 
                                                        seed, 
                                                        t1-t0, 
                                                        t3-t2, 
                                                        score, 
                                                        acc_score, 
                                                       )
                    
                    f.write(output_log)
                    #print(output_log)

                X_testpath = ('%s/%s/X_test%s.csv' % (str(output_dir), 
                                                      subfolder_name, 
                                                      tail))
                
                X_test.to_csv(X_testpath, index=False)

            except Exception as e:
                Path(output_dir).mkdir(exist_ok=True)
                logfile = ('%s/%s/%serror.log' % (str(output_dir), 
                                                  subfolder_name, 
                                                  tail))
                
                with open(logfile, 'w') as f:
                    f.write(traceback.format_exc())
                continue
                #raise
                
            







