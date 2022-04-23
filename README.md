# BERT-Sort: A Zero-shot MLM Semantic Encoder on Ordinal Features for AutoML
This repository provides artifacts for reproducing the results of BERT-Sort paper.The artifacts include the following items.

## Reproducibility Checklist
The reproducibility checklist is available [here](https://anonymous.4open.science/r/BERT-Sort-067E/Reproducibility%20Checklist.pdf).

## Benchmarks Folder
This folder includes 10 data sets that consists of both raw data set and encoded data set where it is encoded through BERT-Sort Encoder with MLM initialization of <img src="https://latex.codecogs.com/svg.latex?&space;M_{1..4}"/>. 

In each data set folder, there are original files and encoded data sets with 4 different MLMs. For instance, `bank/bank.csv` is the original file for raw data set and `bank/bank.csv_bs__roberta.csv` is encoded raw data set with BERT-Sort Encoder which is initiated with _RoBERTa MLM_. Both raw and encoded data sets have been used to evaluate the proposed approach in 5 AutoML platforms.

## Output Folder
This folder includes the configuration files, ground truth and evaluation results. Each folder in `output` contains a configuration file as `config.json` with a set of keys of
`['model', 'mask', 'separator', 'eta', 'lower', 'target_files', 'ground_truth', 'default_grouping', 'default_zeta', 'preprocess']`. For instance, 'outputs/out_bert_base_uncased/config.json' includes all hyperparameters, configuration, ground truth of 42 features, task specification (regression/classification)  for `BERT-base_uncased MLM`.

The key of `target_files` represent task information such as data set filename, a URL reference, type of task (classification or regression for AutoML evaluation), type of evaluation metric (F1 or RMSE). 

The key of `ground_truth` is a dictionary where the keys are representing the feature name (if any) or feature index, and the values are a list of ranked ordinal values. 

Each MLM folder includes a set of dumped pickles (`*.pkl`) which includes: i) input values, ii) OrdinalEncoder output, iii) intermediate steps and iv) final evaluation results of BERT-Sort process for each data set. 

This folder also includes i) `all_outputs.csv`(detailed-evaluation), and `summary.csv` (summary of each data set) for evaluation results of BERT-Sort on 10 data sets with 42 distinct features per MLM. For instance, `out_bert_base_uncased/all_outputs.csv` corresponds to detailed-results of _BERT-base_uncased_ MLM on all 42 features. A heatmap plot of _all_outputs.csv_ is available at `out_bert_base_uncased/all_outputs.png`.

## AuoML Folder
This folder includes all AutoML evaluaion results based on i) raw data set, ii) encoded data set through BERT-Sort. Each experiment located in a file with one of the two following structures.

### Raw data set Format
`automl/<auoml_name>/<data set name>_<seed>_<time_limitaion>.txt`

### BERT-Sort Encoded Data Set Format
`automl/<auoml_name>/<data set name>_<seed>_<time_limitaion>_bs_<model_name>.csv.txt`

Each file includes `<data set name> <seed> <training time> <prediction time> <score>`

The following seeds have been used to split both raw data sets and encoded data sets where we used [`sklearn.model_selection.train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
```python
one_seed = [108] #used in table 5 & figure 8
four_seeds = [108, 180, 234, 309] #used in figure 8
five_seeds =  ['108', '180', '234', '309', '533'] #used in figure 8
```
## Experiment Artifacts for Rebuttal Version of the paper
You may find all results of Table 5 and Table 6 in `https://anonymous.4open.science/r/BERT-Sort-067E/automl/<AUTOML>/<DATASET>_<SEED>_m5_*<METHOD>.txt` (i.e., [`Nursery_108_m5_EncodedBERT.csv.txt`](https://anonymous.4open.science/r/BERT-Sort-067E/automl/autogluon/Nursery_108_m5_EncodedBERT.csv.txt) refers to Nursery data set with seed 108 and encoded value through EncodedBERT approach. <SEEDS> include `[108, 180, 234, 309]`,and <METHOD> include 5 different datasets of `['Raw', 'EncodedBERT', 'bs_roberta', 'OrdinalEncoder', 'GroundTruth']`.

 ## Demo
 A demonstration of the process (normolized score for visualization).
  
 [Watch Demo](https://anonymous.4open.science/r/BERT-Sort-067E/Demo1.mp4)

