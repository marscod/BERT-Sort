# BERT-Sort: A Zero-shot MLM Semantic Encoder on Ordinal Features for AutoML
This repository provides artifacts for reproducing the results of BERT-Sort paper.The artifacts include the following items.

## Reproducibility Checklist
The reproducibility checklist is available [here](https://github.com/marscod/BERT-Sort/blob/main/Reproducibility%20Checklist.pdf).


## Benchmarks Folder
This folder includes 10 data sets that consists of both raw data set and encoded data set where it is encoded through BERT-Sort Encoder with MLM initialization of <img src="https://latex.codecogs.com/svg.latex?&space;M_{1..4}"/>. In each data set folder, there are original files and encoded data sets with 4 different MLMs. For instance, 'bank/bank.csv' is the original file for raw data set and `bank/bank.csv_bs__roberta.csv` is encoded raw data set with BERT-Sort Encoder which is initiated with RoBERTa MLM. Both raw and encoded data sets have been used to evaluate the proposed approach in 5 AutoML platforms.
## Output Folder
This folder includes the configuration files, ground truth and evaluation results. Each folder in `output` contains a configuration file as `config.json` with a set of keys of
`['model', 'mask', 'separator', 'eta', 'lower', 'target_files', 'ground_truth', 'default_grouping', 'default_zeta', 'preprocess']`. 

The key of `target_files` represent task information such as data set filename, a URL reference, type of task (classification or regression for AutoML evaluation), type of evaluation metric (F1 or RMSE). 

The key of `ground_truth` is a dictionary where the keys are representing the feature name (if any) or feature index, and the values are a list of ranked ordinal values. 

Each MLM folder includes a set of dumped pickles (`*.pkl`) which includes intermediate steps of BERT-Sort process and evaluation results for each data set. It includes `summary.csv` and `all_outputs.csv` for evaluation results of BERT-Sort on 10 data sets with 42 distinct features.

## AuoML Folder
This folder includes all AutoML evaluaion results based on i) raw data set, ii) encoded data set through BERT-Sort. Each experiment located in a file with one of the two following structures.

### Raw data set Format
`automl/<auoml_name>/<data set name>_<seed>_<time_limitaion>.txt`

### BERT-Sort Encoded Data Set Format
`automl/<auoml_name>/<data set name>_<seed>_<time_limitaion>_bs_<model_name>.csv.txt`

Each file includes '<data set name> <seed> <training time> <prediction time> <score>'

