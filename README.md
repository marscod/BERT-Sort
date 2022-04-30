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
four_seeds = [108, 180, 234, 309]  # used in Table 5, Table 6, and Figure 8 (4 first values as seeds, 1 seed: 108)
five_seeds =  ['108', '180', '234', '309', '533'] # used in Figure 8
```
## Experiment Artifacts for Rebuttal Version of the paper
You may find all results of Table 5 and Table 6 in `https://anonymous.4open.science/r/BERT-Sort-067E/automl/<AUTOML>/<DATASET>_<SEED>_m5_*<METHOD>.txt` (i.e., [`Nursery_108_m5_EncodedBERT.csv.txt`](https://anonymous.4open.science/r/BERT-Sort-067E/automl/autogluon/Nursery_108_m5_EncodedBERT.csv.txt) refers to Nursery data set with seed 108 and encoded value through EncodedBERT approach. \<SEEDS\> include `[108, 180, 234, 309]`,and \<METHOD\> includes 5 different datasets of `['Raw', 'EncodedBERT', 'bs_roberta', 'OrdinalEncoder', 'GroundTruth']`. 

Similarly, you can find the encoded versions of each data set per encoded method in `benchmark` folder: `benchmarks/<DATA SET>/FILE_<METHOD>.csv` (i.e., [uci_Pittsburgh_Bridges/bridges.data.version2.txt.csv_EncodedBERT.csv](https://anonymous.4open.science/r/BERT-Sort-067E/benchmarks/uci_Pittsburgh_Bridges/bridges.data.version2.txt.csv_EncodedBERT.csv)


## Demo
A demonstration of the process (normolized score for visualization).
[Watch the Demo](https://anonymous.4open.science/r/BERT-Sort-067E/Demo1.mp4)
<img src="Demo1.gif" width="600px"/>

## Reproducing AutoML Experiments
Each AutoML folder include a code where it is producing the evaluation results per data set per encoded method per seed. Each folder contains `run.sh` that allows you to run the code. The following is a link to each code.
Requirements can be found in [automl/requirements.txt](https://anonymous.4open.science/r/BERT-Sort-067E/automl/requirements.txt) and task specification can be found in [automl/task_spec.json](https://anonymous.4open.science/r/BERT-Sort-067E/automl/task_spec.json).

- AutoGluon:
  + [AutoGluon code](https://anonymous.4open.science/r/BERT-Sort-067E/automl/autogluon/autogluon_re.py)
  + [AutoGluon run script](https://anonymous.4open.science/r/BERT-Sort-067E/automl/autogluon/run.sh)
- H2O:
  + [H2O code](https://anonymous.4open.science/r/BERT-Sort-067E/automl/h2o/h2o_re.py)
  + [H2O run script](https://anonymous.4open.science/r/BERT-Sort-067E/automl/h2o/run.sh)
- MLJAR:
  + [MLJAR code](https://anonymous.4open.science/r/BERT-Sort-067E/automl/mljar/mljar_re.py)
  + [MLJAR run script](https://anonymous.4open.science/r/BERT-Sort-067E/automl/mljar/run.sh)
- FLAML:
  + [FLAML code](https://anonymous.4open.science/r/BERT-Sort-067E/automl/flaml/flaml_re.py)
  + [FLAML run script](https://anonymous.4open.science/r/BERT-Sort-067E/automl/flaml/run.sh)
  
### How to run each AutoML experiment?
```shell
pwd     #.../BERT-SORT/
cd automl/h2o     #other options: [mljar,flaml,autogluon]
sh run.sh
```

### Outputs
Each AutoML will generate a set of text file (i.e., [autogluon/Nursery_108_m5_EncodedBERT.csv.txt](https://anonymous.4open.science/r/BERT-Sort-067E/automl/autogluon/Nursery_108_m5_EncodedBERT.csv.txt) it also generates two folders of `output` and `log` where it collects intermediate results and output logs.



## Docker (updated)
You may use [Dockerfile](https://anonymous.4open.science/r/BERT-Sort-067E/Dockerfile) to build a docker with 4 different AutoMLs which have been used in our experiment. You may also use the following shell scripts.
1. Build the Docker from [build.sh](https://anonymous.4open.science/r/BERT-Sort-067E/build.sh) or execute the following commands.
```shell
$(pwd) #this folder: BERT-Sort/

sudo docker build -t automl .
```

2. Run any AutoML on benchmark data sets by using shell scripts or execute the following commands.

2.1. FLAML: [run_flaml_docker.sh](https://anonymous.4open.science/r/BERT-Sort-067E/run_flaml_docker.sh)
```shell
sudo docker run --rm -v $(pwd):/BERT-Sort -it -w /BERT-Sort/automl/flaml --entrypoint python3 automl flaml_re.py
```

2.2. MLJAR : [run_mljar_docker.sh](https://anonymous.4open.science/r/BERT-Sort-067E/run_mljar_docker.sh)
```shell
sudo docker run --rm -v $(pwd):/BERT-Sort -it -w /BERT-Sort/automl/mljar --entrypoint python3 automl mljar_re.py
```

2.3. H2O : [run_h2o_docker.sh](https://anonymous.4open.science/r/BERT-Sort-067E/run_h2o_docker.sh)
```shell
sudo docker run --rm -v $(pwd):/BERT-Sort -it -w /BERT-Sort/automl/h2o --entrypoint python3 automl h2o_re.py
```

2.4. AutoGluon : [run_autogluon_docker.sh](https://anonymous.4open.science/r/BERT-Sort-067E/run_autogluon_docker.sh)
```shell
sudo docker run --rm -v $(pwd):/BERT-Sort -it -w /BERT-Sort/automl/autogluon --entrypoint python3 automl autogluon_re.py
```

By default it generates all results with 5 encoded data sets and each one with 4 seeds.
