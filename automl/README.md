## AuoML Folder
This folder includes all AutoML evaluaion results based on i) raw data set, ii) encoded data set through BERT-Sort. Each experiment located in a file with one of the two following structures.

### Raw data set Format
`automl/<auoml_name>/<data set name>_<seed>_<time_limitaion>.txt`

### BERT-Sort Encoded Data Set Format
`automl/<auoml_name>/<data set name>_<seed>_<time_limitaion>_bs_<model_name>.csv.txt`

Each file includes `<data set name> <seed> <training time> <prediction time> <score>`

