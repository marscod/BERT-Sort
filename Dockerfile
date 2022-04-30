FROM jupyter/scipy-notebook

WORKDIR /usr/src/app

RUN pip install autogluon==0.4.0
RUN pip install FLAML==1.0.1
RUN pip install mljar-supervised==0.11.2
RUN pip install h2o==3.36.1.1
#WORKDIR /BERT_Sort/
ENTRYPOINT ["python3"]
