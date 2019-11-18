# P-SIF: Document Embeddings using Partition Averaging


## Introduction
  - For text classification and information retrieval tasks, text data has to be represented as a fixed dimension vector. 
  - We propose simple feature construction technique named **P-SIF: Document Embeddings using Partition Averaging**
  - We demonstrate our method through experiments on multi-class classification on 20newsGroup dataset, multi-label text classification on Reuters-21578 dataset, Semantic Textual Similarity Tasks (STS 12-16) and other classification tasks.

## Testing
There are 3 folders named 20newsGroup, Reuters and STS which contains code related to multi-class classification on 20newsGroup dataset, multi-label classification on Reuters dataset, and Semantic Texual Similarity Task (STS) on 27 datasets.
#### 20newsGroup
Change directory to 20newsGroup for experimenting on 20newsGroup dataset and create train and test tsv files as follows:
```sh
$ cd 20newsGroup
$ python create_tsv.py
```
Get word vectors for all words in vocabulary:
```sh
$ python Word2Vec.py 200
# Word2Vec.py takes word vector dimension as an argument. We took it as 200.
```
Get Sparse Document Vectors (SCDV) for documents in train and test set and accuracy of prediction on test set:
```sh
$ python ksvd_sif.py 200 40
# ksvd_sif.py takes word vector dimension and number of partitions as arguments. We took word vector dimension as 200 and number of partitions as 60.
```

#### Reuters
Change directory to Reuters for experimenting on Reuters-21578 dataset. As reuters data is in SGML format, parsing data and creating pickle file of parsed data can be done as follows:
```sh
$ python create_data.py
# We don't save train and test files locally. We split data into train and test whenever needed.
```
Get word vectors for all words in vocabulary: 
```sh
$ python Word2Vec.py 200
# Word2Vec.py takes word vector dimension as an argument. We took it as 200.
```
Get Sparse Document Vectors (SCDV) for documents in train and test set:
```sh
$ python ksvd_sif.py 200 40
# ksvd_sif.py takes word vector dimension and number of partitions as arguments. We took word vector dimension as 200 and number of partitions as 60.
```
Get performance metrics on test set:
```sh
$ python metrics.py 200 40
# metrics.py takes word vector dimension and number of partitions as arguments. We took word vector dimension as 200 and number of partitions as 60.
```

#### STS
Change directory to STS for experimenting on STS dataset.
First download paragram_sl999_small.txt from John Wieting's github (https://github.com/jwieting/iclr2016) and keep it in STS/data folder
dataset is inside SentEval folder
for gmm based data partioning, parameters for cluster, weightage etc is stored in parameters_gmm.csv
Create word topic vector for each word by using wordvectors from paragram_sl999_small.txt
```sh
$ python create_word_topic_gmm.py
```
Get similarity score for each sts dataset
```sh
$ python psif_main_gmm.py
# it will output each dataset similarity score and corresponding parameters.
```
for ksvd based data partioning, parameters for cluster, weightage etc is stored in parameters_ksvd.csv
Create word topic vector for each word by using wordvectors from paragram_sl999_small.txt
```sh
$ python create_word_topic_ksvd.py
```
Get similarity score for each sts dataset
```sh
$ python psif_main_ksvd.py
# it will output each dataset similarity score and corresponding parameters.
```

#### Other_Datasets
For running P-SIF on rest of the 7 datasets, go to Other_Datasets folder. 
Inside Other_Datasets folder, each dataset has a folders with the dataset name. 
Follow the Readme.md has been included for running the P-SIF. 
You have to download google embedding from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing and placed in the Other_Dataset folder.

## Requirements
Minimum requirements:
  -  Python 2.7+
  -  NumPy 1.8+
  -  Scikit-learn
  -  Pandas
  -  Gensim

Note: You neednot download 20newsGroup or Reuters-21578 dataset. All datasets are present in their respective directories.

[//]: # (We used SGMl parser for parsing Reuters-21578 dataset from  https://gist.github.com/herrfz/7967781)
