# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 21:13:30 2017

@author: ankit
"""
import time
import warnings
from sklearn.decomposition import PCA as sklearnPCA

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import pandas as pd
import time, pickle
import numpy as np
from sklearn.externals import joblib
from sklearn.mixture import GMM
from ksvd import ApproximateKSVD
from joblib import Parallel, delayed


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


def cluster_GMM(num_clusters, word_vectors,transform_n_nonzero_coefs=None):
    # Initalize a GMM object and use it for clustering.
    # clf =  GMM(n_components=num_clusters,covariance_type="tied", init_params='wc', n_iter=10)
    aksvd = ApproximateKSVD(n_components=num_clusters,transform_n_nonzero_coefs=transform_n_nonzero_coefs)
    dictionary = aksvd.fit(word_vectors).components_
    idx_proba = aksvd.transform(word_vectors)
    idx = np.argmax(idx_proba, axis=1)
    print("Clustering Done...", time.time() - start, "seconds")
    joblib.dump(idx, 'data_files/gmm_latestclusmodel_len2alldata.pkl')
    print("Cluster Assignments Saved...")
    joblib.dump(idx_proba, 'data_files/gmm_prob_latestclusmodel_len2alldata.pkl')
    print("Probabilities of Cluster Assignments Saved...")
    return (idx, idx_proba)

def cluster_GMM_cluster(num_clusters, word_vectors):
	# Initalize a GMM object and use it for clustering.
	clf =  GMM(n_components=num_clusters,
                    covariance_type="tied", init_params='wc', n_iter=10)
	# Get cluster assignments.
	idx = clf.fit_predict(word_vectors)
	print("Clustering Done...", time.time()-start, "seconds")
	# Get probabilities of cluster assignments.
	idx_proba = clf.predict_proba(word_vectors)
	# Dump cluster assignments and probability of cluster assignments.
	joblib.dump(idx, 'data_files/gmm_latestclusmodel_len2alldata.pkl')
	print("Cluster Assignments Saved...")
	joblib.dump(idx_proba, 'data_files/gmm_prob_latestclusmodel_len2alldata.pkl')
	print("Probabilities of Cluster Assignments Saved...")
	return (idx, idx_proba)

def read_GMM(idx_name, idx_proba_name):
    # Loads cluster assignments and probability of cluster assignments.
    idx = joblib.load(idx_name)
    idx_proba = joblib.load(idx_proba_name)
    print("Cluster Model Loaded...")
    return (idx, idx_proba)


def get_probability_word_vectors(word_centroid_map,word_centroid_prob_map, num_clusters,num_features):
    # This function computes probability word-cluster vectors.
    prob_wordvecs = {}
    for word in word_centroid_map:
        #         print(word
        prob_wordvecs[word] = np.zeros(num_clusters * num_features, dtype="float32")
        for index in range(0, num_clusters):
                val_list = [x*word_centroid_prob_map[word][index] for x in model[word]]
                prob_wordvecs[word][index * num_features:(index + 1) * num_features] = val_list
    return prob_wordvecs


def create_cluster_vector_and_gwbowv(prob_wordvecs, wordlist, dimension,num_centroids):
    # This function computes SDV feature vectors.
    bag_of_centroids = np.zeros(num_centroids * dimension, dtype="float32")
    # print(bag_of_centroids
    for word in wordlist:
        try:
            bag_of_centroids += prob_wordvecs[word]
        except:
            pass
    norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
    if (norm != 0):
        bag_of_centroids /= norm
    return bag_of_centroids


# if __name__ == '__main__':
start = time.time()
num_features = 300  # int(sys.argv[1])     # Word vector dimensionality
word_vectors_list = []
model_wv_index2word = []
model ={}
textfile ="data/paragram_sl999_small.txt"
with open(textfile) as infile:
    for i in infile:
        each_word = i.split()[0]
        each_embedding = list(map(float,i.split()[1:]))
        model[each_word] = each_embedding
        word_vectors_list.append(each_embedding)
        model_wv_index2word.append(each_word)
word_vectors = np.array(word_vectors_list)
model_wv_index2word = model_wv_index2word
print("shape of word vectors: ",word_vectors.shape)
print(len(model))



def prob_wordvecs_parallel(param_each):
    n_clusteri = param_each[0]
    transform_n_nonzero_coefs = param_each[1]
    np.random.seed(101)
    print("number of k clusters ", str(n_clusteri))
    num_clusters = n_clusteri  # int(sys.argv[2])
    # Uncomment below line for creating new clusters.
    # transform_n_nonzero_coefs = 2
    idx, idx_proba = cluster_GMM_cluster(num_clusters, word_vectors)
    # Uncomment below lines for loading saved cluster assignments and probabaility of cluster assignments.
    idx_name = "data_files/gmm_latestclusmodel_len2alldata.pkl"
    idx_proba_name = "data_files/gmm_prob_latestclusmodel_len2alldata.pkl"
    # idx, idx_proba = read_GMM(idx_name, idx_proba_name)


    global model_wv_index2word
    # Create a Word / Index dictionary, mapping each vocabulary word to a cluster number
    word_centroid_map = dict(zip(model_wv_index2word, idx))
    global num_features
    word_centroid_prob_map = dict(zip(model_wv_index2word, idx_proba))
    prob_wordvecs = get_probability_word_vectors(word_centroid_map,word_centroid_prob_map, num_clusters,num_features)
    prob_wordvecs_name = "data_files/prob_wordvecs_ksvd_transform_n_nonzero_coefs_"+ str(transform_n_nonzero_coefs)+textfile.split("/")[-1]+"_all_words_"+str(n_clusteri)+".pkl"
    pickle.dump(prob_wordvecs,open(prob_wordvecs_name,"wb"))
    print(str(n_clusteri)+"_"+str(transform_n_nonzero_coefs)+"_done")

parameters = pd.read_csv("parameters_ksvd.csv")
cluster_coef_pairs = parameters[["ksvd clusters","non zero coefs"]].values.tolist()

Parallel(n_jobs=1)(delayed(prob_wordvecs_parallel)(para) for para in cluster_coef_pairs)
