import time
import warnings, random
from gensim.models import Word2Vec
import pandas as pd
import os
import time,pickle, pdb
from nltk.corpus import stopwords
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from KaggleWord2VecUtility import KaggleWord2VecUtility
from numpy import float32
import math
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.svm import SVC, LinearSVC
import pickle
import cPickle
from math import *
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from ksvd import ApproximateKSVD
from sklearn.decomposition import PCA

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


def dictionary_KSVD(num_clusters, word_vectors):
    # Initalize a ksvd object and use it for clustering.
    aksvd = ApproximateKSVD(n_components=num_clusters,transform_n_nonzero_coefs=num_clusters/2)
    dictionary = aksvd.fit(word_vectors).components_
    idx_proba = aksvd.transform(word_vectors)
    idx = np.argmax(idx_proba, axis=1)
    print "Clustering Done...", time.time() - start, "seconds"
    # Get probabilities of cluster assignments.
    # Dump cluster assignments and probability of cluster assignments.
    joblib.dump(idx, 'ksvd_latestclusmodel_len2alldata.pkl')
    print "Cluster Assignments Saved..."

    joblib.dump(idx_proba, 'ksvd_prob_latestclusmodel_len2alldata.pkl')
    print "Probabilities of Cluster Assignments Saved..."
    return (idx, idx_proba)


def dictionary_read_KSVD(idx_name, idx_proba_name):
    # Loads cluster assignments and probability of cluster assignments.
    idx = joblib.load(idx_name)
    idx_proba = joblib.load(idx_proba_name)
    print "Cluster Model Loaded..."
    return (idx, idx_proba)


def get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict):
    # This function computes probability word-cluster vectors.

    prob_wordvecs = {}

    for word in word_centroid_map:
        prob_wordvecs[word] = np.zeros(num_clusters * num_features, dtype="float32")
        for index in range(0, num_clusters):
            try:
                prob_wordvecs[word][index * num_features:(index + 1) * num_features] = model[word] * \
                                                                                       word_centroid_prob_map[word][
                                                                                           index] * word_idf_dict[word]
            except:
                continue
    return prob_wordvecs

def weight_building(weight_file,a_weight):
    f = open(weight_file,"rb")
    lines =  f.readlines()
    weight_dict = {}
    total = 0
    for line in lines:
        word,count = line.split()[:2]
        weight_dict[word] = int(count)
        total = total+int(count)
    for word in weight_dict:
        prob = weight_dict[word]*1.0/total
        weight_dict[word] = a_weight*1.0/(a_weight*1.0+prob)
    return weight_dict

def create_cluster_vector_and_gwbowv(prob_wordvecs, wordlist,  n_comp):
	# This function computes SDV feature vectors.
      bag_of_centroids = np.zeros(n_comp, dtype="float32" )
      #print bag_of_centroids
      for word in wordlist:
          try:
            bag_of_centroids += prob_wordvecs[word] #*weight_dict[word]
          except:
            pass
      norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
      if(norm!=0):
		bag_of_centroids /= norm
      return bag_of_centroids


def pca_truncated_svd(X, X_test,n_comp):
    sklearn_pca = PCA(n_components=n_comp,svd_solver='full')
    X_pca = sklearn_pca.fit_transform(X)
    X_pca_test = sklearn_pca.transform(X_test)
    del sklearn_pca
    return X_pca, X_pca_test

#if __name__ == '__main__':

start = time.time()

num_features = 200#int(sys.argv[1])  # Word vector dimensionality
min_word_count = 20  # Minimum word count
num_workers = 40  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words


# Load the trained Word2Vec model.
model_name = "Doc2VecC_twitter.txt"
WORD_EMBED_DIR_DOC2VEC = "doc2vecC"
model = KeyedVectors.load_word2vec_format(os.path.join(WORD_EMBED_DIR_DOC2VEC, model_name), binary=False)
word_vectors = model.syn0

print word_vectors.shape
#pdb.set_trace()
# Set number of clusters.
num_clusters = 40
idx, idx_proba = dictionary_KSVD(num_clusters, word_vectors)

for counti in range(10):
    data_all = pickle.load(open("data_all.pkl","r"))
    all_x,Y = [],[]
    for each in data_all:
        all_x.append(each[0])
        Y.append(each[1])
    train_data, test_data, Y_train, Y_test = train_test_split(all_x, Y, test_size=0.3, random_state=random.randint(1, 100))

    # Uncomment below line for creating new clusters.


    # Uncomment below lines for loading saved cluster assignments and probabaility of cluster assignments.
    idx_name = "ksvd_latestclusmodel_len2alldata.pkl"
    idx_proba_name = "ksvd_prob_latestclusmodel_len2alldata.pkl"
    #idx, idx_proba = dictionary_read_KSVD(idx_name, idx_proba_name)

    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip(model.wv.index2word, idx))
    # Create a Word / Probability of cluster assignment dictionary, mapping each vocabulary word to
    # list of probabilities of cluster assignments.
    word_centroid_prob_map = dict(zip(model.wv.index2word, idx_proba))

    # Computing tf-idf values.
    traindata = all_x

    tfv = TfidfVectorizer(strip_accents='unicode', dtype=np.float32)
    tfidfmatrix_traindata = tfv.fit_transform(traindata)
    featurenames = tfv.get_feature_names()
    idf = tfv._tfidf.idf_

    # Creating a dictionary with word mapped to its idf value
    print "Creating word-idf dictionary for Training set..."
    a_weight = 0.01
    weight_file = "data/20_news_vocab.txt"
    #weight_dict = weight_building(weight_file, a_weight)
    word_idf_dict = {}
    for pair in zip(featurenames, idf):
        word_idf_dict[pair[0]] = pair[1]

    # Pre-computing probability word-cluster vectors.
    prob_wordvecs = get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict)

    temp_time = time.time() - start
    print "Creating Document Vectors...:", temp_time, "seconds."

    # gwbowv is a matrix which contains normalised document vectors.
    gwbowv = np.zeros((len(train_data), num_clusters * (num_features)), dtype="float32")

    counter = 0
    n_comp = num_features*num_clusters
    for review in train_data:
        # Get the wordlist in each news article.
        words = review.split()
        gwbowv[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs,  words, n_comp)
        counter += 1
        if counter % 1000 == 0:
            print "Train News Covered : ", counter

    gwbowv_name = "SDV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_ksvd_sparse.npy"

    gwbowv_test = np.zeros((len(test_data), num_clusters * (num_features)), dtype="float32")

    counter = 0

    for review in test_data:
        # Get the wordlist in each news article.
        words = review.split()
        gwbowv_test[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, n_comp)
        counter += 1
        if counter % 1000 == 0:
            print "Test News Covered : ", counter

    test_gwbowv_name = "TEST_SDV_" + str(num_clusters) + "cluster_" + str(
        num_features) + "feature_matrix_ksvd_sparse.npy"

    #principal component removal
    gwbowv, gwbowv_test = pca_truncated_svd(gwbowv, gwbowv_test, n_comp-1)

    # saving gwbowv train and test matrices
    #np.save(gwbowv_name, gwbowv)
    #np.save(test_gwbowv_name, gwbowv_test)
    print gwbowv_test.shape
    print gwbowv.shape
    endtime = time.time() - start
    print "SDV created and dumped: ", endtime, "seconds."
    print "Fitting a SVM classifier on labeled training data..."

    param_grid = [
        {'C': np.arange(0.1, 5, 0.3)}]
    scores = ['accuracy']#, 'recall_micro', 'f1_micro', 'precision_micro', 'recall_macro', 'f1_macro', 'precision_macro',
              #'recall_weighted', 'f1_weighted', 'precision_weighted']  # , 'accuracy', 'recall', 'f1']
    for score in scores:
        strt = time.time()
        print num_clusters
        print "# Tuning hyper-parameters for", score, "\n"
        clf = GridSearchCV(LinearSVC(C=1), param_grid, cv=5, scoring='%s' % score)
        clf.fit(gwbowv, Y_train)
        print "Best parameters set found on development set:\n"
        print clf.best_params_
        print "Best value for ", score, ":\n"
        print clf.best_score_
        Y_true, Y_pred = Y_test, clf.predict(gwbowv_test)
        print "Report"
        print classification_report(Y_true, Y_pred, digits=6)
        print "Accuracy: ", clf.score(gwbowv_test, Y_test)
        print "Time taken:", time.time() - strt, "\n"
    endtime = time.time()
    print "Total time taken: ", endtime - start, "seconds."

    print "********************************************************"
