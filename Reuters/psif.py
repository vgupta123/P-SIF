import time
import warnings
from gensim.models import Word2Vec
import pandas as pd
import time
from nltk.corpus import stopwords
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
from numpy import float32
import math
import sys
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.svm import SVC, LinearSVC
import pickle
import cPickle
from math import *
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import label_binarize
from ksvd import ApproximateKSVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from pandas import DataFrame

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


def dictionary_KSVD(num_clusters, word_vectors):
    # Initalize a ksvd object and use it for clustering.
    aksvd = ApproximateKSVD(n_components=num_clusters)
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

def create_cluster_vector_and_gwbowv(prob_wordvecs, weight_dict,wordlist,  n_comp):
	# This function computes SDV feature vectors.
      bag_of_centroids = np.zeros(n_comp, dtype="float32" )
      for word in wordlist:
          try:
            bag_of_centroids += prob_wordvecs[word]*weight_dict[word]
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

if __name__ == '__main__':

    start = time.time()

    num_features = int(sys.argv[1])  # Word vector dimensionality
    min_word_count = 20  # Minimum word count
    num_workers = 40  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    model_name = str(num_features) + "features_" + str(min_word_count) + "minwords_" + str(
        context) + "context_len2alldata"
    # Load the trained Word2Vec model.
    model = Word2Vec.load(model_name)
    # Get wordvectors for all words in vocabulary.
    word_vectors = model.wv.syn0

    # Load all data.
    all = pd.read_pickle('all.pkl')
    # Set number of clusters.
    num_clusters = int(sys.argv[2])
    # Uncomment below line for creating new clusters.
    #idx, idx_proba = dictionary_KSVD(num_clusters, word_vectors)

    # Uncomment below lines for loading saved cluster assignments and probabaility of cluster assignments.
    idx_name = "ksvd_latestclusmodel_len2alldata.pkl"
    idx_proba_name = "ksvd_prob_latestclusmodel_len2alldata.pkl"
    idx, idx_proba = dictionary_read_KSVD(idx_name, idx_proba_name)

    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip(model.wv.index2word, idx))
    # Create a Word / Probability of cluster assignment dictionary, mapping each vocabulary word to
    # list of probabilities of cluster assignments.
    word_centroid_prob_map = dict(zip(model.wv.index2word, idx_proba))

    #building weighting dictionary for sif weighting
    a_weight = 0.01
    weight_file = "data/reuters_vocab.txt"
    weight_dict = weight_building(weight_file, a_weight)

    # Computing tf-idf values.
    traindata = []
    for i in range(0, len(all["text"])):
        traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(all["text"][i], True)))

    tfv = TfidfVectorizer(strip_accents='unicode', dtype=np.float32)
    tfidfmatrix_traindata = tfv.fit_transform(traindata)
    featurenames = tfv.get_feature_names()
    idf = tfv._tfidf.idf_

    # Creating a dictionary with word mapped to its idf value
    print "Creating word-idf dictionary for Training set..."

    word_idf_dict = {}
    for pair in zip(featurenames, idf):
        word_idf_dict[pair[0]] = pair[1]

    # Pre-computing probability word-cluster vectors.
    prob_wordvecs = get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict)

    temp_time = time.time() - start
    print "Creating Document Vectors...:", temp_time, "seconds."
    # Create train and text data.
    lb = MultiLabelBinarizer()
    Y = lb.fit_transform(all.tags)
    train_data, test_data, Y_train, Y_test = train_test_split(all["text"], Y, test_size=0.3, random_state=42)

    train = DataFrame({'text': []})
    test = DataFrame({'text': []})

    train["text"] = train_data.reset_index(drop=True)
    test["text"] = test_data.reset_index(drop=True)
    # gwbowv is a matrix which contains normalised document vectors.
    gwbowv = np.zeros((train["text"].size, num_clusters * (num_features)), dtype="float32")

    counter = 0
    n_comp = num_features*num_clusters
    for review in train["text"]:
        # Get the wordlist in each news article.
        words = KaggleWord2VecUtility.review_to_wordlist(review, \
                                                         remove_stopwords=True)
        gwbowv[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, weight_dict, words, n_comp)
        counter += 1
        if counter % 1000 == 0:
            print "Train text Covered : ", counter

    gwbowv_name = "SDV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_ksvd_sparse.npy"

    gwbowv_test = np.zeros((test["text"].size, num_clusters * (num_features)), dtype="float32")

    counter = 0

    for review in test["text"]:
        # Get the wordlist in each news article.
        words = KaggleWord2VecUtility.review_to_wordlist(review, \
                                                         remove_stopwords=True)
        gwbowv_test[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, weight_dict, words, n_comp)
        counter += 1
        if counter % 1000 == 0:
            print "Test text Covered : ", counter

    test_gwbowv_name = "TEST_SDV_" + str(num_clusters) + "cluster_" + str(
        num_features) + "feature_matrix_ksvd_sparse.npy"

    gwbowv, gwbowv_test = pca_truncated_svd(gwbowv, gwbowv_test, n_comp-1)

    # saving gwbowv train and test matrices
    np.save(gwbowv_name, gwbowv)
    np.save(test_gwbowv_name, gwbowv_test)

    endtime = time.time() - start
    print "SDV created and dumped: ", endtime, "seconds."

    print "********************************************************"

