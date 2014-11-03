#!/usr/bin/env python
"""
======================================================
Classification of text documents using sparse features
======================================================

This is an example showing how scikit-learn can be used to classify documents
by topics using a bag-of-words approach. This example uses a scipy.sparse
matrix to store the features and demonstrates various classifiers that can
efficiently handle sparse matrices.

The dataset used in this example is the 20 newsgroups dataset. It will be
automatically downloaded, then cached.

The bar plot indicates the accuracy, training time (normalized) and test time
(normalized) of each classifier.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()


###############################################################################
# Load some categories from the training set
categories = None

remove = ('headers', 'footers', 'quotes')

print("Loading 20 newsgroups dataset for categories:")
data = fetch_20newsgroups(subset='test', categories=categories,
#data = fetch_20newsgroups(subset='all', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)
y = data.target

print('data loaded')

categories = data.target_names    # for case categories == None


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

data_size_mb = size_mb(data.data)

print("%d documents - %0.3fMB (data set)" % (
    len(data.data), data_size_mb))
print("%d categories" % len(categories))
print()

# split a training set and a test set
#y_train, y_test = data_train.target, data_test.target

print("Extracting features from the dataset using a sparse vectorizer")
t0 = time()
opts.use_hashing = True
opts.n_features = 5000
if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   n_features=opts.n_features)
    #X_train = vectorizer.transform(data_train.data)
    X = vectorizer.transform(data.data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X = vectorizer.fit_transform(data.data)
    #X_train = vectorizer.fit_transform(data_train.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_size_mb / duration))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    print("done in %fs" % (time() - t0))
    print()


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = np.asarray(vectorizer.get_feature_names())


###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)

    t0 = time()
    try:
        score = cross_validation.cross_val_score( clf, X.toarray(), y, cv=5)
    except:
        score = cross_validation.cross_val_score( clf, X, y, cv=5)
    test_time = time() - t0
    print("CV time:  %0.3fs" % test_time)

#    score = metrics.f1_score(y_test, pred)
    print("CV-score:   %s" % str(score))
    print("Mean CV-score:   %f" % np.mean(score))

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s"
                      % (category, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, np.mean(score)


results = []

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
#AAA
#results.append(benchmark(KNeighborsClassifier(n_neighbors=1)));
#results.append(benchmark(MultinomialNB(alpha=.01)))


# Test the Tree Classifier
results.append(benchmark(DecisionTreeClassifier(criterion="gini", splitter="best",max_depth=4)))
results.append(benchmark(DecisionTreeClassifier(criterion="gini", splitter="best",max_depth=5)))
results.append(benchmark(DecisionTreeClassifier(criterion="gini", splitter="best",max_depth=6)))
for i in range(7,17):
    results.append(benchmark(DecisionTreeClassifier(criterion="gini", splitter="best",max_depth=i)))
    #results.append(benchmark(DecisionTreeClassifier(criterion="entropy", splitter="best",max_depth=i)))
results.append(benchmark(DecisionTreeClassifier(criterion="gini", splitter="best")))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score = results
#training_time = np.array(training_time) / np.max(training_time)
#test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='r')
#plt.barh(indices + .3, training_time, .2, label="training time", color='g')
#plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)


print(clf_names)
print( indices)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
