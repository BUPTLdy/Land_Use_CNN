# -*- coding: utf-8 -*-
"""
Created on Mon May 16 20:41:24 2016

@author: ldy
"""



from time import time
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

from sklearn.svm import SVC

import numpy as np

X=np.load('features.npy')
y=np.load('labels.npy')
#y=y.reshape((-1,1))

X=X.reshape((2100,4096))
print X.shape,y.shape
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Fitting the classifier to the training set")
t0 = time()
clf = SVC(C=10000,probability=True).fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test)

print "Accuracy: %.3f" % accuracy_score(y_test, y_pred)
joblib.dump(clf, 'svm_model/svm.pkl') 

