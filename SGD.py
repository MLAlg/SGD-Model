#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 07:36:31 2019

@author: samaneh
"""
#load data from sklearn
from sklearn import datasets
iris = datasets.load_iris()
x_iris, y_iris = iris.data, iris.target
print(x_iris.shape, y_iris.shape)
print(x_iris[0], y_iris[0])

#build traning dataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
x, y = x_iris[:, :2], y_iris # get dataset with only the first two attributes
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33) # Split the dataset into a training and a testing set, Test set will be the 25% taken randomly
print(x_train.shape, y_train.shape)

#standardize the features
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#Plot
import matplotlib.pyplot as plt
colors = ['red', 'green', 'blue']
for i in range(len(colors)):
    xs = x_train[:, 0][y_train == i]
    ys = x_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

# create object classifier
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(x_train, y_train)
print(clf.coef_)
print(clf.intercept_)
'''
# draw hyperplanes
x_min, x_max = x_train[:, 0].min() - .5, x_train[:, 0].max() + .5
y_min, y_max = x_train[:, 1].min() - .5, x_train[:, 1].max() + .5
xs = np.arange(x_min, x_max, .5)
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 6)
for i in [0, 1, 2]:
    axes[i].set_aspect('equal')
    axes[i].set_title('Class '+str(i)+ 'versus the rest')
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal width')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max) 
    sca(axes[i])
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.prism)
    ys = (-clf.intercept_[i] - xs*clf.coef_[i, 0]/clf.coef_[i, 1])
    plt.plot(xs, ys, hold=True) 
'''
# predict a new data target
print(clf.predict(scaler.transform([[4.7, 3.1]]))) #2D array
print(clf.decision_function(scaler.transform([[4.7, 3.1]])))
 
# Evaluating the results on training data
from sklearn import metrics
y_train_pred = clf.predict(x_train)
print(metrics.accuracy_score(y_train, y_train_pred))

# Evaluating the results on test data
y_pred = clf.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))

# Evaluation functions: precision, recall, F1-score
print(metrics.classification_report(y_test, y_pred, target_names=iris.target_names))
# Evaluation function: confusion matrix
print(metrics.confusion_matrix(y_test,y_pred))
# Evaluation function: cross_validation
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
clf = Pipeline([('scaler', StandardScaler()),('linear_model', SGDClassifier())]) # create a composite estimator made by a pipeline of the standarization and the linear model
cv = KFold(n_splits=5, shuffle=True, random_state=33) # create a k-fold cross validation iterator of k=5 folds
scores = cross_val_score(clf, x, y, cv=cv) # by default the score used is the one returned by score method of the estimator (accuracy)
print(scores)
from scipy.stats import sem # calculate mean of scores and standard error
def mean_score(scores):
    return("Mean score: {0: .3f} (+/- {1: .3f})").format(np.mean(scores), sem(scores))
print(mean_score(scores))









































