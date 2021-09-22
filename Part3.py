# CI7520 - Assignment 1: Classic Machine Learning
#Part 3 - Classification Machine Learning

import numpy as np 
import pandas as pd 
from sklearn import datasets, metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve

ptint("This is task 3, machine learning classification")

#load the dataset
data = datasets.load_wine()

#inspect the shapes (rows and columns)
print(data['data'].shape)
print(data['target'].shape)

#Converting the data to a pandas dataframe to help visualise
wine = pd.DataFrame(data.data)
wine.columns = data.feature_names
wine['target']=data.target

#taking a look at the alcohol content in each class
for i in wine.target.unique():
    sns.histplot(wine['alcohol'][wine.target == i], kde=1, label = '{}'.format(i))

plt.legend()
plt.show()

#**** Running the models - Guassian Naive Bayes and Random Forest ***

#train and test data split

X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
print(X_train.shape)
print(X_test.shape)
kfold = model_selection.KFold(n_splits = 10)

#**** Gaussian Model ****
NB_model = GaussianNB()
NB_model.fit(X_train, Y_train)

#Guassian accuracy score
scoring_a = 'accuracy'
NB_a_results = model_selection.cross_val_score(NB_model, X_train, Y_train, cv=kfold, scoring=scoring_a)
print("Naive Bayes model has an accuracy score of: " + str(NB_a_results.mean()*100))

#Gaussian cinfusion matrix
NB_predict = NB_model.predict(X_test)
NB_matrix = confusion_matrix(Y_test, NB_predict)
print("This is Gasussian confusion matrix: ", "\n",  NB_matrix)
sns.heatmap(NB_matrix, annot=True)
plt.show()

#Gaussian classification report
NB_report = classification_report(Y_test, NB_predict)
print('This is the Guassian classification report: ')
print(NB_report)

#*** Random Forest Model *** 
RF_model = RandomForestClassifier()
RF_model.fit(X_train, Y_train)

#Random Forest accuracy score 
RF_a_results = model_selection.cross_val_score(RF_model, X_train, Y_train, cv=kfold, scoring=scoring_a)
print("Random Forest model has an accuracy score of: " +str(RF_a_results.mean()*100))

#Random Forest confusion matrix 
RF_predict = RF_model.predict(X_test)
RF_matrix = confusion_matrix(Y_test, RF_predict)
print("This is the Random Forest confusion matrix: ", "\n", RF_matrix)
sns.heatmap(RF_matrix, annot=True)
plt.show()

#Random Forest classification report 
RF_report = classification_report(Y_test, RF_predict)
print('This is the Random Forest classification report: ')
print(RF_report)


#**** seperate section for ROC Curves: ****
X1, y1 = datasets.load_wine(return_X_y=True)
y1 = y1 == 2

X_train, X_test, y_train, y_test = train_test_split(X1, y1, random_state=42)

#Gaussian ROC Curve  
GNB = GaussianNB()
GNB.fit(X_train, y_train)
GNB_disp = plot_roc_curve(GNB, X_test, y_test)

#Random Forest ROC Curve
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
GNB_disp.plot(ax=ax, alpha = 0.8)

#Both plots on the same graph
plt.show()
