# CI7520 - Assignment 1: Classic Machine Learning
# Part II - Application: kmeans Clustering

print("Clustering of data from wine dataset using KMeans clustering")

#importing necessary libraries and naming them to convenience

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics

from sklearn.cluster import Birch
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

#**<< We have pulled this code from https://github.com/Ayantika22/Kmeans-and-HCA-clustering-Visualization-for-WINE-dataset >>***

wine = datasets.load_wine()  #loading the dataset
wine_data = pd.DataFrame(wine.data) #using a dataframe from pandas library
wine_data.columns = wine.feature_names
wine_data['Type']=wine.target
wine_data.head()
wine_X = wine_data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].values

X = wine.data
y = wine.target

cluster_Kmeans = KMeans(n_clusters=5) # number of clusters
model_kmeans = cluster_Kmeans.fit(wine_X)# kmeans fits for wine_X
prediction_kmeans = model_kmeans.labels_
prediction_kmeans

kmeans = cluster_Kmeans #clustering starts here
centroids = kmeans.cluster_centers_ #number of centroids
labels = kmeans.labels_ #labels
count_clusters = len(set(labels))


def visual():  #function to display the visual plot of the clusters
    plt.scatter(wine_X[prediction_kmeans == 0, 0], wine_X[prediction_kmeans == 0, 12], s = 80, c = '', label = 'Type 0')
    plt.scatter(wine_X[prediction_kmeans == 1, 0], wine_X[prediction_kmeans == 1, 12], s = 80, c = 'yellow', label = 'Type 1')
    plt.scatter(wine_X[prediction_kmeans == 2, 0], wine_X[prediction_kmeans == 2, 12], s = 80, c = 'green', label = 'Type 2')
    plt.title('Kmeans Clustering plot for Wine dataset')
    plt.legend()
    plt.show()


def evaluation_score_metrics(): #function to display the metrics values on the terminal window
    value_1 = (labels,)
    x = (X,) #declaring variable x as a tuple of X, tuples are easy to zip
    print(80* ' ')
    shapes = (X.shape, )
    titles = ['metrics',]
    count_cluster = (count_clusters,)
    print(80* '&')
    print('KMeans\t\tn_feat\tclustr\tHomo\tCompl\tARI\tV-measure\tSilhouette')
    print(80* '&')
    for labelu, title, x,shape, clustrr in zip(value_1, titles, x, shapes, count_cluster): #all the values are zipped and printed respectively below each variable name.
        print(title,'%s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t\t%.3f\t' % (shape,
                                                                    clustrr,
                                                                    metrics.homogeneity_score(y, labelu),
                                                                    metrics.completeness_score(y, labelu),
                                                                    metrics.adjusted_rand_score(y, labelu),
                                                                    metrics.v_measure_score(y, labelu),
                                                                    metrics.silhouette_score(x, labelu)))
    print(80*'&')

visual()
evaluation_score_metrics() #two fucnitons are then called finally to show the output

#**************************************************************************************************************
# CI7520 - Assignment 1: Classic Machine Learning
# Part II - Application: Birch Clustering

print("Clustering of data from wine dataset using Birch cluster")

# Load Datasets
wine = load_wine()
X = wine.data
y = wine.target


# Initiate the Birch cluster
birch = Birch(threshold=1.5, n_clusters=3, branching_factor=30) # parameters that effect birch type clustering
bc = birch.fit(X)


# Labels
labels = bc.labels_
count_clusters = bc.n_clusters
centroids = bc.subcluster_centers_

def evaluation_score_metrics():  # evaluation of score using metrics
    value_1 = (labels,) #a tuple is created called value_1
    x = (X,) #declaring variable x as a tuple of X, tuples are easy to zip
    print(80* ' ')
    shapes = (X.shape,)
    titles = ('clustering using BIRCH')
    count_cluster = (count_clusters,)
    print(80* '&')
    print('BIRCH\t\tn_feat\tclustr\tHomo\tCompl\tARI\tV-measure\tSilhouette')
    print(80* '&')
    for labelz, title, x,shape, clustr in zip(value_1, titles, x, shapes, count_cluster):
        print(title,'\t%s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t\t%.3f\t' % (shape,
                                                                    clustr,
                                                                    metrics.homogeneity_score(y, labelz),
                                                                    metrics.completeness_score(y, labelz),
                                                                    metrics.adjusted_rand_score(y, labelz),
                                                                    metrics.v_measure_score(y, labelz),
                                                                    metrics.silhouette_score(x, labelz)))
    print(80*'&')
#
def visual(): #function to display the visual plot of the clusters
    centroidz = (centroids,)
    z = (X,)
    label_ = (labels,)
    titles = ('clustering using Birch',)
    subfig = 121

    for centroid, labelz, Z, titlez in zip(centroidz, label_, z, titles):

        plt.scatter(Z[labelz == 0,0], Z[labelz ==0,1],s=60, edgecolor='black', label='class_0')
        plt.scatter(Z[labelz == 1,0], Z[labelz == 1,1],s=60, edgecolor='black', label='class_1')
        plt.scatter(Z[labelz == 2,0], Z[labelz == 2,1],s=60, edgecolor='black', label='class_2')
        plt.title(titlez)

        plt.legend()
    plt.show()

visual()
evaluation_score_metrics() #two fucnitons are then called finally to show the output
