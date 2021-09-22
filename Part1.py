# CI7520 - Assignment 1 : Classic Machine Learning
# Task I - Application : Load and overview the datas from wine dataset

import numpy as np #by using import we import necessary libraries
import pandas as pd
from sklearn.datasets import load_wine
import seaborn as sns
import matplotlib.pyplot as plt
wine = load_wine()

# visualize the dataset using pandas DataFrame
# Easy to visualize and understand
def table_form():
    data = np.c_[wine.data, wine.target]
    columns = np.append(wine.feature_names,["Target"])
    data_frame = pd.DataFrame(data, columns = columns) #using pandas as pd and borrwing dataframe function to present the data in tabular columns
    return data_frame

table_form = table_form()
assert table_form.shape == (len(wine.target), 14)

def py_cmd_viz():# Visualize the datasets in python consoles

    print("Keys of Wine Dataset: \n", wine.keys())# Keys of dataset
    print("Data Type:", type(wine.data))# dataset data type
    print("Target Type:", type(wine.target))
    print("Feature name:", wine.feature_names)# Feature names
    print("Description: \n", wine['DESCR'])# Description of Dataset
    print("Shape of data(n_samples, n_dimensions):", wine.data.shape)# shape of the data
    print("Number of classes:", len(wine.target_names))# No of classes
    print("Target names:", wine.target_names)# Target names
    print(table_form)# shows table form of the data suing pandas


py_cmd_viz()

