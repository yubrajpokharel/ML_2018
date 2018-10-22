#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 00:46:13 2018

@author: yubrajpokharel
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import datasets
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#missing data processing
from sklearn.preprocessing import Imputer
#axis 0 -- impute along column
#axis 1 -- impute along rows
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#importing categorized data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])

oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

#splitting data into training and test
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#feature Scaling
X_stanScalar = StandardScaler()
X_train = X_stanScalar.fit_transform(X_train)
X_test = X_stanScalar.transform(X_test)