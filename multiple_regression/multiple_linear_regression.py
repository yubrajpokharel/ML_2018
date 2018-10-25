# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 15:23:51 2018

@author: ypokhrel
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#import datasets
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding independent variable
labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])
oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()

#Avoiding the dummy varaible trap
X = X[:,  1: ]

#splitting data into training and test
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#fit multiple regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#pred the test set result
y_pred = regressor.predict(X_test)


#building optional model using backward elimation
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values= X, axis = 1)