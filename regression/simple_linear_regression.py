# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:11:17 2018

@author: ypokhrel
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import datasets
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#splitting data into training and test
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#graph
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("salary vs Expereince")
plt.xlabel("years of exp")
plt.ylabel("salary")
plt.show()