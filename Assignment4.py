#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:58:52 2018

@author: demaggis
"""
# Data and Packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Importing the data
wine = pd.read_csv("wines_properties.csv")
wine.describe()
wine.head()

# ANSWER 1
customer = pd.get_dummies(wine["Customer_Segment"], prefix = "cs")
customer.head()

#segmenting data
regression = wine[ ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline'] ].join(customer.loc[:, "cs_2" :])
regression.head()

# Fit and summarize OLS model
regression = sm.add_constant(regression, prepend = False)
reg = sm.OLS(endog = regression["Alcohol"], exog = regression[regression.columns[1:]])
result = reg.fit()
print(result.summary())


# comment
# With this regression we are predicting the percentage of alcohol in the wine depending on
# the other variables given. Selecting a few of the more significant observations, we see
# Hue has a large effect on alcohol percentage with an increase of 35% for each single 
# increase in hue. Similarly, Ash has a large negative effect with a 28.5% decrease in alcohol
# for each single increase in Ash.
#-----------------------------------------------------------------------------------

# ANSWER 2
# Logistic regression
regression.describe()

regression["Strong"] = 0
for x in range(len(regression)):
    if regression.iloc[x, 0] > 13.7:
        regression.iloc[x, 16] = 1
logit = sm.Logit(regression["Strong"], regression[regression.columns[1 : -1]])
model = logit.fit()
model.summary2()

# comment
# With the Logit regression we are predicting if a wine will be "Strong" or "weak
# depending on a dummy variable that we created based on the 75th percentile of 
# alcohol included in wines. Our results show that Hue again has the biggest increase
# on the log odds of being a strong wine. Nonflavanoid_Phenols is next with the largest
# increase on log odds of a wine being strong or not.
#-----------------------------------------------------------------------------------

# ANSWER 3
# PCA
wine_s = StandardScaler().fit_transform(wine.iloc[:, 1:13])
matrix = np.corrcoef(wine_s.T)
e_values, e_vectors = np.linalg.eig(matrix)
eigen_values = sum(e_values)
sort_eig = sorted(e_values, reverse = True)
variance = [(i / eigen_values) * 100 for i in sort_eig]
np.cumsum(variance)

eigenvector_values = [ (np.abs(e_values[i]), e_vectors[:, i]) for i in range(len(e_values))]
eigenvector_values.sort()
eigenvector_values.reverse()
eigen4 = np.hstack( (eigenvector_values[0][1].reshape(12, -1), eigenvector_values[1][1].reshape(12, -1), eigenvectorvalues[2][1].reshape(12, -1), eigenvectorvalues[3][1].reshape(12, -1)) )
new_data = pd.DataFrame(wine_s.dot(eigen4))
new_data.columns = ["Component {}".format(i) for i in range(1, 5)]
new_data = new_data.join(wine["Alcohol"]).join(customer.loc[:, "cs_2" :])
variables = new_data.columns.tolist()
variables = variables[4:5]+variables[:4]+variables[5:]
new_data = new_data[variables]
new_data.head()

# Linear regression with PCA components
new_data = sm.add_constant(new_data, prepend = False)
reg = sm.OLS(endog = new_data["Alcohol"], exog = new_data[new_data.columns[1:]])
results = reg.fit()
print(results.summary())

# Logit regression with PCA components 
new_data["Strong"] = 0
for x in range(len(new_data)):
    if new_data.iloc[x, 0] > 13.7:
        new_data.iloc[x, 8] = 1
logit = sm.Logit(new_data["Strong"], new_data[new_data.columns[1 : -1]])
model = logit.fit()
model.summary2()


# comment
# The R-squared variables for the first two regressions are relatively high. Our
# Linear regression model has a R-sqaured of .765 and our Logit has one of .447.
# Our R-squared values decreased margianlly after the PCA which is probably due 
# to the decrease in variability of our inputs. Nonetheless, they are still high
# enough to give us a strong likelihood of accuracy. 