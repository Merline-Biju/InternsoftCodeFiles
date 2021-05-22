# -*- coding: utf-8 -*-
"""
Created on Sat May 22 21:30:41 2021

@author: Merline
"""

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Reading the data from advertising.csv File
data = pd.read_csv('advertising.csv')
print(data.head())


#Visualizing data
fig, axs = plt.subplots(1, 3, sharey = True)
data.plot(kind = 'scatter', x = 'TV', y ='Sales', ax = axs[0], figsize = (14,7))
data.plot(kind = 'scatter', x = 'Radio', y ='Sales', ax = axs[1])
data.plot(kind = 'scatter', x = 'Newspaper', y ='Sales', ax = axs[2])


#Creating x and y for linear regression
feature_cols = ['TV']
X = data[feature_cols]
Y = data.Sales

#Importing Linear regression algo for simple linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
print(lr.fit(X,Y))

print(lr.intercept_)
print(lr.coef_)

# y = mx + c
result = 6.974821488229885 + 0.05546477 * 50
print(result)

#Create dataframe with min and max value of the table
X_new = pd.DataFrame({'TV': [data.TV.min(), data.TV.max()]})
print(X_new.head())

preds = lr.predict(X_new)
preds

data.plot(kind = 'scatter', x = 'TV', y = 'Sales')
plt.plot(X_new, preds, c='red', linewidth = 3)


#importing statsmodel.formula.api
import statsmodels.formula.api as smf
lm = smf.ols(formula = 'Sales ~ TV', data = data).fit()
lm.conf_int()


#Finding the probability values
lm.pvalues

#Finding the R-squared values
lm.rsquared

#Multi Linear Regression
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
Y = data.Sales

lr = LinearRegression()
print(lr.fit(X,Y))

print(lr.intercept_)
print(lr.coef_)

lm = smf.ols(formula = 'Sales ~ TV + Radio + Newspaper', data = data).fit()
lm.conf_int()
lm.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.903
Model:                            OLS   Adj. R-squared:                  0.901
Method:                 Least Squares   F-statistic:                     605.4
Date:                Mon, 11 Jan 2021   Prob (F-statistic):           8.13e-99
Time:                        20:07:35   Log-Likelihood:                -383.34
No. Observations:                 200   AIC:                             774.7
Df Residuals:                     196   BIC:                             787.9
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      4.6251      0.308     15.041      0.000       4.019       5.232
TV             0.0544      0.001     39.592      0.000       0.052       0.057
Radio          0.1070      0.008     12.604      0.000       0.090       0.124
Newspaper      0.0003      0.006      0.058      0.954      -0.011       0.012
==============================================================================
Omnibus:                       16.081   Durbin-Watson:                   2.251
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               27.655
Skew:                          -0.431   Prob(JB):                     9.88e-07
Kurtosis:                       4.605   Cond. No.                         454.
==============================================================================
Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""



lm = smf.ols(formula = 'Sales ~ TV + Radio', data = data).fit()
lm.conf_int()
lm.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.903
Model:                            OLS   Adj. R-squared:                  0.902
Method:                 Least Squares   F-statistic:                     912.7
Date:                Mon, 11 Jan 2021   Prob (F-statistic):          2.39e-100
Time:                        20:09:02   Log-Likelihood:                -383.34
No. Observations:                 200   AIC:                             772.7
Df Residuals:                     197   BIC:                             782.6
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      4.6309      0.290     15.952      0.000       4.058       5.203
TV             0.0544      0.001     39.726      0.000       0.052       0.057
Radio          0.1072      0.008     13.522      0.000       0.092       0.123
==============================================================================
Omnibus:                       16.227   Durbin-Watson:                   2.252
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               27.973
Skew:                          -0.434   Prob(JB):                     8.43e-07
Kurtosis:                       4.613   Cond. No.                         425.
==============================================================================
Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""