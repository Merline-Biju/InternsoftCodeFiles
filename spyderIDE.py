# -*- coding: utf-8 -*-
"""
Created on Sat May 22 23:20:25 2021

@author: Merline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

aaplCSV = pd.read_csv('AAPL Historical Data.csv', usecols=[0,1,2,3,4])

pohl_avg = aaplCSV[['Price','Open','High','Low']].mean(axis = 1)

dayNo = np.arange(1, len(aaplCSV) + 1, 1)

plt.plot(dayNo, pohl_avg, color='r', label='My First Plot "Day Number vs Avg POHL"')