#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:24:46 2022

@author: anthonyorso
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Q1

rct = pd.read_csv("rct.csv")

rct = rct.iloc[:,1:len(rct)]


numerics = ['int16', 'int32', 'int64','float64']

rct_num = rct.select_dtypes(include=numerics)

print(rct_num)

cols = list(rct_num.columns)


print(cols)

r_mn=rct_num.apply(np.mean,axis=0)
print(r_mn)

#Q1b

cuts = list(range(10,80,10))
vals = list(range(1,7))

rct['age_cut']= pd.cut(rct.age, cuts, right = False, labels = vals)





avg_num = rct.groupby(['sex','age_cut'],as_index=False).mean()
print(avg_num)

#Q1c

def viz(y):
    rct_plot=sns.FacetGrid(avg_num, col='sex')
    rct_plot.map(sns.lineplot, "age",y)
    plt.show()
    

col = rct_num.columns
cols = list(col[1:])

    

list(map(viz,cols))

#Q2

teams = pd.read_csv('teams.csv')                                            

bk_teams = ["BR1", "BR2", "BR3", "BR4", "BRO", "BRP", "BRF"]



def count_seasons(name):
   num_years = teams[teams.teamID==name].shape[0]
   print(name + " played " + str(num_years) + " seasons")
    
list(map(count_seasons,bk_teams))
    



