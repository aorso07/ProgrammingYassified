#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 16:14:22 2022

@author: anthonyorso
"""

import pandas as pd
import re

batting = pd.read_csv("batting.csv")
people = pd.read_csv("people.csv")
pitching = pd.read_csv("pitching.csv")

# print(batting.shape)
# print(people.shape)
# print(pitching.shape)

#Q1a

bat = batting[["playerID","HR","SB"]] .groupby("playerID", as_index = False).sum()

bat300 = bat[(bat["HR"]>=300) & (bat["SB"]>=300)]

name300 = pd.merge(bat300,people, on = "playerID", how = "inner")[["nameLast","nameFirst","HR","SB"]].sort_values(by="nameLast")

print(name300)

print()

#Q1b

pitch = pitching[["playerID","W","SO"]].groupby("playerID", as_index = False).sum()
pitch300 = pitch[(pitch["W"]>=300) & (pitch["SO"]>=3000)]


pname300 = pd.merge(pitch300,people, on = "playerID", how = "inner")[["nameLast","nameFirst","W","SO"]].sort_values(by="nameLast")
print(pname300)

#Q1c

#Pete Alonso!

batC = batting[["yearID","playerID","AB","H","HR"]].groupby(["playerID","yearID"], as_index = False).sum()

jointable = pd.merge(batC,people, on = "playerID", how = "inner")

by50 = jointable[jointable.HR >= 50]

by50 = by50.dropna(subset=["H","AB"])

by50["BA"]=by50["H"]/by50["AB"]

min50 = by50[by50.BA==by50.BA.min()][["nameFirst","nameLast"]]
print(min50)

#Q2

plane = pd.read_csv("planes.csv")
flights = pd.read_csv("flights.csv")


#Q2a

pmerge = pd.merge(plane,flights,on="tailnum", how= "inner")

pmerge = pmerge.dropna(subset=["year_x"])

old = pmerge[pmerge.year_x==pmerge.year_x.min()][["tailnum","year_x"]]
print(old)

#It's airplane N381AA made in 1956


#Q2b

allFlight = flights.groupby("tailnum", as_index = False).size()

print(allFlight)

intersect = pd.merge(plane,allFlight, how = "left", on = "tailnum")

print(intersect.shape)

print(plane.shape)

print(intersect["size"].isna().sum())

#The answer is all of them!

#Q3

info = {"grp" : ["A","A","B","B"],
      "sex" : ["F","M","F","M"],
      "meanL" : [0.225,0.47,0.325,0.547],
      "sdL" : [0.106,.325,.106,.308],
      "meanR" : [.34,.57,.4,.647],
      "sdR" : [0.0849, 0.325, 0.0707, 0.274]} 

df = pd.DataFrame(info)


df_pivot = pd.pivot(df, index = "grp", columns= "sex", values=["meanL","sdL","meanR","sdR"])

print(df_pivot)

#Q4a

pc = pd.read_csv("https://raw.github.com/gjm112/DSCI401/main/data/pccc_icd10_dataset.csv")

print(pc.head)

pc2 = pc.loc[:,~pc.columns.str.contains('^g')]

#Q4b


pc2.columns = pc2.columns.str.replace(r'[0-9]+','')

#print(pc2.columns)

pc3 = pd.melt(pc2, id_vars = "id", value_vars = ["dx","pc"], value_name = "code", var_name = "type")

print(pc3)




