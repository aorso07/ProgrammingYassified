#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:12:31 2022

@author: anthonyorso
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns



#os.chdir("/Users/anthonyorso/Documents")


os.chdir("/Users/anthonyorso/Documents")

teams=pd.read_csv("teams.csv")

#print(teams.columns)
#print(teams.shape)

##Q1a

two_thousands = teams.loc[(teams.yearID >= 2000) & (teams.yearID <= 2009), ["yearID","W","L"]]

print(two_thousands)

#Q1b

#The below mutation is being initiated for future questions. 
#Total HRs combined HR + HRA

teams["totalHR"]=teams.HR+teams.HRA


chi200 = teams.loc[(teams.teamID == "CHN") & (teams.HR > 200),"W"]

print(len(chi200))

#The length of this variable is 6, so it happened six times

print(chi200.median())

#The median is 86.5 wins per season for the cubs when at least 200 HRs were achieved

#Q1c

seq = [i for i in range(1,59) for _ in range(4)]

seq.append(59)

print(len(seq))

years = list(range(1789,2022))
print(len(years))

terms = pd.DataFrame()
terms["yearID"]=years
terms["termID"]=seq


termSport=pd.merge(teams,terms,on="yearID")

#this does the equivalent of a SQL inner join. It will add the prez term 
#that matches with the concomitant year number


run_max=termSport[["totalHR","termID"]].groupby("termID").sum()

#This line of code selects just totalHR and termID and groups by termID
#The sum function will create a summary variable of total HRs per term

term_max=run_max[run_max.totalHR == max(run_max.totalHR)]
print(term_max)

#This code allows us to pull the entry where the max is achieved. 
#It happens with the 54th presidential term

#Q1d

leagueRuns=teams[teams.lgID.notna()].groupby(["yearID","lgID"],as_index=False)['totalHR'].sum()
print(leagueRuns)

lgplot=sns.FacetGrid(leagueRuns,col='lgID')
lgplot.map(sns.lineplot,"yearID","totalHR")
plt.show()


#Q1e


teams["winning_record"]=teams.apply(lambda teams: True if teams["W"]>teams["L"] else False, axis=1)

#I'm using a lambda function above to create the categorical variable

runWL= sns.scatterplot(data = teams,x="R",y="RA",hue="winning_record")
plt.show()

#This scatterplot shows a great division in the values, and the data would be 
#an excellent fit for a support vector machine algorithm
#There's a nice division where we can clearly see the proportion of 
#runs made versus runs the other team got is predictive of whether 
#the team had a winning record for the year

#### Q2 ######

flights = pd.read_csv("flights.csv")


#Q2a

#I'm starting by creating a factor variable that converts
#True or False values into an integer so I can add up the responses
#To find the proportion of cancelled flights
#If a value is NULL, the algorithm will assign it 1 (TRUE)
#Once we have made this transofmration, can we sum up the values and divide
#by the number of observations to find the proportion

flights["cancelled"]=flights['dep_time'].isnull().astype(int)

#No I will group by month and use the lambda function to find the proportion
#of cancelled flights with the "cancelled" variable

prop_cancel = flights.groupby("month").apply(lambda x: x["cancelled"].sum()/len(x))
print(prop_cancel)

#The resultant table shows the highest proportion of cancellations in February
#and December. February is one of the snowiest and iciest months of the year
#Thus, it makes sense it has the most cancellations. Likely due to inclement weather
#December also makes sense because the volume of traffic is enormous 
#for the holiday season in addition to winter weather. Cancellations will
#likely happen more for this reason. We also see high proportions in June and July.
#These coincide with high amounts of travel for summer vacations. Again
#More chaos often means more unpredictable flights.

#Q2b

#The below code is going to group data with a delay >60 then groupby the airport
#The lambda function will add everything up and divide by the number of observations
#ro find the avg delay by airport

late = flights[flights["dep_delay"]>60].groupby("origin").apply(lambda x: x["dep_delay"].sum()/len(x))
print(late)

#I later realized this was going out of the way. You can do the whole indexing
#and grouping and then later call the column to pull out and just call
#a simple mean() function! Produces the same result tho

late_again = flights[flights["dep_delay"]>60].groupby("origin")["dep_delay"].mean()
print(late_again)


#EWR    120.160238
#JFK    120.905726
#LGA    126.738812

#To no one's surprise, LaGuardia is the worst airport ever & has the longest delays

#Q2c 

#Because i know there are flight cancellations in the dataset I'll start by
#filter out NAs. After grouping by carrier, I'm going to provide a data summary
#with a lambda functiom to find the avg. Then I'll used indexing to pull
#The actual row number to see what carrier as the longest avg air time

fly_time = flights[flights["air_time"].notna()].groupby("carrier").apply(lambda x: x["air_time"].sum()/len(x))
print(fly_time[fly_time==fly_time.max()])

#Like above, this could have been accomplished by simply calling the variable
#One more time and adding .mean() at the end. The result is, again, the same

fly_again = flights[flights["air_time"].notna()].groupby("carrier")["air_time"].mean()
print(fly_again[fly_again==fly_again.max()])

#Both of these tell us the answer is:
    
#HA    623.087719


#Q2d

#THe below code will filter out cancelled flights to prevent NAs 
#and make sure the ones kept have delays 

delay0=flights.loc[(flights.dep_delay>0) & (flights.cancelled == 0)] 

#A simple displot can be called to produce what we want
#However, I'm going to add a col_wrap to prevent an itty bitty picture with 12
#graphs in one long row

delay_hist=sns.displot(delay0,x="dep_delay",col="month",col_wrap=4)
delay_hist.set(xlim=(0,700))
plt.show()

#Q2e

#The bottom code filters flights with delays >60 then groups them
#With the as_index= False method, we can turn the results into a full data frame
#to ease the plotting process. The summary is size() to count the number of flights
#that meet these criteria per carrier. The sort method will then show 
#which carriers had the most amounts of delays. The head() function will 
#select the top 5
top5 = flights[flights.dep_delay>60].groupby("carrier",as_index=False).size().sort_values(by="size",ascending=False).head()

#The top5 have been dumped into this list for filtering purposes
top5vals=["EV","B6","UA","DL","AA"]



delaytop5=flights[(flights.dep_delay>60)&(flights.carrier.isin(top5vals))]

#A simple bodxplot function will now show the distribution of values per carrier
#As we can see the medians across all carriers are roughly the same. 
#However American Airlines and Delta have the most egregious delays

sns.boxplot(data=delaytop5,x="dep_delay",y="carrier")
plt.show()


