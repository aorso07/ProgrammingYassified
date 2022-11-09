# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 00:31:07 2022

@author: mfros
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import random

# basic file IO(input/output) and verify that it was loaded correctly
main = pd.read_csv('titanic_traintest.csv', index_col = 0)

validation = pd.read_csv('test (2).csv',index_col=0)


#print out first 5 rows of data file to confirm it read correctly
pd.set_option('max_columns', None)

print(main.head())

# exploration (and the appropriate visualization)
# in particular, it happens a lot during data cleaning and EDA
# for a reminder and discussion of these ideas, review the 030 and 040 videos

#high - level description of the training file variables
# https://towardsdatascience.com/data-types-in-statistics-347e152e8bee
print(main.info())
print(main.describe())
print(main.shape)
print(main.columns)
print(main.index)
#print(main.sort_values(['Sex','Age']))

#practice series or dataframe
type(main["Ticket"]) #series
type(main[["Ticket","Fare"]]) #dataframe

#slice
print(main[0:10])
#select one row
main.loc[5]

#select multiple rows as dataframe
main.loc[[5,6]]

#select multiple rows and columns as dataframe
main.loc[[5,6],["Name","Ticket"]]

#select all rows with specific columns
main.loc[:,["Ticket","Fare"]]
main.loc[0:10,["Ticket","Fare"]]

#select multiple rows as dataframe
main.iloc[[5,6]]
main.iloc[0:10,[7,8]]


#########################################################################
#Sorting and Subsetting
#########################################################################
#Sorting values
print(main.sort_values(["Age", "Fare"], ascending = [False, False]))

#Filtering 
print(main[main["Age"] > 60]) # can also be used for dates
print(main[main["Cabin"] == "C23 C25 C27"])
print(main[(main["Age"] > 60) & (main["Cabin"] == "C23 C25 C27")]) #filter both conditions adding parenthesis around each condition
print(main[main["Cabin"].isin(["B30","A5","B19","C23 C25 C27"])])

#Aggregating Data
#Summary Statistics
print("Fare Statistics")
print(main["Fare"].mean())
print(main["Fare"].median())
print(main["Fare"].mode())
print(main["Fare"].min())
print(main["Fare"].max())
print(main["Fare"].var())
print(main["Fare"].std())
print(main["Fare"].sum())
print(main["Fare"].quantile(.4))

#using agg
#function to return 30th percentile
def pct30(column):
    return column.quantile(0.3)
print(main["Fare"].agg(pct30))

#use agg for more than one column
print(main[["Fare","Age"]].agg(pct30))

#use agg to call multiple functions
def pct40(column):
    return column.quantile(0.4)
print(main[["Fare","Age"]].agg([pct30,pct40]))

#using cumsum
print(main[["Fare","Age"]].cumsum()) #can also do cummax(), cummin(), cumprod()

#counting categorical variable
print(main.drop_duplicates(subset = ["Cabin","Embarked"]))

#count age & cabin counts
print(main["Age"].value_counts())
print(main["Cabin"].value_counts(sort = True))
print(main[["Sex","Survived"]].value_counts(sort = False))
print(main[["Sex", "Survived"]].value_counts(sort = False,normalize = True)) #get proportions
print(main[["Embarked","Sex","Survived"]].value_counts(sort = False))
print(main[["Embarked","Sex", "Survived"]].value_counts(sort = False,normalize = True)) #get proportions

#group by method
print(main.groupby("Sex")["Age"].median())
print(main.groupby("Sex")["Age"].agg([np.min, np.max, np.median, np.mean]))
print(main.groupby("Sex")["Survived"].sum()) # Survived = 1
print(main.groupby("Sex")["Survived"].size()) # count by sex
print(main.groupby(["Sex","Survived"]).size()) # count of survived by sex

#pivot table 
print(main.pivot_table(index = ["Sex", "Survived"] \
                       , aggfunc = 'size'))
print(main.pivot_table(index = ["Age", "Survived"], \
                       aggfunc = 'size',dropna= True))
print(main[main["Age"] > 60].pivot_table(index = ["Age", "Survived"], \
                       aggfunc = 'size',dropna= True))
####################################################################################    
# Data Visualization
####################################################################################    
#histogram for male and female passenger age distribution
main[main["Sex"]=="male"]["Age"].hist(bins=10, alpha=0.7)
main[main["Sex"]=="female"]["Age"].hist(bins=10, alpha=0.7)
plt.ylabel("Number of Passengers")
plt.xlabel("Passengers by Age")
plt.title("Distribution of Ages by Gender")
plt.legend(["Male","Female"])
plt.show()

#histogram for passengers with siblings distribution
main["SibSp"].hist(bins=10)
plt.ylabel("Number of Passengers")
plt.xlabel("Passengers by Siblings")
plt.title("Distribution of Passengers with Siblings")
#plt.legend(["Male","Female"])
plt.show()

#histogram for passengers with parch (parents) distribution
main["Parch"].hist(bins=10)
plt.ylabel("Number of Passengers")
plt.xlabel("Passengers by Families")
plt.title("Distribution of Passengers with Families")
#plt.legend(["Male","Female"])
plt.show()


#bar chart of passengers by sex and died v/s survived
print(main.groupby(["Sex","Survived"]).size())
main.groupby(["Sex","Survived"]).size().unstack().plot(kind = "bar", figsize = (8,6))
plt.xticks( rotation = 'horizontal')
plt.ylabel("Number of Passengers")
plt.xlabel("Total Number of Passengers by Gender")
plt.title("Number of Passengers by Gender- Deaths and Survivors")
plt.legend(['Died','Survived'])
plt.show()
#plt.figure().clear()
#plt.clf()
#plt.cla()
#plt.close()


#bar chart of passengers by port of embarkment and class to supplement death/survived visual
main.groupby(["Embarked","Pclass"]).size().unstack().plot(kind = "bar", figsize = (8,6))
plt.xticks(rotation = 'horizontal')
plt.ylabel("Total Number of Passengers")
plt.xlabel("Total Number of Passengers by Port")
plt.title("Number of Passengers by Port - Passenger Class")
plt.legend(['1','2','3'])
plt.show()

#bar chart of passengers by port of embarkment and died v/s survived
print(main.groupby(["Embarked","Survived"]).size())
main.groupby(["Embarked","Survived"]).size().unstack().plot(kind = "bar", figsize = (8,6))
plt.xticks( rotation = 'horizontal')
plt.ylabel("Number of Passengers")
plt.xlabel("Total Number of Passengers by port of embarkment")
plt.title("Number of Passengers by port of embarkment- Deaths and Survivors")
plt.legend(['Died','Survived'])
plt.show()

#bar chart of number of siblings/spouse not in the first class vs died/survived

notrich_main=main[main['Pclass']!=1]
print(notrich_main.shape)


notrich_main.groupby(["SibSp","Survived"]).size().unstack().plot(kind = "bar", figsize = (8,6))
plt.xticks(rotation = 'horizontal')
plt.ylabel("Total Number of Passengers")
plt.xlabel("Total Number of Passengers by Sibling/Spouse Count")
plt.title("Number of Passengers by Sibling/Spouse Count (Not in 1st Class) - Survived or Died")
plt.legend(['Died','Survived'])
plt.show()

#bar chart of number of siblings/spouse in the first class vs died/survived

rich_main=main[main['Pclass']==1]
print(rich_main.shape)


rich_main.groupby(["SibSp","Survived"]).size().unstack().plot(kind = "bar", figsize = (8,6))
plt.xticks(rotation = 'horizontal')
plt.ylabel("Total Number of Passengers")
plt.xlabel("Total Number of Passengers by Sibling/Spouse Count")
plt.title("Number of Passengers by Sibling/Spouse Count (Not in 1st Class) - Survived or Died")
plt.legend(['Died','Survived'])
plt.show()


###

main[main["Age"] > 0].plot(x = "Age", \
                y = 'Fare', kind = "scatter", figsize = (8,6))
#plt.xticks( rotation = 'horizontal')
plt.ylabel("Fare")
plt.xlabel("Age")
plt.title("Distribution of Fare By Passenger Class")
#plt.legend(['Died','Survived'])
plt.show()

#Corr Matrix



##############################################################################
# Feature Engineering/Lambda
##############################################################################
#Add new Columns
#main["AgeInWeeks"] = main["Age"] * 52
#print(main.head())

#cabin_count column will be the number of assigned cabins for each person
main["cabin_count"] = main.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))

validation["cabin_count"] = validation.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))

print(main["cabin_count"].value_counts(sort = False))
print(main.pivot_table(index ="Survived", columns = "cabin_count", values = "Ticket", \
                       aggfunc = 'count'))
print(main.pivot_table(index ="Survived", columns = "cabin_count",  \
                       aggfunc = 'size'))  
    
# Get the location of each cabin based on the first letter of the cabin name
# hoping the cabin location is a useful indicator of survival
main["cabin_letter"] = main.Cabin.apply(lambda x: str(x)[0])
validation["cabin_letter"] = validation.Cabin.apply(lambda x: str(x)[0])

print(main.cabin_letter.value_counts())
print(main.pivot_table(index ="Survived", columns = "cabin_letter",  \
                       aggfunc = 'size'))  
    
#Exploration of names to discover titles that may be useful
print(main.Name.head(50))
main["title"] = main.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
validation["title"] = validation.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())


print(main.title.value_counts())
print(main.pivot_table(index ="Survived", columns = "title",  \
                       aggfunc = 'size'))  
                       
#Correlation Matrix based on new features
                       
sns.heatmap(main.corr(),annot=True,fmt=".2f")
plt.title('Correlation Matrix of Numerical Measures')
plt.show()
    
#####################
# see how many of our variables have null (or empty) values that we might need to fix.
print(main.isna())
print(main.isna().any())
print(main.isna().sum())
print(main.isnull().sum())
main.isnull().sum().plot(kind = "bar")
plt.show()



# total passenger list organized by survival
slicedby_survive = main.groupby("Survived").size()
print(slicedby_survive)

survivors = slicedby_survive[1]
deaths = slicedby_survive[0]
percent_of_survivors = (survivors/(survivors + deaths)) * 100
percent_of_deaths = (deaths/(survivors + deaths)) * 100
print(percent_of_deaths, percent_of_survivors)

# Plot bar graph for Total Number of Passengers - Deaths and Survived
objects = ('Number of Deaths', 'Number of Survivors')
bars = np.arange(len(objects))
slicedby_survive.plot(kind = "bar", figsize = (8,6), color = 'b')
plt.xticks(bars, objects, rotation = 'horizontal')
plt.ylabel("Number of Passengers")
plt.xlabel("Total Number of Passengers")
plt.title("Total Number of Passengers - Deaths and Survived on Titanic")
plt.show()

#The following useful function is used to create counts and percentages in the pie
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}% ({v:d})'.format(p=pct, v=val)
    return my_autopct

# Plot pie chart for Total Number of Passengers Deaths and Survived
colors = ['gold','yellowgreen','lightskyblue','lightcoral']
plt.pie(slicedby_survive, shadow = True, colors = colors, labels = ['Died','Survived'], \
        autopct = make_autopct(slicedby_survive))
plt.title('Total Number of Passengers - Deaths and Survived on Titanic')    
plt.show()

#more complicated grouping
# we will group by sex and survival
slicedby_survive_sex = main.groupby(["Sex", "Survived"]).size()
plt.pie(slicedby_survive_sex, shadow = True, colors = colors, \
        labels = ['Number of Female - Deaths','Number of Female - Survivors' \
                  ,'Number of Male - Deaths','Number of Male - Survivors'], 
        autopct = make_autopct(slicedby_survive_sex))
plt.title('Total Number of Passengers - Deaths and Survived on Titanic')    
plt.show()

#it is said that women and children tended to survive
#it is also said that wealthy people survived more. Is this true? Let's try to find out
class_survived = main.groupby(["Pclass", "Survived"])
objects = ('Pclass 1', 'Pclass 2', 'Pclass 3')
bars = np.arange(len(objects))
class_survived.size().unstack().plot(kind = "bar", figsize = (8,6))
plt.xticks(bars, objects, rotation = 'horizontal')
plt.ylabel("Number of Passengers")
plt.xlabel("Total Number of Passengers in Pclass")
plt.title("Number of Passengers Traveling in Class 1, 2, 3- Deaths and Survivors")
plt.legend(['Died','Survived'])
plt.show()

##################################
# Removes the NULL values for Age
##################################
#dropping missing values - this should only be used to drop rows
#main.dropna()
#replace missing values with 0 - do with caution
#main.fillna(0)
# This calculates the same medians as before
print(main.groupby('Sex')['Age'].median())
medians = main.groupby('Sex')['Age'].median()

# Now change the index to sex, so that you can "join" based on the sex attribute
main = main.set_index(['Sex'])
# Fill in the missing value with the appropriate male/female median value
main['Age'] = main['Age'].fillna(medians)
#optionally reset the index to what it was before continuing onwards
main = main.reset_index()


validMedians = validation.groupby('Sex')['Age'].median()
validation = validation.set_index(['Sex'])
validation['Age'] = validation['Age'].fillna(validMedians)
validation = validation.reset_index()

faremedian = validation.groupby('Pclass')['Fare'].median()
validation=validation.set_index(['Pclass'])
validation['Fare']=validation['Fare'].fillna(faremedian)
validation = validation.reset_index()



##Discretize Categorical Variables for Classification

cleanup={"Sex": {'male': 0, 'female':1}, "Embarked": {'Q': 0,'S': 1,'C': 2}}

main=main.replace(cleanup)
validation=validation.replace(cleanup)

portmedian=main["Embarked"].median()

main["Embarked"]=main["Embarked"].fillna(portmedian)


#print(validation[validation["Fare"].isnull()])


main=main[['Survived','Pclass','Sex','Name','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','cabin_count','cabin_letter','title']]


##################################


X = main.iloc[:,[1,2,4,5,6,8,10,11]]
RFdata=validation.iloc[:,[0,1,3,4,5,7,9,10]]


Y = main["Survived"]
random.seed(1)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2)
print(len(X_train), len(X_test))
print(len(Y_train), len(Y_test))

#Time for logistic regression!

logisticReg=LogisticRegression(solver='lbfgs', max_iter=1000)
logisticReg.fit(X_train, Y_train)
logisticReg.score(X_train, Y_train)

# predict log reg on the test data
random.seed(1)
predictions_logReg = logisticReg.predict(X_test)

# Print accuracy score for log reg on test data
print("Accuracy Score for Logistic Regression is: ")
print(accuracy_score(Y_test, predictions_logReg))
print()

# Print classification Report of Prediction
print("Classification Report: ")
print(classification_report(Y_test, predictions_logReg))

# Print confusion matrix for predictions made
confLR = confusion_matrix(Y_test, predictions_logReg)
print(confLR)

disp = ConfusionMatrixDisplay(confusion_matrix=confLR, display_labels=logisticReg.classes_)
disp.plot()
plt.title("Confusion Matrix for Logistic Regression")
plt.show()

#support vector machines (SVM) applied to the train data
#kernel = linear
svmclf = svm.SVC(kernel='linear')
svmclf.fit(X_train, Y_train)
svmclf.score(X_train, Y_train)

 
#predict SVM algorithm on test data
random.seed(1)
predictions_svm = svmclf.predict(X_test)

# Print accuracy score for SVM Algorithm on test data
print("Accuracy Score for SVM  is: ")
print(accuracy_score(Y_test, predictions_svm))
print()

# Print classification Report of Prediction
print("Classification Report for SVM: ")
print(classification_report(Y_test, predictions_svm))

# Print confusion matrix for predictions made
confSVM = confusion_matrix(Y_test, predictions_svm)
print("Confusion Matrix for SVM: ")
print(confSVM)
print()

SVMdisp = ConfusionMatrixDisplay(confusion_matrix=confSVM, display_labels=svmclf.classes_)
SVMdisp.plot()
plt.title("Confusion Matrix for SVM")
plt.show()


#Random Forest

random.seed(1)

rf = RandomForestClassifier(n_estimators = 100, n_jobs=2)
rf.fit(X_train, Y_train)
rf.score(X_train, Y_train)

# predict Random Forest Algorithm on the test data
predictions_rf = rf.predict(X_test)

# Print accuracy score for Random Forest Algorithm on test data
print("Accuracy Score for RandomForestClassifier  is: ")
print(accuracy_score(Y_test, predictions_rf))
print()

# Print classification Report of Prediction
print("Classification Report for RandomForestClassifier: ")
print(classification_report(Y_test, predictions_rf))

# Print confusion matrix for predictions made
confRF = confusion_matrix(Y_test, predictions_rf, labels=rf.classes_)
print("Confusion Matrix for RandomForestClassifier: ")
print(confRF)
print()
dispRF = ConfusionMatrixDisplay(confusion_matrix=confRF, display_labels=rf.classes_)
dispRF.plot()
plt.title("Confusion Matrix for Random Forest Classifier")
plt.show()

RFvalidation = rf.predict(RFdata)
output = pd.DataFrame({'PassengerId': range(892,1310), 'Survived': RFvalidation})
output.to_csv('submission.csv')