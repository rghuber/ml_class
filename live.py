#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

df=pd.read_csv("drug_descriptors.txt")

#print(df)
#print(df.describe())

#print(df.columns) # all the columns in the data we read in

feature_names=df.columns[5:] # get all columns starting from col5
#print(feature_names)

X=df.loc[:,feature_names] # get a subtable of only features (not things we want to predict)

df[feature_names] = RobustScaler().fit_transform(df.loc[:,feature_names])

train,test = train_test_split(df)

# y is what we want to predict, X is the features
X_train=train.loc[:,feature_names]
X_test=test.loc[:,feature_names]

y_train_cns=train.loc[:,' cns']
y_test_cns=test.loc[:,' cns']

y_train_logp=train.loc[:,' xlogp']
y_test_logp=test.loc[:,' xlogp']

#cns_SVC=SVC(C=100.0,kernel='poly',degree=3) # what model do i want? (type, and hyperparameters)
#cns_SVC.fit(X_train,y_train_cns) # train model
#y_pred_cns=cns_SVC.predict(X_test) # use model to predict
#print(y_test_cns)
#print(y_pred_cns)
#
#print(confusion_matrix(y_test_cns,y_pred_cns))

#cns_DT=DecisionTreeClassifier(min_samples_split=5)
#cns_DT.fit(X_train,y_train_cns)
#y_pred_cns=cns_DT.predict(X_test)
#print(y_test_cns.to_list())
#print(list(y_pred_cns))
#print(confusion_matrix(y_test_cns,y_pred_cns))

#cns_RF=RandomForestClassifier()
#cns_RF.fit(X_train,y_train_cns)
#y_pred_cns=cns_RF.predict(X_test)
#print(y_test_cns.to_list())
#print(list(y_pred_cns))
#print(confusion_matrix(y_test_cns,y_pred_cns))

logp_ridge=Ridge()
logp_ridge.fit(X_train,y_train_logp)
y_pred_logp=logp_ridge.predict(X_test)
print(np.corrcoef(y_pred_logp,y_test_logp))

