#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale,robust_scale
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Ridge


df = pd.read_csv("drug_descriptors.txt")
features = [i for i in df.columns[5:40]]

#print df.describe()

df[features]=scale(df[features])

#print df.describe()

X=df.loc[:,features]

#corr_X=np.corrcoef(X)
#print corr_X

train, test = train_test_split(df, test_size=0.2, random_state=3)

X_training = train.loc[:,features]
X_test = test.loc[:,features]

# get y for cns
y_df = train.loc[:,[' cns']]
y_cns_training = [i[0] for i in y_df.to_numpy()]
y_df = test.loc[:,[' cns']]
y_cns_test = [i[0] for i in y_df.to_numpy()]

# get y for xlogp
y_df = train.loc[:,[' xlogp']]
y_xlogp_training = [i[0] for i in y_df.to_numpy()]
y_df = test.loc[:,[' xlogp']]
y_xlogp_test = [i[0] for i in y_df.to_numpy()]

# get y for tpsa
y_df = train.loc[:,[' tpsa']]
y_tpsa_training = [i[0] for i in y_df.to_numpy()]
y_df = test.loc[:,[' tpsa']]
y_tpsa_test = [i[0] for i in y_df.to_numpy()]

#print y_tpsa_test
#print len(y_tpsa_test)
#print len(y_tpsa_training)

#PLAIN SVC Classifier
svc_cns = SVC()
svc_cns.fit(X_training, y_cns_training)
y_pred_cns = svc_cns.predict(X_test)
print confusion_matrix(y_cns_test, y_pred_cns)

#GRID SEARCH FOR SVC Classifier Parameters (using cross-validation)
param_grid = [
        {'kernel' : ['rbf', 'poly'], 'C' : [1.0, 5.0, 10.0] }
        ]
#svc_cns_gs = GridSearchCV( svc_cns, param_grid, scoring='balanced_accuracy', cv = 10, refit = True, verbose = False)
#svc_cns_gs.fit(X_training, y_cns_training)
#y_pred_cns = svc_cns_gs.predict(X_test)
print confusion_matrix(y_cns_test, y_pred_cns)

#BAGGING Ensemble Classifier based on 10 KNeighborsClassifiers
bagging_cns = BaggingClassifier( KNeighborsClassifier(), max_samples=0.5, max_features=0.5 )
bagging_cns.fit(X_training, y_cns_training)
y_pred_cns = bagging_cns.predict(X_test)
print confusion_matrix(y_cns_test, y_pred_cns)

#RIDGE Regression
ridge_tpsa = Ridge()
ridge_tpsa.fit(X_training, y_tpsa_training)
y_pred_tpsa = ridge_tpsa.predict(X_test)
print np.corrcoef(y_pred_tpsa, y_tpsa_test)
