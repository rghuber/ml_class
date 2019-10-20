#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import robust_scale,scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, ElasticNet, Ridge
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def print_accuracy_class(y_test,pred):
    tp=0.0
    fp=0.0
    tn=0.0
    fn=0.0
    total=len(y_test)
    for i in range(len(y_test)):
        if y_test[i] == 0 and pred[i] == 0:
            tn+=1.0
        if y_test[i] == 1 and pred[i] == 1:
            tp+=1.0
        if y_test[i] == 0 and pred[i] == 1:
            fp+=1.0
        if y_test[i] == 1 and pred[i] == 0:
            fn+=1.0

    tpr  = 100.0 * ( tp / ( fn + tp ))
    fpr  = 100.0 * ( fp / ( tn + fp ))
    tnr  = 100.0 * ( tn / ( tn + fp ))
    fnr  = 100.0 * ( fn / ( fn + tp ))
    acc  = 100.0 * (tpr*tnr*1.0e-4)**0.5
    print "TPR %8.3f  FPR %8.3f"%(tpr,fpr)
    print "FNR %8.3f  TNR %8.3f"%(fnr,tnr)
    try:
        prec = 100.0 * ( tp / ( fp + tp ))
        print "ACC %8.3f PREC %8.3f"%(acc,prec)
    except:
        pass

def print_accuracy_reg(y_test,pred):
    mue=0.0
    msqe=0.0
    for i in range(len(y_test)):
        mue+=abs(y_test[i]-pred[i])
        msqe+=(y_test[i]-pred[i])**2.0
    mue=mue/float(len(pred))
    msqe=(msqe/float(len(pred)))**0.5
    r=np.corrcoef(y_test,pred)[0][1]
    print "MUE: %8.3f  RMSE: %8.3f  Pearson R: %8.3f"%(mue,msqe,r)

def plot_accuracy_reg(y_test,pred,name):
    # plot square plot
    min_tot=min([min(y_test),min(pred)])
    max_tot=max([max(y_test),max(pred)])
    lim_lower=1.1*min_tot
    lim_upper=1.1*max_tot

    plt.figure(1,figsize=(20,20))
    plt.scatter(y_test,pred)
    plt.xlabel("Y Test")
    plt.ylabel("Y Predicted")
    plt.xlim(lim_lower,lim_upper)
    plt.ylim(lim_lower,lim_upper)
    plt.plot([lim_lower,lim_upper],[lim_lower,lim_upper],ls="--")
    plt.savefig("%s.svg"%(name),format='svg',dpi=300)
    plt.clf()

# READ DATA SET
df=pd.read_csv("drug_descriptors.txt")

names=[i for i in df.columns[5:42]]
#print names

df[names]=robust_scale(df[names])
#print df.head(5)
X=df.loc[:,names]

# ANALYZE DESCRIPTOR CORRELATIONS
corr_X=np.corrcoef(X)
of=open("corr.csv","w")
of.write(", ")
for i in range(len(names)):
    of.write("%20s, "%(names[i]))
of.write("\n")
for i in range(len(names)):
    of.write("%20s, "%(names[i]))
    for j in range(len(names)):
        of.write("%8.5f, "%(corr_X[i][j]))
    of.write("\n")
of.close()

# SPLIT TO TEST/TRAINING
train,test=train_test_split(df, test_size=0.2, random_state=2)

X_training=train.loc[:,names]
X_test=test.loc[:,names]

# GET CNS VALUES
y_df=train.loc[:,[' cns']]
cns_training=[i[0] for i in  y_df.to_numpy()]
y_df=test.loc[:,[' cns']]
cns_test=[i[0] for i in  y_df.to_numpy()]

# GET xLogP VALUES
y_df=train.loc[:,[' xlogp']]
xlogp_training=[i[0] for i in  y_df.to_numpy()]
y_df=test.loc[:,[' xlogp']]
xlogp_test=[i[0] for i in  y_df.to_numpy()]

# GET TPSA VALUES
y_df=train.loc[:,[' tpsa']]
tpsa_training=[i[0] for i in  y_df.to_numpy()]
y_df=test.loc[:,[' tpsa']]
tpsa_test=[i[0] for i in  y_df.to_numpy()]



# CLASSIFICATION: Predict Categories - e.g. CNS
print "### CLASSIFICATION - CNS ###"

# SVM Classifier
print "Running SVM Classifier"
svcl_cns = SVC(gamma='auto')
svcl_cns.fit(X_training,cns_training)
pred=svcl_cns.predict(X_test)
print_accuracy_class(cns_test,pred)

# KNeighbors Classifier
print "Running KNeighbors Classifier"
knc_cns = KNeighborsClassifier(n_neighbors=5,algorithm='ball_tree',leaf_size=20)
knc_cns.fit(X_training, cns_training)
pred = knc_cns.predict(X_test)
print_accuracy_class(cns_test,pred)

# DescisionTree Classifier
print "Running DescisionTree Classifier"
dt_cns = DecisionTreeClassifier()
dt_cns.fit(X_training, cns_training)
pred = dt_cns.predict(X_test)
print_accuracy_class(cns_test,pred)

# Random Forest Classifier
print "Running Random Forest Classifier"
rf_cns = RandomForestClassifier(n_estimators=100)
rf_cns.fit(X_training, cns_training)
pred = rf_cns.predict(X_test)
print_accuracy_class(cns_test,pred)

# Bagging Classifier
print "Running Bagging Classifier"
bg_cns = BaggingClassifier(DecisionTreeClassifier(), max_samples = 0.5, max_features = 1.0, n_estimators = 100)
bg_cns.fit(X_training, cns_training)
pred = bg_cns.predict(X_test)
print_accuracy_class(cns_test,pred)

# AdaBoost Classifier
print "Running AdaBoost Classifier"
ada_cns = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators = 100, learning_rate = 1)
ada_cns.fit(X_training, cns_training)
pred = ada_cns.predict(X_test)
print_accuracy_class(cns_test,pred)

print ""
print ""
# REGRESSION: Predict continuous quantities - e.g. xLogP
print "### REGRESSION - xLogP ###"

#print "Running Lasso Regression for xLogP"
#lasso_xlogp = Lasso(max_iter=1e6)
#lasso_xlogp.fit(X_training, xlogp_training)
#pred = lasso_xlogp.predict(X_test)
#print_accuracy_reg(xlogp_test,pred)
#plot_accuracy_reg(xlogp_test,pred,"Lasso_xLogP")

#print "Running LassoCV Regression for xLogP"
#lassocv_xlogp = LassoCV(max_iter=1e6,cv=10)
#lassocv_xlogp.fit(X_training, xlogp_training)
#pred = lassocv_xlogp.predict(X_test)
#print_accuracy_reg(xlogp_test,pred)
#plot_accuracy_reg(xlogp_test,pred,"LassoCV_xLogP")

print "Running ElasticNet Regression for xLogP"
elnet_xlogp = ElasticNet(max_iter=1e5)
elnet_xlogp.fit(X_training, xlogp_training)
pred = elnet_xlogp.predict(X_test)
print_accuracy_reg(xlogp_test,pred)
plot_accuracy_reg(xlogp_test,pred,"Elastic_xLogP")

print "Running SVR Regression for xLogP"
SVR_xlogp = SVR(max_iter=1e5,gamma='auto')
SVR_xlogp.fit(X_training, xlogp_training)
pred = SVR_xlogp.predict(X_test)
print_accuracy_reg(xlogp_test,pred)
plot_accuracy_reg(xlogp_test,pred,"SVR_xLogP")

print "Running Ridge Regression for xLogP"
ridge_xlogp = Ridge(alpha=0.01,max_iter=1e6)
ridge_xlogp.fit(X_training, xlogp_training)
pred = ridge_xlogp.predict(X_test)
print_accuracy_reg(xlogp_test,pred)
plot_accuracy_reg(xlogp_test,pred,"Ridge_xLogP")

print ""
print ""
# REGRESSION: Predict continuous quantities - e.g. TPSA
print "### REGRESSION - xLogP ###"


#print "Running Lasso Regression for TPSA"
#lasso_tpsa = Lasso(max_iter=1e6)
#lasso_tpsa.fit(X_training, tpsa_training)
#pred = lasso_tpsa.predict(X_test)
#print_accuracy_reg(tpsa_test,pred)
#plot_accuracy_reg(tpsa_test,pred,"Lasso_TPSA")

#print "Running LassoCV Regression for TPSA"
#lassocv_tpsa = LassoCV(max_iter=1e6,cv=10)
#lassocv_tpsa.fit(X_training, tpsa_training)
#pred = lassocv_tpsa.predict(X_test)
#print_accuracy_reg(tpsa_test,pred)
#plot_accuracy_reg(tpsa_test,pred,"LassoCV_TPSA")

print "Running ElasticNet Regression for TPSA"
elnet_tpsa = ElasticNet(max_iter=1e5)
elnet_tpsa.fit(X_training, tpsa_training)
pred = elnet_tpsa.predict(X_test)
print_accuracy_reg(tpsa_test,pred)
plot_accuracy_reg(tpsa_test,pred,"Elastic_TPSA")

print "Running SVR Regression for TPSA"
SVR_tpsa = SVR(max_iter=1e5,gamma='auto')
SVR_tpsa.fit(X_training, tpsa_training)
pred = SVR_tpsa.predict(X_test)
print_accuracy_reg(tpsa_test,pred)
plot_accuracy_reg(tpsa_test,pred,"SVR_TPSA")

print "Running Ridge Regression for TPSA"
ridge_tpsa = Ridge(alpha=0.2,max_iter=1e5)
ridge_tpsa.fit(X_training, tpsa_training)
pred = ridge_tpsa.predict(X_test)
print_accuracy_reg(tpsa_test,pred)
plot_accuracy_reg(tpsa_test,pred,"Ridge_TPSA")


