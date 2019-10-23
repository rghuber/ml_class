#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import hamming_loss, confusion_matrix
from sklearn.preprocessing import robust_scale,scale,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, ElasticNet, Ridge, RidgeCV, RidgeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size':30,
    })

def print_accuracy_class(y_test,y_pred):
    tp=0.0
    fp=0.0
    tn=0.0
    fn=0.0
    for i in range(len(y_test)):
        if y_test[i] == 0 and y_pred[i] == 0:
            tn+=1.0
        if y_test[i] == 1 and y_pred[i] == 1:
            tp+=1.0
        if y_test[i] == 0 and y_pred[i] == 1:
            fp+=1.0
        if y_test[i] == 1 and y_pred[i] == 0:
            fn+=1.0

    tpr  = 100.0 * ( tp / ( fn + tp ))
    fpr  = 100.0 * ( fp / ( tn + fp ))
    tnr  = 100.0 * ( tn / ( tn + fp ))
    fnr  = 100.0 * ( fn / ( fn + tp ))
    acc  = 100.0 * (tpr*tnr*1.0e-4)**0.5
    sens = 100.0 * ( tp / ( tp + fn ))
    spec = 100.0 * ( tn / ( tn + fp ))
    print "TPR  %8.3f (%4d)   FPR  %8.3f (%4d)"%(tpr,tp,fpr,fp)
    print "FNR  %8.3f (%4d)   TNR  %8.3f (%4d)"%(fnr,fn,tnr,tn)
    print "SENS %8.3f   SPEC %8.3f"%(sens,spec)
    try:
        prec = 100.0 * ( tp / ( fp + tp ))
        print "ACC  %8.3f   PREC %8.3f"%(acc,prec)
    except:
        pass
    print ""

def print_accuracy_reg(y_test,y_pred):
    mue=0.0
    msqe=0.0
    for i in range(len(y_test)):
        mue+=abs(y_test[i]-y_pred[i])
        msqe+=(y_test[i]-y_pred[i])**2.0
    mue=mue/float(len(y_pred))
    msqe=(msqe/float(len(y_pred)))**0.5
    r=np.corrcoef(y_test,y_pred)[0][1]
    print "MUE: %8.3f  RMSE: %8.3f  Pearson R: %8.3f"%(mue,msqe,r)
    print ""

def plot_accuracy_reg(y_test,y_pred,name):
    # plot square plot
    min_tot=min([min(y_test),min(y_pred)])
    max_tot=max([max(y_test),max(y_pred)])
    span=max_tot-min_tot
    lim_lower=min_tot-0.05*span
    lim_upper=max_tot+0.05*span

    plt.figure(1,figsize=(20,20))
    plt.title(name)
    plt.scatter(y_test,y_pred)
    plt.xlabel("Y Test")
    plt.ylabel("Y Predicted")
    plt.xlim(lim_lower,lim_upper)
    plt.ylim(lim_lower,lim_upper)
    plt.plot([lim_lower,lim_upper],[lim_lower,lim_upper],ls="--")
    plt.savefig("%s.svg"%(name),format='svg',dpi=300)
    plt.clf()

# READ DATA SET
df=pd.read_csv("drug_descriptors.txt")

names=[i for i in df.columns[5:]]
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
train,test=train_test_split(df, test_size=0.2, random_state=3)

poly = PolynomialFeatures(2)

X_training=train.loc[:,names]
#X_training=poly.fit_transform(X_training)

X_test=test.loc[:,names]
#X_test=poly.fit_transform(X_test)

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
param_grid = [
        {'C' : [1.0,10.0,100.0,1000.0], 'kernel' : ['poly','rbf'], 'degree' : [2]}
        ]
svcl_cns_gs = GridSearchCV(svcl_cns, param_grid, scoring='balanced_accuracy', cv=5, refit = True, verbose = True )
svcl_cns_gs.fit(X_training,cns_training)
#print sorted(svcl_cns_gs.cv_results_)
y_pred=svcl_cns_gs.predict(X_test)
#print confusion_matrix(cns_test,y_pred)
print_accuracy_class(cns_test,y_pred)

# KNeighbors Classifier
print "Running KNeighbors Classifier"
knc_cns = KNeighborsClassifier(algorithm='ball_tree')
param_grid = [
        {'n_neighbors' : [2,5,10,20,50], 'leaf_size' : [ 10, 20, 30, 40, 50] }
        ]
knc_cns_gs = GridSearchCV(knc_cns, param_grid, scoring='balanced_accuracy', cv=5, refit = True, verbose = True )
knc_cns_gs.fit(X_training, cns_training)
print sorted(knc_cns_gs.cv_results_)
y_pred = knc_cns_gs.predict(X_test)
print_accuracy_class(cns_test,y_pred)

# DescisionTree Classifier
print "Running DescisionTree Classifier"
dt_cns = DecisionTreeClassifier()
param_grid = [
        {'max_depth' : [2,5,10,20], 'max_features' : [ 2, 5, 10, 20] }
        ]
dt_cns_gs = GridSearchCV(dt_cns, param_grid, scoring='balanced_accuracy', cv=5, refit = True, verbose = True )
dt_cns_gs.fit(X_training, cns_training)
#print sorted(dt_cns_gs.cv_results_)
y_pred = dt_cns_gs.predict(X_test)
print
print_accuracy_class(cns_test,y_pred)

# Random Forest Classifier
print "Running Random Forest Classifier"
rf_cns = RandomForestClassifier(n_estimators=10)
param_grid = [
        {'n_estimators' : [5,10,20,50,100], 'max_features' : [ 10, 20, 30] }
        ]
rf_cns_gs = GridSearchCV(rf_cns, param_grid, scoring='balanced_accuracy', cv=5, refit = True, verbose = True )
rf_cns_gs.fit(X_training, cns_training)
print rf_cns_gs.cv_results_['rank_test_score']
y_pred = rf_cns_gs.predict(X_test)
print_accuracy_class(cns_test,y_pred)

# Ridge Classifier
print "Running Ridge Classifier"
ridge_cns = RidgeClassifier()
param_grid = [
        {'alpha' : [1.0,0.1,0.01,0.001,0.0001] }
        ]
ridge_cns_gs = GridSearchCV(ridge_cns, param_grid, scoring='balanced_accuracy', cv=5, refit = True, verbose = True, return_train_score = True )
ridge_cns_gs.fit(X_training, cns_training)
print ridge_cns_gs.cv_results_['rank_test_score']
y_pred = ridge_cns_gs.predict(X_test)
print_accuracy_class(cns_test,y_pred)

# Bagging Classifier
print "Running Bagging Classifier"
bg_cns = BaggingClassifier(RandomForestClassifier(), max_samples = 0.8, max_features = 1.0, n_estimators = 10)
scores = cross_val_score(bg_cns, X_training, cns_training, cv=5)
print scores
bg_cns.fit(X_training, cns_training)
y_pred = bg_cns.predict(X_test)
print_accuracy_class(cns_test,y_pred)

# AdaBoost Classifier
print "Running AdaBoost Classifier"
ada_cns = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators = 10, learning_rate = 0.01)
scores = cross_val_score(ada_cns, X_training, cns_training, cv=5)
print scores
ada_cns.fit(X_training, cns_training)
y_pred = ada_cns.predict(X_test)
print_accuracy_class(cns_test,y_pred)

# Gradient Boosting Classifier
print "Running Gradient Boosting Classifier"
gb_cns = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 3)
scores = cross_val_score(gb_cns, X_training, cns_training, cv=5)
print scores
gb_cns.fit(X_training, cns_training)
y_pred = gb_cns.predict(X_test)
print_accuracy_class(cns_test,y_pred)

# Voting Classifier
print "Running Voting Classifier"
vote_cns = VotingClassifier(estimators=[('svc',svcl_cns),('dt',dt_cns),('ridge',ridge_cns),('rf',rf_cns)])
scores = cross_val_score(vote_cns, X_training, cns_training, cv=5)
print scores
vote_cns.fit(X_training, cns_training)
y_pred = vote_cns.predict(X_test)
print_accuracy_class(cns_test,y_pred)

print ""
print ""
# REGRESSION: Predict continuous quantities - e.g. xLogP
print "### REGRESSION - xLogP ###"

print "Running Lasso Regression for xLogP"
lasso_xlogp = Lasso(max_iter=1e6)
lasso_xlogp.fit(X_training, xlogp_training)
y_pred = lasso_xlogp.predict(X_test)
print_accuracy_reg(xlogp_test,y_pred)
plot_accuracy_reg(xlogp_test,y_pred,"Lasso_xLogP")

#print "Running LassoCV Regression for xLogP"
#lassocv_xlogp = LassoCV(max_iter=1e6,cv=10)
#lassocv_xlogp.fit(X_training, xlogp_training)
#y_pred = lassocv_xlogp.predict(X_test)
#print_accuracy_reg(xlogp_test,y_pred)
#plot_accuracy_reg(xlogp_test,y_pred,"LassoCV_xLogP")

print "Running ElasticNet Regression for xLogP"
elnet_xlogp = ElasticNet(max_iter=1e5)
elnet_xlogp.fit(X_training, xlogp_training)
y_pred = elnet_xlogp.predict(X_test)
print_accuracy_reg(xlogp_test,y_pred)
plot_accuracy_reg(xlogp_test,y_pred,"Elastic_xLogP")

print "Running SVR Regression for xLogP"
SVR_xlogp = SVR(max_iter=1e5,gamma='auto',C=100.0,kernel='rbf')
SVR_xlogp.fit(X_training, xlogp_training)
y_pred = SVR_xlogp.predict(X_test)
print_accuracy_reg(xlogp_test,y_pred)
plot_accuracy_reg(xlogp_test,y_pred,"SVR_xLogP")

#print "Running RidgeCV Regression for xLogP"
#ridgeCV_xlogp = RidgeCV(alphas=[1.0,0.1,0.01,0.001],fit_intercept=True,normalize=False,store_cv_values=True)
#ridgeCV_xlogp.fit(X_training, xlogp_training)
#print ridgeCV_xlogp.alpha_
#y_pred = ridgeCV_xlogp.predict(X_test)
#print_accuracy_reg(xlogp_test,y_pred)
#plot_accuracy_reg(xlogp_test,y_pred,"Ridge_xLogP")

print "Running Ridge Regression for xLogP"
ridge_xlogp = Ridge(alpha=0.0001,max_iter=1e6)
ridge_xlogp.fit(X_training, xlogp_training)
y_pred = ridge_xlogp.predict(X_test)
print_accuracy_reg(xlogp_test,y_pred)
plot_accuracy_reg(xlogp_test,y_pred,"Ridge_xLogP")



print ""
print ""
# REGRESSION: Predict continuous quantities - e.g. TPSA
print "### REGRESSION - TPSA ###"


print "Running Lasso Regression for TPSA"
lasso_tpsa = Lasso(max_iter=1e6)
lasso_tpsa.fit(X_training, tpsa_training)
y_pred = lasso_tpsa.predict(X_test)
y_pred_training = lasso_tpsa.predict(X_training)
plot_accuracy_reg(tpsa_training,y_pred_training,"Lasso_TPSA_Training")
print_accuracy_reg(tpsa_test,y_pred)
plot_accuracy_reg(tpsa_test,y_pred,"Lasso_TPSA")

#print "Running LassoCV Regression for TPSA"
#lassocv_tpsa = LassoCV(max_iter=1e6,cv=10)
#lassocv_tpsa.fit(X_training, tpsa_training)
#y_pred = lassocv_tpsa.predict(X_test)
#print_accuracy_reg(tpsa_test,y_pred)
#plot_accuracy_reg(tpsa_test,y_pred,"LassoCV_TPSA")

print "Running ElasticNet Regression for TPSA"
elnet_tpsa = ElasticNet(max_iter=1e5)
elnet_tpsa.fit(X_training, tpsa_training)
y_pred = elnet_tpsa.predict(X_test)
print_accuracy_reg(tpsa_test,y_pred)
plot_accuracy_reg(tpsa_test,y_pred,"Elastic_TPSA")

print "Running SVR Regression for TPSA"
SVR_tpsa = SVR(max_iter=1e5,gamma='auto',C=100.0,kernel='rbf')
SVR_tpsa.fit(X_training, tpsa_training)
y_pred = SVR_tpsa.predict(X_test)
print_accuracy_reg(tpsa_test,y_pred)
plot_accuracy_reg(tpsa_test,y_pred,"SVR_TPSA")

print "Running Ridge Regression for TPSA"
ridge_tpsa = Ridge(alpha=0.0001,max_iter=1e5)
ridge_tpsa.fit(X_training, tpsa_training)
y_pred = ridge_tpsa.predict(X_test)
print_accuracy_reg(tpsa_test,y_pred)
plot_accuracy_reg(tpsa_test,y_pred,"Ridge_TPSA")


