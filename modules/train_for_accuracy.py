import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pyper
import ast

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, BaseCrossValidator
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb
from lightgbm import early_stopping, record_evaluation

from . import make_data_for_prediction


class StratifiedKeyBasedKFold(BaseCrossValidator):
    """train test split based on another key values fro OptunaLightGBMTuner"""

    def __init__(self, n_splits, keys, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.keys = keys
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """split test/train data"""
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        skf_split_iterator = skf.split(X, self.keys)

        for train_indices, test_indices in skf_split_iterator:
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def calc_acc(y_test_preds, test_data, metrics, plot=True):
    if isinstance(y_test_preds, list):
        y_test_pred = pd.DataFrame(y_test_preds).mean().values
    else:
        y_test_pred = y_test_preds
    y_test = test_data[1].values
    if plot:
        yyplot(y_test, y_test_pred)
    if metrics == "r2":
        acc = np.corrcoef(y_test, y_test_pred)[0][1]
    elif metrics == "rmse":
        acc = np.sqrt(mean_squared_error(y_test, y_test_pred))
    elif metrics == "mae":
        acc = mean_absolute_error(y_test, y_test_pred)
    return acc

def calc_acc_each_RILs(y_test_preds, test_data, metrics, plot=True):
    if isinstance(y_test_preds, list):
        y_test_pred = pd.DataFrame(y_test_preds).mean().values
    else:
        y_test_pred = y_test_preds
    y_test = test_data[1].values
    RIL_ids = test_data[1].index.str[:3]
    acc_all = []
    for RIL_id in RIL_ids.unique():
        y_test_pred_RIL = y_test_pred[RIL_ids == RIL_id]
        y_test_RIL = y_test[RIL_ids == RIL_id]
        if plot:
            yyplot(y_test_RIL, y_test_pred_RIL)
        if metrics == "r2":
            acc = np.corrcoef(y_test_RIL, y_test_pred_RIL)[0][1]
        elif metrics == "rmse":
            acc = np.sqrt(mean_squared_error(y_test_RIL, y_test_pred_RIL))
        elif metrics == "mae":
            acc = mean_absolute_error(y_test_RIL, y_test_pred_RIL)
        acc_all.append(acc)
    return [acc_all, RIL_ids.unique().values]

def train_GBLUP(test_data, train_data, genotype):
    X_index = pd.concat([test_data[0], train_data[0]]).sort_index().index.values
    X_column = test_data[0].columns.values
    y_train = train_data[1].sort_index()
    y_train = pd.DataFrame(y_train).reset_index()
    y_train.columns = ["line", "y"]
    
    r = pyper.R(use_numpy='True', use_pandas='True')
    r.assign("genotype", genotype)
    r.assign("X_index", X_index)
    r.assign("X_column", X_column)
    r.assign("y_train", y_train)
    code = """
    library(rrBLUP)
    X <- read.csv(genotype)
    X <- X[X_column, X_index]
    M <- t(X-1)
    A <- A.mat(M)
    ans <- kin.blup(data=y_train,geno='line',pheno='y',K=A)
    pred <- ans$pred
    VarG <- ans$Vg
    VarE <- ans$Ve
    """
    r(code)
    all_pred = r.get("pred")
    all_pred = pd.DataFrame({"y":all_pred})
    all_pred.index = X_index
    
    # calc estimated heritaility
    VarG = r.get("VarG")
    VarE = r.get("VarE")
    h2 = VarG / (VarG + VarE)
    
    # predicted values of test phenotype
    test_pred = all_pred.loc[test_data[0].index.values, :]
    
    # calc prediction accuracy
    r2 = calc_acc(test_pred.y.values, test_data, "r2", plot=False)
    acc_RIL = calc_acc_each_RILs(test_pred.y.values, test_data, "r2", plot=False)
    
    return test_pred.y.values, h2, r2, acc_RIL

def GBLUP_coef(train_data, genotype):
    coefs = []
    original_skf = StratifiedKeyBasedKFold(n_splits=5, keys=train_data[1].index.str[:3].values, shuffle=True, random_state=1024)
    for train, test in original_skf.split(range(train_data[1].shape[0])):
        X_train = train_data[0].iloc[train, :]
        y_train = train_data[1].iloc[train]
        X_index = X_train.sort_index().index.values
        X_column = X_train.columns.values
        y_train = y_train.sort_index()
        r = pyper.R(use_numpy='True', use_pandas='True')
        r.assign("genotype", genotype)
        r.assign("X_index", X_index)
        r.assign("X_column", X_column)
        r.assign("y", y_train.values)
        code = """
        library(rrBLUP)
        X <- read.csv(genotype)
        X <- X[X_column, X_index]
        M <- t(X-1)
        M[is.na(M)] <- 0
        ans <- mixed.solve(y,Z=M)
        coef <- ans$u
        """
        r(code)
        coef = r.get("coef")
        coefs.append(coef)
    return coefs

def train_lightgbm(test_data, train_data, CV, params):
    y_test_preds = []
    feature_importances = []
    # separate validation data
    original_skf = StratifiedKeyBasedKFold(n_splits=5, keys=train_data[1].index.str[:3].values, shuffle=True, random_state=1024)
    for train, valid in original_skf.split(range(train_data[0].shape[0])):
        X_train, X_valid = train_data[0].iloc[train, :], train_data[0].iloc[valid, :]
        y_train, y_valid = train_data[1].iloc[train], train_data[1].iloc[valid]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

        # train
        evals_result = {}
        gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=[lgb_eval, lgb_train],
                    valid_names=['eval', 'train'],
                    callbacks=[early_stopping(50, verbose=False), record_evaluation(evals_result)]
                    )
        feature_importances.append(gbm.feature_importance())
        y_test_pred = gbm.predict(test_data[0], num_iteration=gbm.best_iteration)
        y_test_preds.append(y_test_pred)
    
    # calc prediction accuracy
    r2 = calc_acc(y_test_preds, test_data, "r2", plot=False)
    acc_RIL = calc_acc_each_RILs(y_test_preds, test_data, "r2", plot=False)
    
    return y_test_preds, r2, acc_RIL, feature_importances

def train_randomforest(test_data, train_data, CV, params):
    
    X_train = train_data[0].fillna(1)
    y_train = train_data[1]
    X_test = test_data[0].fillna(1)
    # train model
    clf = RandomForestRegressor(**params)
    clf.fit(X_train, y_train)
    # get feature importances
    feature_importances = []
    original_skf = StratifiedKeyBasedKFold(n_splits=5, keys=y_train.index.str[:3].values, shuffle=True, random_state=1024)
    for train, test in original_skf.split(range(y_train.shape[0])):
        clf = RandomForestRegressor(**params)
        clf.fit(X_train.iloc[train, :], y_train.iloc[train])
        feature_importances.append(clf.feature_importances_)
    # predicted values of test phenotype
    test_pred = clf.predict(X_test)
    # calc prediction accuracy
    r2 = calc_acc(test_pred, test_data, "r2", plot=False)
    acc_RIL = calc_acc_each_RILs(test_pred, test_data, "r2", plot=False)
    
    return test_pred, r2, acc_RIL, feature_importances

def train_Lasso(test_data, train_data, params):    
    
    alpha = params["alpha"]        
    X_train = train_data[0].fillna(1)
    y_train = train_data[1]
    X_test = test_data[0].fillna(1)
    # train model
    clf = Lasso(alpha=alpha, max_iter=100000)
    clf.fit(X_train, y_train)
    # get coefs
    coefs = []
    original_skf = StratifiedKeyBasedKFold(n_splits=5, keys=y_train.index.str[:3].values, shuffle=True, random_state=1024)
    for train, test in original_skf.split(range(y_train.shape[0])):
        clf = Lasso(alpha=alpha, max_iter=100000)
        clf.fit(X_train.iloc[train, :], y_train.iloc[train])
        coefs.append(clf.coef_)
    # predicted values of test phenotype
    test_pred = clf.predict(X_test)
    # calc prediction accuracy
    r2 = calc_acc(test_pred, test_data, "r2", plot=False)
    acc_RIL = calc_acc_each_RILs(test_pred, test_data, "r2", plot=False)
    
    return test_pred, r2, acc_RIL, coefs

def train_Enet(test_data, train_data, params):
    
    alpha = params["alpha"]
    l1_ratio = params["l1_ratio"]      
    X_train = train_data[0].fillna(1)
    y_train = train_data[1]
    X_test = test_data[0].fillna(1)
    # train model
    clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000)
    clf.fit(X_train, y_train)
    # get coefs
    coefs = []
    original_skf = StratifiedKeyBasedKFold(n_splits=5, keys=y_train.index.str[:3].values, shuffle=True, random_state=1024)
    for train, test in original_skf.split(range(y_train.shape[0])):
        clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000)
        clf.fit(X_train.iloc[train, :], y_train.iloc[train])
        coefs.append(clf.coef_)
        
    # predicted values of test phenotype
    test_pred = clf.predict(X_test)
    # calc prediction accuracy
    r2 = calc_acc(test_pred, test_data, "r2", plot=False)
    acc_RIL = calc_acc_each_RILs(test_pred, test_data, "r2", plot=False)
    
    return test_pred, r2, acc_RIL, coefs
