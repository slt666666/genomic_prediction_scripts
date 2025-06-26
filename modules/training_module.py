import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pyper
import ast

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb
from lightgbm import early_stopping, record_evaluation

from . import make_data_for_prediction


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
    X_train = train_data[0]
    X_index = X_train.sort_index().index.values
    X_column = X_train.columns.values
    y_train = train_data[1].sort_index()
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
    return coef

def train_lightgbm(test_data, train_data, CV, params):
    y_test_preds = []
    feature_importances = []
    # separate validation data
    skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=0)
    for train, valid in skf.split(range(train_data[0].shape[0]),  train_data[0].index.str[:3].values):
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
    
    X_train = train_data[0].fillna(0)
    y_train = train_data[1]
    X_test = test_data[0].fillna(0)
    # train model
    clf = RandomForestRegressor(**params)
    clf.fit(X_train, y_train)
    # get coefs
    feature_importances = clf.feature_importances_
    # predicted values of test phenotype
    test_pred = clf.predict(X_test)
    # calc prediction accuracy
    r2 = calc_acc(test_pred, test_data, "r2", plot=False)
    acc_RIL = calc_acc_each_RILs(test_pred, test_data, "r2", plot=False)
    
    return test_pred, r2, acc_RIL, feature_importances

def train_Lasso(test_data, train_data, params):    
    
    alpha = params["alpha"]        
    X_train = train_data[0].fillna(0)
    y_train = train_data[1]
    X_test = test_data[0].fillna(0)
    # train model
    clf = Lasso(alpha=alpha, max_iter=100000)
    clf.fit(X_train, y_train)
    # get coefs
    coefs = clf.coef_
    # predicted values of test phenotype
    test_pred = clf.predict(X_test)
    # calc prediction accuracy
    r2 = calc_acc(test_pred, test_data, "r2", plot=False)
    acc_RIL = calc_acc_each_RILs(test_pred, test_data, "r2", plot=False)
    
    return test_pred, r2, acc_RIL, coefs

def train_Enet(test_data, train_data, params):
    
    alpha = params["alpha"]
    l1_ratio = params["l1_ratio"]      
    X_train = train_data[0].fillna(0)
    y_train = train_data[1]
    X_test = test_data[0].fillna(0)
    # train model
    clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000)
    clf.fit(X_train, y_train)
    # get coefs
    coefs = clf.coef_
    # predicted values of test phenotype
    test_pred = clf.predict(X_test)
    # calc prediction accuracy
    r2 = calc_acc(test_pred, test_data, "r2", plot=False)
    acc_RIL = calc_acc_each_RILs(test_pred, test_data, "r2", plot=False)
    
    return test_pred, r2, acc_RIL, coefs

# def train_AEnet(test_data, train_data, params):
    
#     alpha = params["alpha"]
#     l1_ratio = params["l1_ratio"]      
#     X_train = train_data[0].fillna(1)
#     y_train = train_data[1]
#     X_test = test_data[0].fillna(1)
#     # train model
#     clf = AdaptiveElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000)
#     clf.fit(X_train, y_train)
#     # get coefs
#     coefs = clf.coef_
#     # predicted values of test phenotype
#     test_pred = clf.predict(X_test)
#     # calc prediction accuracy
#     r2 = calc_acc(test_pred, test_data, "r2", plot=False)
#     acc_RIL = calc_acc_each_RILs(test_pred, test_data, "r2", plot=False)
    
#     return test_pred, r2, acc_RIL, coefs

# def train_BayesA(test_data, train_data, genotype):
#     X_index = pd.concat([test_data[0], train_data[0]]).sort_index().index.values
#     X_column = test_data[0].columns.values
#     y = pd.concat([test_data[1], train_data[1]]).sort_index()
#     y.loc[test_data[1].index.values] = np.nan

#     r = pyper.R(use_numpy='True', use_pandas='True')
#     r.assign("genotype", genotype)
#     r.assign("X_index", X_index)
#     r.assign("X_column", X_column)
#     r.assign("y", y.values)
#     code = """
#     library(BGLR)
#     X <- read.csv(genotype)
#     X <- X[X_column, X_index]
#     X[is.na(X)] <- 1
#     X <- t(X)
#     nIter=10000
#     burnIn=2000
#     fmBRR=BGLR(y=y,ETA=list( list(X=X,model='BayesA')), 
#                nIter=nIter,burnIn=burnIn)
#     yHat <- fmBRR$yHat
#     bHat <- fmBRR$ETA[[1]]$b
#     """
#     r(code)
#     all_pred = r.get("yHat")
#     coefs = r.get("bHat")
    
#     all_pred = pd.DataFrame({"y":all_pred})
#     all_pred.index = X_index
    
#     # predicted values of test phenotype
#     test_pred = all_pred.loc[test_data[0].index.values, :]
    
#     # calc prediction accuracy
#     r2 = calc_acc(test_pred.y.values, test_data, "r2", plot=False)
#     acc_RIL = calc_acc_each_RILs(test_pred.y.values, test_data, "r2", plot=False)
    
#     return test_pred.y.values, r2, acc_RIL, coefs
    
# def train_BayesB(test_data, train_data, genotype):
#     X_index = pd.concat([test_data[0], train_data[0]]).sort_index().index.values
#     X_column = test_data[0].columns.values
#     y = pd.concat([test_data[1], train_data[1]]).sort_index()
#     y.loc[test_data[1].index.values] = np.nan

#     r = pyper.R(use_numpy='True', use_pandas='True')
#     r.assign("genotype", genotype)
#     r.assign("X_index", X_index)
#     r.assign("X_column", X_column)
#     r.assign("y", y.values)
#     code = """
#     library(BGLR)
#     X <- read.csv(genotype)
#     X <- X[X_column, X_index]
#     X[is.na(X)] <- 1
#     X <- t(X)
#     nIter=10000
#     burnIn=2000
#     fmBRR=BGLR(y=y,ETA=list( list(X=X,model='BayesB')), 
#                nIter=nIter,burnIn=burnIn)
#     yHat <- fmBRR$yHat
#     bHat <- fmBRR$ETA[[1]]$b
#     """
#     r(code)
#     all_pred = r.get("yHat")
#     coefs = r.get("bHat")
    
#     all_pred = pd.DataFrame({"y":all_pred})
#     all_pred.index = X_index
    
#     # predicted values of test phenotype
#     test_pred = all_pred.loc[test_data[0].index.values, :]
    
#     # calc prediction accuracy
#     r2 = calc_acc(test_pred.y.values, test_data, "r2", plot=False)
#     acc_RIL = calc_acc_each_RILs(test_pred.y.values, test_data, "r2", plot=False)
    
#     return test_pred.y.values, r2, acc_RIL, coefs

# def train_BayesC(test_data, train_data, genotype):
#     X_index = pd.concat([test_data[0], train_data[0]]).sort_index().index.values
#     X_column = test_data[0].columns.values
#     y = pd.concat([test_data[1], train_data[1]]).sort_index()
#     y.loc[test_data[1].index.values] = np.nan

#     r = pyper.R(use_numpy='True', use_pandas='True')
#     r.assign("genotype", genotype)
#     r.assign("X_index", X_index)
#     r.assign("X_column", X_column)
#     r.assign("y", y.values)
#     code = """
#     library(BGLR)
#     X <- read.csv(genotype)
#     X <- X[X_column, X_index]
#     X[is.na(X)] <- 1
#     X <- t(X)
#     nIter=10000
#     burnIn=2000
#     fmBRR=BGLR(y=y,ETA=list( list(X=X,model='BayesC')), 
#                nIter=nIter,burnIn=burnIn)
#     yHat <- fmBRR$yHat
#     bHat <- fmBRR$ETA[[1]]$b
#     """
#     r(code)
#     all_pred = r.get("yHat")
#     coefs = r.get("bHat")
    
#     all_pred = pd.DataFrame({"y":all_pred})
#     all_pred.index = X_index
    
#     # predicted values of test phenotype
#     test_pred = all_pred.loc[test_data[0].index.values, :]
    
#     # calc prediction accuracy
#     r2 = calc_acc(test_pred.y.values, test_data, "r2", plot=False)
#     acc_RIL = calc_acc_each_RILs(test_pred.y.values, test_data, "r2", plot=False)
    
#     return test_pred.y.values, r2, acc_RIL, coefs

# def train_BLasso(test_data, train_data, genotype):
#     X_index = pd.concat([test_data[0], train_data[0]]).sort_index().index.values
#     X_column = test_data[0].columns.values
#     y = pd.concat([test_data[1], train_data[1]]).sort_index()
#     y.loc[test_data[1].index.values] = np.nan

#     r = pyper.R(use_numpy='True', use_pandas='True')
#     r.assign("genotype", genotype)
#     r.assign("X_index", X_index)
#     r.assign("X_column", X_column)
#     r.assign("y", y.values)
#     code = """
#     library(BGLR)
#     X <- read.csv(genotype)
#     X <- X[X_column, X_index]
#     X[is.na(X)] <- 1
#     X <- t(X)
#     nIter=10000
#     burnIn=2000
#     fmBRR=BGLR(y=y,ETA=list( list(X=X,model='BL')), 
#                nIter=nIter,burnIn=burnIn)
#     yHat <- fmBRR$yHat
#     bHat <- fmBRR$ETA[[1]]$b
#     """
#     r(code)
#     all_pred = r.get("yHat")
#     coefs = r.get("bHat")
    
#     all_pred = pd.DataFrame({"y":all_pred})
#     all_pred.index = X_index
    
#     # predicted values of test phenotype
#     test_pred = all_pred.loc[test_data[0].index.values, :]
    
#     # calc prediction accuracy
#     r2 = calc_acc(test_pred.y.values, test_data, "r2", plot=False)
#     acc_RIL = calc_acc_each_RILs(test_pred.y.values, test_data, "r2", plot=False)
    
#     return test_pred.y.values, r2, acc_RIL, coefs

# def train_BRR(test_data, train_data, genotype):
#     X_index = pd.concat([test_data[0], train_data[0]]).sort_index().index.values
#     X_column = test_data[0].columns.values
#     y = pd.concat([test_data[1], train_data[1]]).sort_index()
#     y.loc[test_data[1].index.values] = np.nan

#     r = pyper.R(use_numpy='True', use_pandas='True')
#     r.assign("genotype", genotype)
#     r.assign("X_index", X_index)
#     r.assign("X_column", X_column)
#     r.assign("y", y.values)
#     code = """
#     library(BGLR)
#     X <- read.csv(genotype)
#     X <- X[X_column, X_index]
#     X[is.na(X)] <- 1
#     X <- t(X)
#     nIter=10000
#     burnIn=2000
#     fmBRR=BGLR(y=y,ETA=list( list(X=X,model='BRR')), 
#                nIter=nIter,burnIn=burnIn)
#     yHat <- fmBRR$yHat
#     bHat <- fmBRR$ETA[[1]]$b
#     """
#     r(code)
#     all_pred = r.get("yHat")
#     coefs = r.get("bHat")
    
#     all_pred = pd.DataFrame({"y":all_pred})
#     all_pred.index = X_index
    
#     # predicted values of test phenotype
#     test_pred = all_pred.loc[test_data[0].index.values, :]
    
#     # calc prediction accuracy
#     r2 = calc_acc(test_pred.y.values, test_data, "r2", plot=False)
#     acc_RIL = calc_acc_each_RILs(test_pred.y.values, test_data, "r2", plot=False)
    
#     return test_pred.y.values, r2, acc_RIL, coefs

# def train_RKHS(test_data, train_data, genotype):
#     X_index = pd.concat([test_data[0], train_data[0]]).sort_index().index.values
#     X_column = test_data[0].columns.values
#     y = pd.concat([test_data[1], train_data[1]]).sort_index()
#     y.loc[test_data[1].index.values] = np.nan

#     r = pyper.R(use_numpy='True', use_pandas='True')
#     r.assign("genotype", genotype)
#     r.assign("X_index", X_index)
#     r.assign("X_column", X_column)
#     r.assign("y", y.values)
#     code = """
#     library(BGLR)
#     X <- read.csv(genotype)
#     X <- X[X_column, X_index]
#     X[is.na(X)] <- 1
#     X <- t(X)
#     nIter=10000
#     burnIn=2000
#     fmBRR=BGLR(y=y,ETA=list( list(X=X,model='BayesA')), 
#                nIter=nIter,burnIn=burnIn)
#     yHat <- fmBRR$yHat
#     bHat <- fmBRR$ETA[[1]]$b
#     """
#     r(code)
#     all_pred = r.get("yHat")
#     coefs = r.get("bHat")
    
#     all_pred = pd.DataFrame({"y":all_pred})
#     all_pred.index = X_index
    
#     # predicted values of test phenotype
#     test_pred = all_pred.loc[test_data[0].index.values, :]
    
#     # calc prediction accuracy
#     r2 = calc_acc(test_pred.y.values, test_data, "r2", plot=False)
#     acc_RIL = calc_acc_each_RILs(test_pred.y.values, test_data, "r2", plot=False)
    
#     return test_pred.y.values, r2, acc_RIL, coefs

# def train_all_models(test_data, train_data, genotype, family, CV, trait_name, param_path):
    
#     #params
#     LGBM_params = pd.read_csv("{}/LGBM_{}_params.csv".format(param_path, family), index_col=0)
#     LGBM_params.columns = ["trait", "params"]
#     LGBM_params = ast.literal_eval(LGBM_params.loc[LGBM_params["trait"] == trait_name, "params"].values[0])
#     RF_params = pd.read_csv("{}/RF_{}_params.csv".format(param_path, family), index_col=0)
#     RF_params.columns = ["trait", "params"]
#     RF_params = ast.literal_eval(RF_params.loc[RF_params["trait"] == trait_name, "params"].values[0])
#     Lasso_params = pd.read_csv("{}/Lasso_{}_params.csv".format(param_path, family), index_col=0)
#     Lasso_params.columns = ["trait", "params"]
#     Lasso_params = ast.literal_eval(Lasso_params.loc[Lasso_params["trait"] == trait_name, "params"].values[0])
#     Enet_params = pd.read_csv("{}/Enet_{}_params.csv".format(param_path, family), index_col=0)
#     Enet_params.columns = ["trait", "params"]
#     Enet_params = ast.literal_eval(Enet_params.loc[Enet_params["trait"] == trait_name, "params"].values[0])
    
#     # train all models
#     summary = []
#     print("GBLUP start")
#     test_pred, h2, r2, acc_RIL = train_GBLUP(test_data, train_data, genotype)
#     scores = GWAS(train_data, genotype)
#     summary.append(["GBLUP", test_pred, r2, acc_RIL, scores])
    
#     print("LightGBM start")
#     test_preds, r2, acc_RIL, feature_importances = train_lightgbm(test_data, train_data, CV, LGBM_params)
#     summary.append(["LGBM", test_preds, r2, acc_RIL, feature_importances])
    
#     print("RandomForest start")
#     test_pred, r2, acc_RIL, feature_importance = train_randomforest(test_data, train_data, CV, RF_params)
#     summary.append(["RF", test_pred, r2, acc_RIL, feature_importance])
    
#     print("Lasso start")
#     test_pred, r2, acc_RIL, coefs = train_Lasso(test_data, train_data, Lasso_params)
#     summary.append(["Lasso", test_pred, r2, acc_RIL, coefs])
    
#     print("ElasticNet start")
#     test_pred, r2, acc_RIL, coefs = train_Enet(test_data, train_data, Enet_params)
#     summary.append(["Enet", test_pred, r2, acc_RIL, coefs])
    
#     print("BayesA start")
#     test_pred, r2, acc_RIL, coefs = train_BayesA(test_data, train_data, genotype)
#     summary.append(["BayesA", test_pred, r2, acc_RIL, coefs])
    
#     print("BayesB start")
#     test_pred, r2, acc_RIL, coefs = train_BayesB(test_data, train_data, genotype)
#     summary.append(["BayesB", test_pred, r2, acc_RIL, coefs])
    
#     print("BayesC start")
#     test_pred, r2, acc_RIL, coefs = train_BayesC(test_data, train_data, genotype)
#     summary.append(["BayesC", test_pred, r2, acc_RIL, coefs])
    
#     print("Bayesian Lasso start")
#     test_pred, r2, acc_RIL, coefs = train_BLasso(test_data, train_data, genotype)
#     summary.append(["BLasso", test_pred, r2, acc_RIL, coefs])
    
#     print("Bayesian RR start")
#     test_pred, r2, acc_RIL, coefs = train_BRR(test_data, train_data, genotype)
#     summary.append(["BRR", test_pred, r2, acc_RIL, coefs])
    
#     return summary

# def train_all_models_selected_HBs(family, phenotype, CV, trait_name, HB_num):
    
#     results = []
#     for model in ["GBLUP", "LGBM", "RF", "Lasso", "Enet", "BayesA", "BayesB", "BayesC", "BLasso", "BRR"]:
#         print(model, " start")
        
#         genotype = "../data/PART5_Select_models_and_HBs/selected_genotype/{}_{}_{}.csv.gz".format(trait_name, model, HB_num)
#         test_data, train_data, position_data = make_data_for_prediction.make_data(family, trait_name, genotype, phenotype, CV, seed=1024, plot=False)
        
#         summary = []
#         for i in range(CV):
#             tmp_test_data = test_data[i]
#             tmp_train_data = train_data[i]
            
#             if model == "GBLUP":
#                 test_pred, h2, r2, acc_RIL = train_GBLUP(tmp_test_data, tmp_train_data, genotype)
#                 scores = GWAS(tmp_train_data, genotype)
#                 summary.append(["GBLUP", test_pred, r2, acc_RIL, scores])
                
#             elif model == "LGBM":
#                 LGBM_params = pd.read_csv("../data/PART5_Select_models_and_HBs/LGBM_selected_HBs_{}_params.csv".format(family), index_col=0)
#                 LGBM_params.columns = ["trait", "HB_num", "params"]
#                 LGBM_params = ast.literal_eval(LGBM_params.loc[(LGBM_params["trait"] == trait_name) & (LGBM_params["HB_num"] == HB_num), "params"].values[0])
#                 test_preds, r2, acc_RIL, feature_importances = train_lightgbm(tmp_test_data, tmp_train_data, CV, LGBM_params)
#                 summary.append(["LGBM", test_preds, r2, acc_RIL, feature_importances])
                
#             elif model == "RF":
#                 RF_params = pd.read_csv("../data/PART5_Select_models_and_HBs/RF_selected_HBs_{}_params.csv".format(family), index_col=0)
#                 RF_params.columns = ["trait", "HB_num", "params"]
#                 RF_params = ast.literal_eval(RF_params.loc[(RF_params["trait"] == trait_name) & (RF_params["HB_num"] == HB_num), "params"].values[0])
#                 test_pred, r2, acc_RIL, feature_importance = train_randomforest(tmp_test_data, tmp_train_data, CV, RF_params)
#                 summary.append(["RF", test_pred, r2, acc_RIL, feature_importance])
            
#             elif model == "Lasso":
#                 Lasso_params = pd.read_csv("../data/PART5_Select_models_and_HBs/Lasso_selected_HBs_{}_params.csv".format(family), index_col=0)
#                 Lasso_params.columns = ["trait", "HB_num", "params"]
#                 Lasso_params = ast.literal_eval(Lasso_params.loc[(Lasso_params["trait"] == trait_name) & (Lasso_params["HB_num"] == HB_num), "params"].values[0])
#                 test_pred, r2, acc_RIL, coefs = train_Lasso(tmp_test_data, tmp_train_data, Lasso_params)
#                 summary.append(["Lasso", test_pred, r2, acc_RIL, coefs])
            
#             elif model == "Enet":
#                 Enet_params = pd.read_csv("../data/PART5_Select_models_and_HBs/Enet_selected_HBs_{}_params.csv".format(family), index_col=0)
#                 Enet_params.columns = ["trait", "HB_num", "params"]
#                 Enet_params = ast.literal_eval(Enet_params.loc[(Enet_params["trait"] == trait_name) & (Enet_params["HB_num"] == HB_num), "params"].values[0])
#                 test_pred, r2, acc_RIL, coefs = train_Enet(tmp_test_data, tmp_train_data, Enet_params)
#                 summary.append(["Enet", test_pred, r2, acc_RIL, coefs])
                
#             elif model == "BayesA":
#                 test_pred, r2, acc_RIL, coefs = train_BayesA(tmp_test_data, tmp_train_data, genotype)
#                 summary.append(["BayesA", test_pred, r2, acc_RIL, coefs])
                
#             elif model == "BayesB":
#                 test_pred, r2, acc_RIL, coefs = train_BayesB(tmp_test_data, tmp_train_data, genotype)
#                 summary.append(["BayesB", test_pred, r2, acc_RIL, coefs])
                
#             elif model == "BayesC":
#                 test_pred, r2, acc_RIL, coefs = train_BayesC(tmp_test_data, tmp_train_data, genotype)
#                 summary.append(["BayesC", test_pred, r2, acc_RIL, coefs])
    
#             elif model == "BLasso":
#                 test_pred, r2, acc_RIL, coefs = train_BLasso(tmp_test_data, tmp_train_data, genotype)
#                 summary.append(["BLasso", test_pred, r2, acc_RIL, coefs])
            
#             elif model == "BRR":
#                 test_pred, r2, acc_RIL, coefs = train_BRR(tmp_test_data, tmp_train_data, genotype)
#                 summary.append(["BRR", test_pred, r2, acc_RIL, coefs])
        
#         results.append(summary)
    
#     return results