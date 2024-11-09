import optuna
import optuna.integration.lightgbm as lgb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, BaseCrossValidator, cross_val_score
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from lightgbm import early_stopping


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
    
    
def tuning_LightGBM(X, y, optuna_seed=1024, n_jobs=8):
    
    # LightGBM Optuna Tuning
    lgb_train = lgb.Dataset(X, y)
    original_skf = StratifiedKeyBasedKFold(n_splits=5, keys=y.index.str[:3].values, shuffle=True, random_state=1024)
    lgbm_params = {
        'objective': 'regression',
        'n_jobs': 4,
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
    }
    tuner_cv = lgb.LightGBMTunerCV(
        lgbm_params,
        lgb_train,
        num_boost_round=10000,
        optuna_seed=optuna_seed,
        callbacks=[early_stopping(50)],
        folds=original_skf,
    )

    tuner_cv.run()
    print('LightGBM Best params:{}'.format(tuner_cv.best_params))

    best_params = tuner_cv.best_params
    print("  Params: ")
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))
    return best_params

def objective_variable_data_Lasso(X, y, optuna_seed):
    def objective(trial):
        alpha = trial.suggest_float('alpha', 0.0001, 1)
        regr = Lasso(alpha = alpha, random_state=optuna_seed)
        original_skf = StratifiedKeyBasedKFold(n_splits=5, keys=y.index.str[:3].values, shuffle=True, random_state=1024)
        score = cross_val_score(regr, X, y, cv=original_skf, scoring="neg_mean_squared_error")
        return np.mean(score)
    return objective     

def tuning_Lasso(X, y, optuna_seed=1024):
    # Lasso Optuna Tuning
    X = X.fillna(1)
    y = y
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_variable_data_Lasso(X, y, optuna_seed), n_trials=100)
    print('Lasso Best params:{}'.format(study.best_params))
    return study.best_params
    
def objective_variable_data_Enet(X, y, optuna_seed):
    def objective(trial):
        alpha = trial.suggest_float('alpha', 0.0001, 1)
        l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
        regr = ElasticNet(alpha = alpha,
                          l1_ratio = l1_ratio,
                          random_state=optuna_seed)
        original_skf = StratifiedKeyBasedKFold(n_splits=5, keys=y.index.str[:3].values, shuffle=True, random_state=1024)
        score = cross_val_score(regr, X, y, cv=original_skf, scoring="neg_mean_squared_error")
        return np.mean(score)
    return objective 

def tuning_ElasticNet(X, y, optuna_seed=1024):
    # ElasticNet Optuna Tuning
    X = X.fillna(1)
    y = y
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_variable_data_Enet(X, y, optuna_seed), n_trials=100)
    print('ElasticNet Best params:{}'.format(study.best_params))
    return study.best_params

def objective_variable_data_RF(X, y, optuna_seed):
    def objective(trial):
        criterion = trial.suggest_categorical('criterion', ['squared_error', 'absolute_error'])
        bootstrap = trial.suggest_categorical('bootstrap',[True, False])
        max_depth = trial.suggest_int('max_depth', 1, 1000)
        max_features = trial.suggest_categorical('max_features', ['sqrt','log2'])
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 1000)
        n_estimators =  trial.suggest_int('n_estimators', 1, 1000)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 5)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

        regr = RandomForestRegressor(bootstrap = bootstrap, criterion = criterion,
                                     max_depth = max_depth, max_features = max_features,
                                     max_leaf_nodes = max_leaf_nodes,n_estimators = n_estimators,
                                     min_samples_split = min_samples_split,min_samples_leaf = min_samples_leaf,
                                     n_jobs=4)
        original_skf = StratifiedKeyBasedKFold(n_splits=5, keys=y.index.str[:3].values, shuffle=True, random_state=1024)
        score = cross_val_score(regr, X, y, cv=original_skf, scoring="neg_mean_squared_error")
        return np.mean(score)
    return objective

def tuning_RandomForest(X, y, optuna_seed=1024):
    # ElasticNet Optuna Tuning
    X = X.fillna(1)
    y = y
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_variable_data_RF(X, y, optuna_seed), n_trials=100)
    print('RandomForest Best params:{}'.format(study.best_params))
    return study.best_params
