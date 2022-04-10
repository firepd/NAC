import lightgbm as lgb
import gc
import pandas as pd
import numpy as np
from data import Data
import sklearn.metrics as metrics


class LGBtrain(object): 
    def __init__(self):
        self.base_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': ['binary_logloss'],  # binary_logloss, auc, binary_error
            'nthread': -1,
            'learning_rate': 0.1,
            'num_leaves': 32,
            'max_depth': 6,  # -1 means no limit
            'min_data_in_leaf': 4,
            'min_gain_to_split': 0.01,
            'lambda_l1': 0.01,
            'lambda_l2': 0.01,

            'bagging_fraction': 0.8,
            'bagging_freq': 4,
            'feature_fraction': 1,

            'max_bin': 250,
            'min_data_in_bin': 4,
            'verbosity': -1,
        }
        
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, params=None, evel_func=None, cols_to_use=[], cols_to_drop=[]):
        assert((len(cols_to_use)>0)*1 + (len(cols_to_drop)>0)*1 != 2)

        dtrain = lgb.Dataset(X_train, label=y_train)
        
        data_sets = [dtrain]
        eval_names = ['train']
        
        if X_valid is not None:
            dvalid = lgb.Dataset(X_valid, label=y_valid)
            data_sets.append(dvalid)
            eval_names.append('valid')
        
        if params is not None:
            self.base_params.update(params)
        
        self.model = lgb.train(self.base_params, dtrain, valid_sets=data_sets, valid_names=eval_names, feval=evel_func,
                      verbose_eval=60, num_boost_round=2000, early_stopping_rounds=200)
        
        return self.model
    
    def get_importance(self):
        df_imp = pd.DataFrame({'features': self.model.feature_name(), 
                               'importance':self.model.feature_importance()})
        df_imp = df_imp.sort_values(['importance'], ascending=False)
        return df_imp
    
    def predict(self, X_test):
        return self.model.predict(X_test, num_iteration=self.model.best_iteration)


SEED = 128
np.random.seed(SEED)
path = './data/lol_75W.csv'  # dota2018


dataset = Data(path, team_size=5)
n_individual = dataset.n_individual
encoding = True


def evaluate(pred, label):
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    auc = metrics.roc_auc_score(label, pred)
    pred = np.clip(pred, 0.001, 0.999)
    logloss = metrics.log_loss(label, pred)
    pred = (pred > 0.5) * 1
    acc = (label == pred).sum() / len(label)
    return auc, acc, logloss


train_x, train_y = dataset.get_all(encoding=encoding)
print(train_x.shape)
valid_x, valid_y = dataset.get_all('valid', encoding=encoding)
test_x, test_y = dataset.get_all('test', encoding=encoding)

model = LGBtrain()
model.fit(train_x, train_y, valid_x, valid_y)

pred = model.predict(train_x)
auc, acc, logloss = evaluate(pred, train_y)
print('train auc: {:.4f}, acc: {:.4f}, logloss: {:.4f}'.format(auc, acc, logloss))

pred = model.predict(valid_x)
auc, acc, logloss = evaluate(pred, valid_y)
print('valid auc: {:.4f}, acc: {:.4f}, logloss: {:.4f}'.format(auc, acc, logloss))

pred = model.predict(test_x)
auc, acc, logloss = evaluate(pred, test_y)
print('test auc: {:.4f}, acc: {:.4f}, logloss: {:.4f}'.format(auc, acc, logloss))


