#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import os

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append("..")
from utils import *
from preprocessing_utils import *
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)


# In[2]:


# ema
train, valid = load_train_valid()
train_x, train_y, valid_x, valid_y = preprocessing(train, valid, is_test=False)


# In[3]:


train, valid = load_train_valid()
train_x, train_y, valid_x, valid_y = preprocessing(train, valid, is_test=False)

lec = LabelEncoder()
train_y = lec.fit_transform(train_y)
valid_y = lec.transform(valid_y)
# RF
rf = RandomForestClassifier()
rf.fit(train_x, train_y)
prediction_rf = rf.predict_proba(valid_x)
 
# lightGBM
lgbm = lgb.LGBMClassifier()
lgbm.fit(train_x, train_y, eval_metric='logloss')
prediction_lgbm = lgbm.predict_proba(valid_x)

# XGBoost 
xgbm = xgb.XGBClassifier()
xgbm.fit(train_x, train_y, eval_metric='logloss')
prediction_xgbm = xgbm.predict_proba(valid_x)

# Catboost 
catboost = cat.CatBoostClassifier()
catboost.fit(train_x, train_y)
prediction_catboost = catboost.predict_proba(valid_x)

# loss 
multi_loloss_rf = log_loss(valid_y, prediction_rf)
multi_loloss_lgbm = log_loss(valid_y, prediction_lgbm)
multi_loloss_xgbm = log_loss(valid_y, prediction_xgbm)
multi_loloss_cat = log_loss(valid_y, prediction_catboost)

print(multi_loloss_rf)
print(multi_loloss_lgbm)
print(multi_loloss_xgbm)
print(multi_loloss_cat)


# In[4]:


pd.Series(rf.feature_importances_, index=rf.feature_names_in_).sort_values(ascending=False)


# In[2]:


train, valid = load_train_valid()
train_x, train_y, valid_x, valid_y = preprocessing(train, valid, is_test=False)

lec = LabelEncoder()
train_y = lec.fit_transform(train_y)
valid_y = lec.transform(valid_y)
# RF
rf = RandomForestClassifier()
rf.fit(train_x, train_y)
prediction_rf = rf.predict_proba(valid_x)
 
# lightGBM
lgbm = lgb.LGBMClassifier()
lgbm.fit(train_x, train_y, eval_metric='logloss')
prediction_lgbm = lgbm.predict_proba(valid_x)

# XGBoost 
xgbm = xgb.XGBClassifier()
xgbm.fit(train_x, train_y, eval_metric='logloss')
prediction_xgbm = xgbm.predict_proba(valid_x)

# Catboost 
catboost = cat.CatBoostClassifier()
catboost.fit(train_x, train_y)
prediction_catboost = catboost.predict_proba(valid_x)

# loss 
multi_loloss_rf = log_loss(valid_y, prediction_rf)
multi_loloss_lgbm = log_loss(valid_y, prediction_lgbm)
multi_loloss_xgbm = log_loss(valid_y, prediction_xgbm)
multi_loloss_cat = log_loss(valid_y, prediction_catboost)

print(multi_loloss_rf)
print(multi_loloss_lgbm)
print(multi_loloss_xgbm)
print(multi_loloss_cat)


# In[2]:


train, valid = load_train_valid()
train_x, train_y, valid_x, valid_y = preprocessing(train, valid, is_test=False)

lec = LabelEncoder()
train_y = lec.fit_transform(train_y)
valid_y = lec.transform(valid_y)

# lightGBM
lgbm = lgb.LGBMClassifier()
lgbm.fit(train_x, train_y, eval_metric='logloss')
prediction_lgbm = lgbm.predict_proba(valid_x)

# XGBoost 
xgbm = xgb.XGBClassifier()
xgbm.fit(train_x, train_y, eval_metric='logloss')
prediction_xgbm = xgbm.predict_proba(valid_x)

# Catboost 
catboost = cat.CatBoostClassifier()
catboost.fit(train_x, train_y)
prediction_catboost = catboost.predict_proba(valid_x)

# loss 
multi_loloss_lgbm = log_loss(valid_y, prediction_lgbm)
multi_loloss_xgbm = log_loss(valid_y, prediction_xgbm)
multi_loloss_cat = log_loss(valid_y, prediction_catboost)

print(multi_loloss_lgbm)
print(multi_loloss_xgbm)
print(multi_loloss_cat)


# ## Test

# In[ ]:


train_x, train_y, valid_x, valid_y = load_train_valid()
lec = LabelEncoder()
train_y = lec.fit_transform(train_y)
valid_y = lec.transform(valid_y)
train_x, valid_x= preprocessing(train_x, valid_x)

# feature selection
train_x = train_x.drop(columns=['awayTeam', "day"])
valid_x = valid_x.drop(columns=["awayTeam", "day"])

# 함수 사용해서 이상치 값 삭제

oulier_idx_shotsAwayTeam = get_outlier(df=train_x, column='shots(awayTeam)', weight=1.5)
train_x.drop(oulier_idx_shotsAwayTeam, axis=0, inplace=True)
train_y.drop(oulier_idx_shotsAwayTeam, axis=0, inplace=True)
train_x.reset_index(drop= True, inplace= True)
train_y.reset_index(drop= True, inplace= True)

# lightGBM
lgbm = lgb.LGBMClassifier()
lgbm.fit(train_x, train_y)
prediction_lgbm = lgbm.predict_proba(valid_x)

# XGBoost 
xgbm = xgb.XGBClassifier()
xgbm.fit(train_x, train_y)
prediction_xgbm = xgbm.predict_proba(valid_x)

# Catboost 
catboost = cat.CatBoostClassifier()
catboost.fit(train_x, train_y)
prediction_catboost = catboost.predict_proba(valid_x)

# loss 
multi_loloss_lgbm = log_loss(valid_y, prediction_lgbm)
multi_loloss_xgbm = log_loss(valid_y, prediction_xgbm)
multi_loloss_cat = log_loss(valid_y, prediction_catboost)

print(multi_loloss_lgbm)
print(multi_loloss_xgbm)
print(multi_loloss_cat)


# In[ ]:


train_x, train_y, test = load_train_test()

lec = LabelEncoder()
train_y = lec.fit_transform(train_y)
train_x, test= preprocessing(train_x, test)
train_x = train_x.drop(columns=['awayTeam', "day"])
test = test.drop(columns=["awayTeam", "day"])

#
# lightGBM
lgbm = lgb.LGBMClassifier()
lgbm.fit(train_x, train_y)
prediction_lgbm = lgbm.predict_proba(test)

# XGBoost 
xgbm = xgb.XGBClassifier()
xgbm.fit(train_x, train_y)
prediction_xgbm = xgbm.predict_proba(test)

# Catboost 
catboost = cat.CatBoostClassifier()
catboost.fit(train_x, train_y)
prediction_catboost = catboost.predict_proba(test)

prediction_voting = (prediction_lgbm+prediction_xgbm+prediction_catboost)/3

sample_submission = pd.read_csv('/home/workspace/DACON/soccer/Data/sample_submission.csv')
sample_submission.iloc[:,1:] = prediction_xgbm
sample_submission.to_csv('sample_submission_drop2col_voting.csv', index= False)


# In[2]:


train, valid = load_train_valid()
train_x, train_y, valid_x, valid_y = preprocessing(train, valid, is_test=False)


# In[4]:


stats_columns = [
    'halfTimeGoals(homeTeam)',
    'halfTimeGoals(awayTeam)',
    'shots(homeTeam)',
    'shots(awayTeam)',
    'shotsOnTarget(homeTeam)',
    'shotsOnTarget(awayTeam)',
    'corners(homeTeam)',
    'corners(awayTeam)',
    'fouls(homeTeam)',
    'fouls(awayTeam)',
    'yellowCards(homeTeam)',
    'yellowCards(awayTeam)',
    'redCards(homeTeam)',
    'redCards(awayTeam)'
    ]
pair_stats = train.groupby('match')[stats_columns].mean().reset_index() 


# In[7]:


train.value_counts


# In[11]:


train[stats_columns].ewm(alpha=0.3).mean().values[-1]


# In[12]:


len(stats_columns)


# In[17]:


train['match'].nunique()


# In[32]:


temp = train[train['match'] == "Man United-Aston Villa"]
temp['halfTimeGoals(homeTeam)'].ewm(alpha=0.4).mean().values[-1]


# In[29]:


temp = train[train['match'] == "Man United-Aston Villa"]
temp['halfTimeGoals(homeTeam)']


# In[ ]:




