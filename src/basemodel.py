#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import random
import os
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings("ignore")


seed_everything(42)


train = pd.read_csv('/home/workspace/DACON/soccer/Data/train.csv')
test = pd.read_csv('/home/workspace/DACON/soccer/Data/test.csv')

train['year'] = train['date'].apply(lambda x : int(x[0:4]))
train['month'] = train['date'].apply(lambda x : int(x[5:7]))
train['day'] = train['date'].apply(lambda x : int(x[8:10]))
train.drop(columns=['date'], inplace=True)

test['year'] = test['date'].apply(lambda x : int(x[0:4]))
test['month'] = test['date'].apply(lambda x : int(x[5:7]))
test['day'] = test['date'].apply(lambda x : int(x[8:10]))
test.drop(columns=['date'], inplace=True)


train['home_win'] = train['result'].apply(lambda x: 1 if x=='H' else 0) 
dic = {}
for team in train['homeTeam'].unique():
    value = train[train['homeTeam'] == team]['home_win'].sum()
    dic[team] = value

label_dic={}
for idx, (team, _) in enumerate(sorted(dic.items(), key= lambda x: x[1])):
    label_dic[team] = idx


# 빠진 것 matchID season date result goals(homeTeam) goals(awayTeam) homeTeam awayTeam	
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

train['match'] = train['homeTeam'] + '-' + train['awayTeam']
pair_stats = train.groupby('match')[stats_columns].mean().reset_index() # match mean

# test_with_stats
test['match'] = test['homeTeam'] + '-' + test['awayTeam']
test_with_stats = test.merge(pair_stats, on='match', how='left')
test_with_stats.fillna(pair_stats[stats_columns].mean(), inplace=True) # pair_stats mean


train_x = train.drop(columns=['matchID', 'goals(homeTeam)', 'goals(awayTeam)', 'result'])
train_y = train['result']

test_x = test_with_stats.drop(columns=['matchID'])
test_x = test_x[train_x.columns]


from sklearn.preprocessing import LabelEncoder

encoding_target = list(train_x.dtypes[train_x.dtypes == "object"].index)

for i in encoding_target:
    le = LabelEncoder()
    le.fit(train_x[i])
    train_x[i] = le.transform(train_x[i])
    
    # test 데이터의 새로운 카테고리에 대해 le.classes_ 배열에 추가
    for case in np.unique(test_x[i]):
        if case not in le.classes_: 
            le.classes_ = np.append(le.classes_, case)
    
    test_x[i] = le.transform(test_x[i])


model = LogisticRegression(max_iter=100,
                           penalty='l2',
                           C=1.0)


model.fit(train_x, train_y) 
prediction = model.predict_proba(test_x)

display(model.classes_)
display(prediction)


sample_submission = pd.read_csv('/home/workspace/DACON/soccer/Data/sample_submission.csv')
sample_submission

sample_submission.iloc[:,1:] = prediction
sample_submission

sample_submission.to_csv('baseline_submission.csv', index=False)