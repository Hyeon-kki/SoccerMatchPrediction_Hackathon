import os
import warnings
import random
from utils import *
import pandas as pd 
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


big_6 = ['Man City', 'Arsenal', 'Liverpool', 'Aston Villa', 'Tottenham', 'Man United']
train, test = load_train_test()
train['big_6'] = train['homeTeam'].apply(lambda x: 1 if x in big_6 else 0)
test['big_6'] = test['homeTeam'].apply(lambda x: 1 if x in big_6 else 0)
test.reset_index(drop= True, inplace= True)

hometeam_list = list(train['homeTeam'].unique())
dic = team_encoding(train)

lec = LabelEncoder()
lec.fit(train['result'])

result_array = np.zeros((len(test), 3))
multi_logloss_sum = 0
is_big6 = [0, 1]

for idx in is_big6:
    team_train = train[train['big_6'] == idx]
    team_test = test[test['big_6'] == idx]
    test_idx = team_test.index.values
    train_x, train_y, test_x= preprocessing(team_train, team_test, dic, is_test=True)
    train_y = lec.transform(train_y)

    if len(test) != 0:
        model = LogisticRegression(max_iter=100, penalty='l2', C=1.0)
        model.fit(train_x, train_y) 
        prediction = model.predict_proba(test_x)
        result_array[test_idx] = prediction

    else:
        continue

sample_submission = pd.read_csv('/home/workspace/DACON/soccer/Data/sample_submission.csv')
sample_submission.iloc[:,1:] = result_array
sample_submission.to_csv('Final.csv', index= False)