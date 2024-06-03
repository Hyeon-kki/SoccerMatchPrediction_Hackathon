import os
import warnings
import random 
import pandas as pd 
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 


def team_encoding(train):
    train['home_win'] = train['result'].apply(lambda x: 1 if x=='H' else 0) 
    dic = {}
    for team in train['homeTeam'].unique():
        value = train[train['homeTeam'] == team]['home_win'].sum()
        dic[team] = value

    label_dic={}
    for idx, (team, _) in enumerate(sorted(dic.items(), key= lambda x: x[1])):
        label_dic[team] = idx
    
    return label_dic

def EWMA(train, test, columns):
    df_columns = ['match'] + columns
    ema_onMatch = pd.DataFrame(columns = df_columns)
    for idx, match in enumerate(train['match'].unique()):
        temp = train[train['match'] == match]
        value = temp[columns].ewm(alpha=0.3).mean().values[-1]
        value = list(value)
        value.insert(0, match)
        ema_onMatch.loc[idx] = value
    return ema_onMatch

def get_outlier(df=None, column=None, weight=1.5):
  # target 값과 상관관계가 높은 열을 우선적으로 진행
  quantile_25 = np.percentile(df[column].values, 15)
  quantile_75 = np.percentile(df[column].values, 85)

  IQR = quantile_75 - quantile_25
  IQR_weight = IQR*weight
  
  lowest = quantile_25 - IQR_weight
  highest = quantile_75 + IQR_weight
  
  outlier_idx = df[column][ (df[column] < lowest) | (df[column] > highest) ].index
  return outlier_idx

# team별로 지정 기간동안 평균 골 
def homeGoal_day_mean(train, test, day):
    # hometeam
    train[f'home_Goal_{day}mean'] = -1
    test[f'home_Goal_{day}mean'] = -1
    
    teams = train['homeTeam'].unique()
    for team in tqdm(teams):
        team_df = train[train['homeTeam'] == team]
        # day 길이만큼 없는 팀을 고료
        if len(team_df) < day:
            ch_day = len(team_df)  
        else:
            ch_day = day
        idx = team_df['goals(homeTeam)'].rolling(ch_day).mean().index.values
        val = team_df['goals(homeTeam)'].rolling(ch_day).mean().values
        train[f'home_Goal_{day}mean'].loc[idx] = val
        test_idx = test[test['homeTeam'] == team].index
        test[f'home_Goal_{day}mean'].loc[test_idx] = val[-1]
    train[f'home_Goal_{day}mean'] = train[f'home_Goal_{day}mean'].fillna(0)

# team별로 지정 기간동안 평균 골 
def awayGoal_day_mean(train, test, day):
    # awayteam
    train[f'away_Goal_{day}mean'] = -1
    test[f'away_Goal_{day}mean'] = -1
    
    teams = train['awayTeam'].unique()
    for team in tqdm(teams):
        team_df = train[train['awayTeam'] == team]
        # day 길이만큼 없는 팀을 고료
        if len(team_df) < day:
            ch_day = len(team_df)  
        else:
            ch_day = day
        idx = team_df['goals(awayTeam)'].rolling(ch_day).mean().index.values
        val = team_df['goals(awayTeam)'].rolling(ch_day).mean().values
        train[f'away_Goal_{day}mean'].loc[idx] = val
        test_idx = test[test['awayTeam'] == team].index
        test[f'away_Goal_{day}mean'].loc[test_idx] = val[-1]
    train[f'away_Goal_{day}mean'] = train[f'away_Goal_{day}mean'].fillna(0)
    

# team별로 지정 기간동안 승리 확률 
def homeWin_day_mean(train, test, day):
    # hometeam
    train[f'home_winRate_{day}mean'] = -1
    test[f'home_winRate_{day}mean'] = -1
    train['win'] = -1
    train['win'] = train['result'].apply(lambda x: 1 if x == 'H' else 0)
    
    teams = train['homeTeam'].unique()
    for team in tqdm(teams):
        team_df = train[train['homeTeam'] == team]
        # day 길이만큼 없는 팀을 고료
        if len(team_df) < day:
            ch_day = len(team_df)  
        else:
            ch_day = day
        idx = team_df['win'].rolling(ch_day).mean().index.values
        val = team_df['win'].rolling(ch_day).mean().values
        train[f'home_winRate_{day}mean'].loc[idx] = val
        test_idx = test[test['homeTeam'] == team].index
        test[f'home_winRate_{day}mean'].loc[test_idx] = val[-1]
    train.drop(columns=['win'], inplace=True)
    train[f'home_winRate_{day}mean'] = train[f'home_winRate_{day}mean'].fillna(0)

# team별로 지정 기간동안 승리 확률 
def awayWin_day_mean(train, test, day):
    # awayteam
    train[f'away_winRate_{day}mean'] = -1
    test[f'away_winRate_{day}mean'] = -1
    train['win'] = -1
    train['win'] = train['result'].apply(lambda x: 1 if x == 'A' else 0)
    
    teams = train['awayTeam'].unique()
    for team in tqdm(teams):
        team_df = train[train['awayTeam'] == team]
        # day 길이만큼 없는 팀을 고료
        if len(team_df) < day:
            ch_day = len(team_df)  
        else:
            ch_day = day
        idx = team_df['win'].rolling(ch_day).mean().index.values
        val = team_df['win'].rolling(ch_day).mean().values
        train[f'away_winRate_{day}mean'].loc[idx] = val
        test_idx = test[test['awayTeam'] == team].index
        test[f'away_winRate_{day}mean'].loc[test_idx] = val[-1]
    train.drop(columns=['win'], inplace=True)
    train[f'away_winRate_{day}mean'] = train[f'away_winRate_{day}mean'].fillna(0)

# 지정 기간별로 평균 지표 구하기
def home_day_mean(train, test, columns, day):
    # hometeam
    for column in tqdm(columns):
        teams = train['homeTeam'].values
        train[f'home_{column}_{day}mean'] = -1
        test[f'home_{column}_{day}mean'] = -1

        for team in tqdm(teams):
            team_df = train[train['homeTeam'] == team]
            idx = team_df[column].rolling(day).mean().index.values
            val = team_df[column].rolling(day).mean().values
            train[f'home_{column}_{day}mean'].loc[idx] = val
            test_idx = test[test['homeTeam'] == team].index
            test[f'home_{column}_{day}mean'].loc[test_idx] = val[-1]
        train[f'home_{column}_{day}mean'] = train[f'home_{column}_{day}mean'].fillna(0)
        test[f'home_{column}_{day}mean'] = test[f'home_{column}_{day}mean'].fillna(0)

def away_day_mean(train, test, columns, day):
    # awayteam
    for column in tqdm(columns):
        teams = train['awayTeam'].values
        train[f'away_{column}_{day}mean'] = -1
        test[f'away_{column}_{day}mean'] = -1

        for team in tqdm(teams):
            team_df = train[train['awayTeam'] == team]
            idx = team_df[column].rolling(day).mean().index.values
            val = team_df[column].rolling(day).mean().values
            train[f'away_{column}_{day}mean'].loc[idx] = val
            test_idx = test[test['awayTeam'] == team].index
            test[f'away_{column}_{day}mean'].loc[test_idx] = val[-1]
        train[f'away_{column}_{day}mean'] = train[f'away_{column}_{day}mean'].fillna(0)
        test[f'away_{column}_{day}mean'] = test[f'away_{column}_{day}mean'].fillna(0)


def preprocessing(train, test, dic, is_test=False):
    # Date col preprocessing
    train['year'] = train['date'].apply(lambda x : int(x[0:4]))
    train['month'] = train['date'].apply(lambda x : int(x[5:7]))
    train['day'] = train['date'].apply(lambda x : int(x[8:10]))
    test['year'] = test['date'].apply(lambda x : int(x[0:4]))
    test['month'] = test['date'].apply(lambda x : int(x[5:7]))
    test['day'] = test['date'].apply(lambda x : int(x[8:10]))
    train.drop(columns=['date'], inplace=True)
    test.drop(columns=['date'], inplace=True)

    #  match feature create 
    train['match'] = train['homeTeam'] + '-' + train['awayTeam']
    test['match'] = test['homeTeam'] + '-' + test['awayTeam']

    # homeTeam awayTeam  최근 3경기 득점량 평균  
    # home_day_mean(train, test, ['halfTimeGoals(homeTeam)', "shots(homeTeam)", 'shotsOnTarget(homeTeam)', 'corners(homeTeam)'], 5)
    # away_day_mean(train, test, ['halfTimeGoals(awayTeam)', 'shots(awayTeam)', 'shotsOnTarget(awayTeam)', 'corners(awayTeam)'], 5)

    # hometeam / awayteam label encoding ( Test에서 성능 향상)
    label_dic = dic
    train['homeTeam'] = train['homeTeam'].apply(lambda x: label_dic[x])
    train['awayTeam'] = train['awayTeam'].apply(lambda x: label_dic[x])
    test['homeTeam'] = test['homeTeam'].apply(lambda x: label_dic[x])
    test['awayTeam'] = test['awayTeam'].apply(lambda x: label_dic[x])

    # 일정 기간 승리 비율 (성능 향상)
    homeWin_day_mean(train, test, 5)
    awayWin_day_mean(train, test, 5)

    # 성능 하락
    # homeWin_day_mean(train, test, 10)
    # awayWin_day_mean(train, test, 10)
    # homeWin_day_mean(train, test, 20)
    # awayWin_day_mean(train, test, 20)
    # homeWin_day_mean(train, test, 20)

    # 일정 기간 평균 골 비율 (성능 향상)
    homeGoal_day_mean(train, test, 6)
    awayGoal_day_mean(train, test, 6)


    # feature selection
    train = train.drop(columns=['matchID', 'goals(homeTeam)', 'goals(awayTeam)',  'home_win'])

    # 아래의 feature는 훈련에 사용하지 않는다.
    # matchID season date result goals(homeTeam) goals(awayTeam) homeTeam awayTeam	
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

    # 모든 시즌에 대해서 진행해보았으나 test에서 202324 시즌 만 했을 때가 성능이 가장 좋았다. 
    latest_two_season_df = train[train['season'] >= 202324]
    pair_stats = latest_two_season_df.groupby('match')[stats_columns].mean().reset_index() 
    
    pair_stats = train.groupby('match')[stats_columns].mean().reset_index() 
    test_with_stats = test.merge(pair_stats, on='match', how='left')

    '''EWMA 실험''' 
    # 가정: 이전 시즌에 대한 가중치는 낮게 설정하는 것이 성능향상에 도움이 될 것으로 생각된다. 
    # match_df = EWMA(train, test, stats_columns)
    # test_with_stats = test.merge(match_df, on='match', how='left')
    
    # 2부리그에서 1부리그로 처음 올라온 팀은 그전에 경기기록이 없다. 따라서, 평균으로 값 대체 
    # 값이 없는 팀은 신생으로 올라온 2부팀이라서 min값을 넣었지만 성능이 떨어진다. 
    test_with_stats.fillna(train[stats_columns].mean(), inplace=True) # pair_stats mean
    if is_test == True:
        col_list = [col for col in train.columns if col != 'result']
        test = test_with_stats[col_list]
    else:
        test = test_with_stats[train.columns]

    # label encoding
    encoding_target = list(train.dtypes[train.dtypes == "object"].index)
    encoding_target.remove('result')
    for i in encoding_target:
        le = LabelEncoder()
        le.fit(train[i])
        train[i] = le.transform(train[i])
        
        # test 데이터의 새로운 카테고리에 대해 le.classes_ 배열에 추가
        for case in np.unique(test[i]):
            if case not in le.classes_: 
                le.classes_ = np.append(le.classes_, case)
        
        test[i] = le.transform(test[i])

    # outlier 제거 (성능향상) (ppt 정리)
    # oulier_idx_shotsHomeTeam = get_outlier(df=train, column='shots(homeTeam)', weight=1.5)
    # train.drop(oulier_idx_shotsHomeTeam, axis=0, inplace=True)
    # train.reset_index(drop=True, inplace=True)
    # oulier_idx_shotsAwayTeam = get_outlier(df=train, column='shots(awayTeam)', weight=1.5)
    # train.drop(oulier_idx_shotsAwayTeam, axis=0, inplace=True)
    # train.reset_index(drop=True, inplace=True)

    # Scaler (성능하락)
    # scaler = StandardScaler()
    # train = scaler.fit_transform(train)
    # test = scaler.transform(test)

    
    # split X and y (test and valid)

    if is_test:
        train_x = train.drop(columns=['result'])
        train_y = train['result']

        return train_x, train_y, test
    else:
        train_x = train.drop(columns=['result'])
        train_y = train['result']

        test_x = test.drop(columns=['result'])
        test_y = test['result']
        return train_x, train_y, test_x, test_y