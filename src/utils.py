import pandas as pd

# Load Train Valid
def load_train_valid():
    train = pd.read_csv('/home/workspace/DACON/soccer/Data/train.csv')
    valid = train[-88:][['season', 'homeTeam', "awayTeam", 'date', 'result']]
    train = train[:-88]
    return train, valid
 
# Load Train Test
def load_train_test():
    train = pd.read_csv('/home/workspace/DACON/soccer/Data/train.csv')
    test = pd.read_csv('/home/workspace/DACON/soccer/Data/test.csv')
    return train, test

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
  quantile_25 = np.percentile(df[column].values, 15)
  quantile_75 = np.percentile(df[column].values, 85)

  IQR = quantile_75 - quantile_25
  IQR_weight = IQR*weight
  
  lowest = quantile_25 - IQR_weight
  highest = quantile_75 + IQR_weight
  
  outlier_idx = df[column][ (df[column] < lowest) | (df[column] > highest) ].index
  return outlier_idx

def homeGoal_day_mean(train, test, day):
    # hometeam
    train[f'home_Goal_{day}mean'] = -1
    test[f'home_Goal_{day}mean'] = -1
    
    teams = train['homeTeam'].unique()
    for team in tqdm(teams):
        team_df = train[train['homeTeam'] == team]

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

def awayGoal_day_mean(train, test, day):
    # awayteam
    train[f'away_Goal_{day}mean'] = -1
    test[f'away_Goal_{day}mean'] = -1
    
    teams = train['awayTeam'].unique()
    for team in tqdm(teams):
        team_df = train[train['awayTeam'] == team]

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

def homeWin_day_mean(train, test, day):
    # hometeam
    train[f'home_winRate_{day}mean'] = -1
    test[f'home_winRate_{day}mean'] = -1
    train['win'] = -1
    train['win'] = train['result'].apply(lambda x: 1 if x == 'H' else 0)
    
    teams = train['homeTeam'].unique()
    for team in tqdm(teams):
        team_df = train[train['homeTeam'] == team]
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

def awayWin_day_mean(train, test, day):
    # awayteam
    train[f'away_winRate_{day}mean'] = -1
    test[f'away_winRate_{day}mean'] = -1
    train['win'] = -1
    train['win'] = train['result'].apply(lambda x: 1 if x == 'A' else 0)
    
    teams = train['awayTeam'].unique()
    for team in tqdm(teams):
        team_df = train[train['awayTeam'] == team]
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

def home_day_mean(train, test, columns, day):
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

    # homeTeam awayTeam  최근 3경기 득점량 평균 (성능하락)
    # home_day_mean(train, test, ['halfTimeGoals(homeTeam)', "shots(homeTeam)", 'shotsOnTarget(homeTeam)', 'corners(homeTeam)'], 5)
    # away_day_mean(train, test, ['halfTimeGoals(awayTeam)', 'shots(awayTeam)', 'shotsOnTarget(awayTeam)', 'corners(awayTeam)'], 5)

    # hometeam / awayteam label encoding
    label_dic = dic
    train['homeTeam'] = train['homeTeam'].apply(lambda x: label_dic[x])
    train['awayTeam'] = train['awayTeam'].apply(lambda x: label_dic[x])
    test['homeTeam'] = test['homeTeam'].apply(lambda x: label_dic[x])
    test['awayTeam'] = test['awayTeam'].apply(lambda x: label_dic[x])

    # 5일간 승리 비율
    homeWin_day_mean(train, test, 5)
    awayWin_day_mean(train, test, 5)

    # 6일간 평균 골 비율 
    homeGoal_day_mean(train, test, 6)
    awayGoal_day_mean(train, test, 6)

    # feature selection
    train = train.drop(columns=['matchID', 'goals(homeTeam)', 'goals(awayTeam)',  'home_win'])

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

    # 202324 시즌으로 설정 했을 때 성능이 가장 좋았다. 
    latest_two_season_df = train[train['season'] >= 202324]
    pair_stats = latest_two_season_df.groupby('match')[stats_columns].mean().reset_index() 
    
    pair_stats = train.groupby('match')[stats_columns].mean().reset_index() 
    test_with_stats = test.merge(pair_stats, on='match', how='left')

    '''EWMA 실험''' 
    # 가정: 이전 시즌에 대한 가중치는 낮게 설정하는 것이 성능향상에 도움이 될 것으로 생각된다. 
    # 성능 하락
    # match_df = EWMA(train, test, stats_columns)
    # test_with_stats = test.merge(match_df, on='match', how='left')

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
        
        for case in np.unique(test[i]):
            if case not in le.classes_: 
                le.classes_ = np.append(le.classes_, case)
        
        test[i] = le.transform(test[i])

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

