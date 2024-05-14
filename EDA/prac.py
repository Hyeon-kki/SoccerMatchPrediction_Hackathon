# team별로 지정 기간동안 승리 확률 
def away_day_mean(train, test, day):
    # awayteam
    train[f'away_winRate_{day}mean'] = -1
    test[f'away_winRate_{day}mean'] = -1
    train['win'] = -1
    train['win'] = train['result'].apply(lambda x: 1 if x == 'A' else 0)
    
    teams = train['awayTeam'].unique().values 
    for team in tqdm(teams):
        team_df = train[train['awayTeam'] == team]
        idx = team_df['win'].rolling(day).mean().index.values
        val = team_df['win'].rolling(day).mean().values
        train[f'away_winRate_{day}mean'].loc[idx] = val
        test_idx = test[test['awayTeam'] == team].index
        test[f'away_winRate_{day}mean'].loc[test_idx] = val[-1]
    # train[f'away_winRate_{day}mean'] = train[f'away_winRate_{day}mean'].fillna(0)
    # test[f'away_winRate_{day}mean'] = test[f'away_winRate_{day}mean'].fillna(0)