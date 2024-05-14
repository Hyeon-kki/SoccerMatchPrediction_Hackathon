import pandas as pd

def load_train_valid():
    train = pd.read_csv('/home/workspace/DACON/soccer/Data/train.csv')
    
    # sm의 Logit 모듈 쓸려고 전처리 한것
    # train = train[train['result'] != 'D']
    # train.reset_index(inplace=True, drop= True)

    valid = train[-88:][['season', 'homeTeam', "awayTeam", 'date', 'result']]
    train = train[:-88]

    return train, valid, 

def load_train_test():
    train = pd.read_csv('/home/workspace/DACON/soccer/Data/train.csv')
    test = pd.read_csv('/home/workspace/DACON/soccer/Data/test.csv')

    return train, test

