import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pandas_ta as ta

default_param = {
    'win_size':31,
    'stride':1,
    'split':True,
    'number_y':1,
    'random_state':420,
    'test_size':0.2,
}

# Apply strategy
# such as ma and rsi momentum into the data
def apply_strategy(df, dropna = True):
    df = df.copy(deep=True)
    
    __strategy = ta.Strategy(
        name="main",
        ta=[
            {"kind": "wma", "length": 100},
            {"kind": "wma", "length": 200},
            {"kind": "rsi", "length": 14},
            {"kind": "squeeze", "length": 16, "lazybear":True},
            # {"kind": "bbands", "length": 200, "mamode":"ema"},
        ]
    )

    df.ta.strategy(__strategy)
    if dropna: df.dropna(inplace=True)
    return df

# Split data into windows
# Split data into same size windows.
def windows_split(df, win_size, stride=1):
    max_row = df.shape[0]
    splitted = []
    
    for i in range(0,df.shape[0],stride):
        if i+win_size < max_row:
            window = df.iloc[i:i+win_size]
            splitted.append(window)
        
    return splitted


def windows_split_old(df, win_size, stride=1):
    return [df.iloc[i:i+win_size] for i in range(0,df.shape[0],stride)]


# Normalize the data
def percent_diff_normalize(df, columns, open):
    df = df.copy(deep=True)
    reference = df[open].iloc[0]
    for col in columns:
        series = df[col]
        df[col] = (series-reference)/series
    return df


def split_y(X, number_y=1):
    n_X = []
    n_y = []
    for i in range(len(X)):
        x, y = X[i][:-1*number_y], X[i][-1*number_y:][:4]
        n_X.append(x)
        n_y.append(y)
    return n_X, n_y


def preprocessing(df, param=default_param):
    df = df.copy(deep=True)
    
    # Apply_strategy data into the dataframe.
    p = apply_strategy(df)
    
    # Scale strategy features.
    scale_rsi = p['RSI_14']/100
    max_mom = max(abs(p['SQZ_20_2.0_20_1.5_LB']))
    scale_mom = p['SQZ_20_2.0_20_1.5_LB']/max_mom
    p['RSI_14'] = scale_rsi
    p['SQZ_20_2.0_20_1.5_LB'] = scale_mom
    
    # Drop unnecesary features.
    p.drop(columns=['Date','SQZ_NO','Volume'], inplace=True)

    # Split data into the same windows size.
    pw = windows_split(p, win_size=param['win_size'], stride=param['stride'])
    
    # Percent of difference between timeseries of open, closed, high and low.
    columns = ['Open', 'High', 'Low', 'Close', 'WMA_100', 'WMA_200']
    pw = [ percent_diff_normalize(p, columns, 'Open') for p in pw ]
    
    pw = [ p.fillna(value=0) for p in pw ]
    
    # Convert to numpy array
    pw = [p.to_numpy() for p in pw]
    
    X, y = split_y(pw, param['number_y'])
    X, y = np.stack(X), np.stack(y)
    
    if param['split']:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=param['random_state'])
        return (X_train, X_test, y_train, y_test), list(p.columns)

    return (X, y), list(p.columns)
    # return p