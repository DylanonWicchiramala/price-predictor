from typing import Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pandas_ta as ta
import yfinance

class loader():
    
    def __init__(self, tickers:str, *arg, **kwargs):
        self.dataframe = self.load_data_from_yfinance(tickers, *arg, **kwargs)
    
    
    def __call__(self):
        return self.dataframe
    
    
    def load_data_from_yfinance(self, tickers:str, *args, **kwargs):
        tick = yfinance.Ticker(tickers)
        price_df = tick.history(*args, **kwargs)[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
        return price_df
    


class preprocessor():

    default_param = {
        'win_size':31,
        'stride':1,
        'number_y':1,
        'split':False,
        'random_state':420,
        'test_size':0.2,
    }
    
    def __init__(self, dataframe=None, preprocess_param=default_param):
        self.dataframe = dataframe
        self.preprocess_param=preprocess_param
        
        self.dataset = self.preprocessing(self.dataframe, preprocess_param)
        
        if preprocess_param['split']:
            self.split(test_size=preprocess_param['test_size'], random_state=preprocess_param['random_state'])
        
    
    def __call__(self):
        return self.dataset
    
    
    # Apply strategy
    # such as ma and rsi momentum into the data
    def apply_strategy(self, df, dropna = True):
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
    

    def windows_split_old(self, df, win_size, stride=1):
        max_row = df.shape[0]
        splitted = []
        
        for i in range(0,df.shape[0],stride):
            if i+win_size < max_row:
                window = df.iloc[i:i+win_size]
                splitted.append(window)
            
        return splitted
    
    
    # Split data into windows
    # Split data into same size windows.
    def windows_split(self, df, win_size, stride=1):
        splitted = []
        
        for i in list(range(0,df.shape[0],stride))[::-1]:
            if i-win_size >= 0:
                window = df.iloc[i-win_size+1:i+1]
                splitted.append(window)
            
        return splitted[::-1]


    def windows_split_old(self, df, win_size, stride=1):
        return [df.iloc[i:i+win_size] for i in range(0,df.shape[0],stride)]


    # Normalize the data
    def percent_diff_normalize(self, df, columns, open):
        df = df.copy(deep=True)
        reference = df[open].iloc[0]
        for col in columns:
            series = df[col]
            df[col] = (series-reference)/reference
        df.fillna(value=0, inplace=True)
        return (reference, df)


    def split_y(self, X, number_y=1):
        n_X = []
        n_y = []
        for i in range(len(X)):
            if number_y==0:
                x, y = X[i][:], [X[i][-1:][0][:4]]
            else:
                x, y = X[i][:-1*number_y], [X[i][-1*number_y:][0][:4]]
            n_X.append(x)
            n_y.append(y)
        return n_X, n_y


    def preprocessing(self, df, param=default_param):
        df = df.copy(deep=True)
        
        # Apply_strategy data into the dataframe.
        p = self.apply_strategy(df)
        
        # Scale strategy features.
        scale_rsi = p['RSI_14']/100
        max_mom = max(abs(p['SQZ_20_2.0_20_1.5_LB']))
        scale_mom = p['SQZ_20_2.0_20_1.5_LB']/max_mom
        p['RSI_14'] = scale_rsi
        p['SQZ_20_2.0_20_1.5_LB'] = scale_mom
        
        # Drop unnecesary features.
        p.drop(columns=['SQZ_NO','Volume','SQZ_ON','SQZ_OFF'], inplace=True)
        
        # Split data into the same windows size.
        pw = self.windows_split(p, win_size=param['win_size'], stride=param['stride'])
        
        # Percent of difference between timeseries of open, closed, high and low.
        target_col = ['Open', 'High', 'Low', 'Close', 'WMA_100', 'WMA_200']
        pw_norm = [self.percent_diff_normalize(p, target_col, 'Open') for p in pw]
        
        initial_price, pw_norm = zip(*pw_norm)
        current_date = [p.iloc[-1]['Date'] for p in pw_norm]
        
        for p in pw_norm:
            p.drop(columns=['Date'], inplace=True)
        
        columns = pw_norm[0].columns
        
        # Convert to numpy array
        pw_norm = [p.to_numpy() for p in pw_norm]
        
        X, y = self.split_y(pw_norm, param['number_y'])
        X, y = np.stack(X), np.stack(y)
        
        return {
            "columns": columns,
            "initial price": initial_price,
            "current date": current_date,
            "x": X,
            'y': y,
        }
        
        
    
    def split(self, test_size=0.2, random_state=None):
        X, y = self.dataset['x'], self.dataset['y']
        columns = self.dataset['columns']
        initial_price = self.dataset['initial price']
        current_date = self.dataset['current date']
        
        X_train, X_test, y_train, y_test, initial_price_train, initial_price_test, current_date_train, current_date_test = train_test_split(X, y, initial_price, current_date, test_size=test_size, random_state=random_state)
        
        self.dataset['train'] = {
            "columns": columns,
            "initial price": initial_price_train,
            "current date": current_date_train,
            'x':X_train,
            'y':y_train,
            }
        self.dataset['test'] = {
            "columns": columns,
            "initial price": initial_price_test,
            "current date": current_date_test,
            'x':X_test,
            'y':y_test,
            }
        return self.dataset
    
    
    
if __name__ == '__main__':
    
    v_preprocess_param = {
        'win_size':31,
        'stride':1,
        'split':True,
        'test_size':0.1,
        'number_y':1,
        'random_state':420,
    }

    tickers = 'BTC-USD'

    prices_df_val = loader(tickers=tickers, interval="1d", start='2023-01-01').dataframe

    val_sets = preprocessor(prices_df_val, preprocess_param=v_preprocess_param).dataset
    
    print(val_sets['test'])