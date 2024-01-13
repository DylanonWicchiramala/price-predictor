from typing import Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pandas_ta as ta
import yfinance
import torch

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
        'features_x':['High_delta', 'Low_delta', 'Close_delta', 'RSI_14', 'WMA_100_delta', 'WMA_200_delta'], 
        'features_y':['Close_delta'],
        'convert_to_torch':False,
    }
    
    def __init__(self, dataframe=None, preprocess_param=default_param):
        self.dataframe = dataframe
        self.preprocess_param=preprocess_param
        
        if preprocess_param['split']:
            dataset = self.train_test_split(test_size=preprocess_param['test_size'], random_state=preprocess_param['random_state'])
            dataset['original'] = self.dataset
            self.dataset = dataset
        
    
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
                # {"kind": "squeeze", "length": 16, "lazybear":True},
                # {"kind": "bbands", "length": 200, "mamode":"ema"},
            ]
        )

        df.ta.strategy(__strategy)
        if dropna: df.dropna(inplace=True)
        return df
    

    # Split data into windows
    # Split data into same size windows.
    def windows_split(self, df, win_size, stride=1):
        splitted = []
        
        for i in list(range(0,df.shape[0],stride))[::-1]:
            if i-win_size >= 0:
                window = df.iloc[i-win_size+1:i+1]
                splitted.append(window)
            
        return splitted[::-1]
        
    
    def col_delta_p(self, df, columns, ref_column):
        df = df.copy(deep=True)
        reference = df[ref_column]
        for col in columns:
            series = df[col]
            df[col+'_delta'] = (series-reference)/reference
        df.fillna(value=0, inplace=True)
        return df


    def time_delta_p(self, df, columns):
        df = df.copy(deep=True)
        for col in columns:
            series = df[col]
            sft_series = series.shift(periods=1)
            sft_series[0] = df[col].iloc[0]
            df[col+'_delta'] = (series-sft_series)/sft_series
        df.fillna(value=0, inplace=True)
        return df


    def split_X_y(self, dataset, num_y):
        if num_y == 0:
            return dataset
        else:
            X = [df.iloc[:-num_y] for df in dataset]
            y = [df.iloc[-num_y:] for df in dataset]
            return X, y


    def preprocessing(self, df, param=default_param):
        df = df.copy(deep=True)
        
        # Apply_strategy data into the dataframe.
        p = self.apply_strategy(df)
        
        # Scale strategy features.
        scale_rsi = p['RSI_14']/100
        p['RSI_14'] = scale_rsi
        # max_mom = max(abs(p['SQZ_20_2.0_20_1.5_LB']))
        # scale_mom = p['SQZ_20_2.0_20_1.5_LB']/max_mom
        # p['SQZ_20_2.0_20_1.5_LB'] = scale_mom
        
        
        # compare between 2 columns
        p_norm = self.col_delta_p(p, ['High', 'Low', 'Close'], 'Open')
        p_norm = self.col_delta_p(p_norm, ['WMA_100', 'WMA_200'], 'Close')
        
        # Split data into the same windows size.
        pw = self.windows_split(p_norm, win_size=param['win_size'], stride=param['stride'])
        
        if param['number_y']!=0:
            X, y = self.split_X_y(pw, num_y=param['number_y'])
            X = self.feature_select(X, param['feature_x'])
            y = self.feature_select(y, param['feature_y'])
            if param['convert_to_torch']:
                X = self.to_torch(X)
                y = self.to_torch(y)
            return X, y
        
        else:
            pw = self.feature_select(pw, param['feature_x'])
            if param['convert_to_torch']:
                pw = self.to_torch(pw)
            
            return pw
    
    
    def feature_select(self, dataset, feature):
        if feature is not None:
            return [df[feature] for df in dataset]
        else:
            return dataset
    
    
    def to_torch(self, dataset):
        arr = np.stack([df.to_numpy() for df in dataset])
        return torch.from_numpy(arr).float()


    def train_test_split(self, test_size=0.2, random_state=None):
        ds_train, ds_test = train_test_split(self.dataset, test_size=test_size, random_state=random_state)
        
        dataset = {
            'train': ds_train,
            'test': ds_test,
        }
        return dataset
    
    
if __name__ == '__main__':
    
    v_preprocess_param = {
        'win_size':31,
        'stride':1,
        'split':True,
        'test_size':0.1,
        'random_state':420,
    }

    tickers = 'BTC-USD'

    prices_df_val = loader(tickers=tickers, interval="1d", start='2023-01-01').dataframe

    val_sets = preprocessor(prices_df_val, preprocess_param=v_preprocess_param).dataset
    
    print(val_sets['test'])