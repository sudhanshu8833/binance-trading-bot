# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%


from datetime import date
import pandas as pd
import time
import sys
import traceback
# local modules
from binance.client import Client
from binance.client import BaseClient
from binance.enums import *
from indicator import indicators


# local file
import secrets
import json
import yfinance as yf
import numpy as np
import datetime as dt
import requests 
import json 
import pandas as pd 
import numpy as np  
import requests
import time
import urllib
from finta import TA

# %%

df3 = pd.read_csv("parameters.csv")
df3.set_index("parameters",inplace=True)
df2=df3.T
import numpy as np  
import requests
import time
import urllib
from finta import TA

# %%

df3 = pd.read_csv("parameters.csv")
df3.set_index("parameters",inplace=True)
df2=df3.T

import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
BaseClient.FUTURES_TESTNET_URL='https://fapi.binance.{}/fapi'
BaseClient.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'

# client = Client("cc9feab03d1264ed67c07738cdd42502dd80a8b67fedaf2e5f9b6e9c55a2faad", "e30d3db72358639f29b6280bf1c54fd564e7b1eb5cb13f020739fd197f396e1b")  #elclis
client = Client("86c637e3177c4280ca59adfeaab720357561be2c817eccf84cc0e3470a666c93", "f77d6c868d6caf55cb9b04e980b7aa18ce729fc1685ab1417e1bea820c7ac3ef")  #elclis


pd.options.mode.chained_assignment = None



def StochRSI(series, period=14, smoothK=3, smoothD=3):
    # Calculate RSI 
    delta = series.diff().dropna()
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period-1]] = np.mean( ups[:period] ) #first value is sum of avg gains
    ups = ups.drop(ups.index[:(period-1)])
    downs[downs.index[period-1]] = np.mean( downs[:period] ) #first value is sum of avg losses
    downs = downs.drop(downs.index[:(period-1)])
    rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() /          downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() 
    rsi = 100 - 100 / (1 + rs)

    # Calculate StochRSI 
    stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    stochrsi_K = stochrsi.rolling(smoothK).mean()
    stochrsi_D = stochrsi_K.rolling(smoothD).mean()

    return stochrsi, stochrsi_K, stochrsi_D

# %%
def candle(symbol, interval):

    global df2,client

    client = Client("GBCTCkf6qgDQSZrPJWp513J69pJ2yVC8Fntdos7REMs5kyWn4ICJ2FNKnX9CM7WW", "v0gKOvAfruQaXGbk77W1CsIWf9CVR9kL0U2DEyru2pUwAapXrfyfAMGrEZIdSyaN")  #sudhanshu real api

    BaseClient.FUTURES_TESTNET_URL='https://fapi.binance.{}/fapi'
    BaseClient.FUTURES_URL = 'https://fapi.binance.{}/fapi'
    BaseClient.testnet=False
    data=client.futures_klines(symbol=symbol,interval=interval)

    # BaseClient.FUTURES_TESTNET_URL='https://fapi.binance.{}/fapi'
    # BaseClient.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
    client = Client("cc9feab03d1264ed67c07738cdd42502dd80a8b67fedaf2e5f9b6e9c55a2faad", "e30d3db72358639f29b6280bf1c54fd564e7b1eb5cb13f020739fd197f396e1b")  #sudhanshu real api


    df = pd.DataFrame(data)
    df.columns = ['Datetime',
                'Open', 'High', 'Low', 'Close', 'volume',
                'close_time', 'qav', 'num_trades',
                'taker_base_vol', 'taker_quote_vol', 'ignore']
    df.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in df.close_time]
    
    df.drop(['close_time','qav','num_trades','taker_base_vol', 'taker_quote_vol', 'ignore'],axis=1,inplace=True)
           
    
    
    df['Open']=pd.to_numeric(df["Open"], downcast="float")
    df["High"]=pd.to_numeric(df["High"], downcast="float")
    df["Low"]=pd.to_numeric(df["Low"], downcast="float")
    df["Close"]=pd.to_numeric(df["Close"], downcast="float")
    df["volume"]=pd.to_numeric(df["volume"], downcast="float")
    df['ATR']=TA.ATR(df,int(df2['ATR'][0]))
    print(df)
    HAdf = df[['Open', 'High', 'Low', 'Close']]

    HAdf['Close'] = round(((df['Open'] + df['High'] + df['Low'] + df['Close'])/4),4)


    for i in range(len(df)):
        if i == 0:
            HAdf.iat[0,0] = round(((df['Open'].iloc[0] + df['Close'].iloc[0])/2),4)
        else:
            HAdf.iat[i,0] = round(((HAdf.iat[i-1,0] + HAdf.iat[i-1,3])/2),4)

    HAdf['High'] = HAdf.loc[:,['Open', 'Close']].join(df['High']).max(axis=1)
    HAdf['Low'] = HAdf.loc[:,['Open', 'Close']].join(df['Low']).min(axis=1)


    HAdf['Stoch-k']=StochRSI(HAdf['Close'],int(df2['STOCH-PERIOD'][0]),int(df2['STOCH-K'][0]),int(df2['STOCH-D'][0]))[1]*100
    HAdf['Stoch-d']=StochRSI(HAdf['Close'],int(df2['STOCH-PERIOD'][0]),int(df2['STOCH-K'][0]),int(df2['STOCH-D'][0]))[2]*100
    HAdf['RSI']=TA.RSI(HAdf,int(df2['RSI'][0]))
    HAdf['ATR']=df['ATR']
    # df1['STOCH-k']=TA.STOCH(df1,14)
    # df1['STOCH-D']=TA.STOCHD(df1,14)

    return HAdf[:-1]

candle('BTCUSDT','5m')



# %%
