# %%
# importing the threading module
from datetime import date
import pandas as pd
import time
import sys
# local modules
from binance.client import Client
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
from multiprocessing import Process
import threading

client = Client("9358c5e083cedf0f310b4dd17c1c2be8760b2752a8ca890046c994da91a37b9c", "fa8e91b3077b01fc9f7fba888495b1a9f919a033e13cc51221fdb8e275c25496")


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

    global df2
    root_url = 'https://api.binance.com/api/v1/klines'
    url = root_url + '?symbol=' + symbol + '&interval=' + interval
    data = json.loads(requests.get(url).text)
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
    df1=pd.DataFrame()
    Open=[]
    
    df1['Close']=((df['Open'] + df['High'] + df['Low'] + df['Close'])/4)
    for i in range(len(df)):
        if i==0:
            Open.append(0)

        else:
            Open.append((df['Open'][i-1]+df['Close'][i-1])/2)
    df1['Open']=np.array(Open)
    df1['volume']=df['volume']
    df1['ATR']=df['ATR']
    High=[]
    Low=[]
    High.append(0)
    Low.append(0)
    for i in range(1,len(df)):

        High.append(max(df['High'][i],df1['Close'][i],df1['Open'][i]))
        Low.append(min(df['Low'][i],df1['Close'][i],df1['Open'][i]))

    df1['High']=np.array(High)
    df1['Low']=np.array(Low)

    df1['Stoch-k']=StochRSI(df1['Close'],int(df2['STOCH-PERIOD'][0]),int(df2['STOCH-K'][0]),int(df2['STOCH-D'][0]))[1]*100
    df1['Stoch-d']=StochRSI(df1['Close'],int(df2['STOCH-PERIOD'][0]),int(df2['STOCH-K'][0]),int(df2['STOCH-D'][0]))[2]*100
    df1['RSI']=TA.RSI(df1,int(df2['RSI'][0]))

    # df1['STOCH-k']=TA.STOCH(df1,14)
    # df1['STOCH-D']=TA.STOCHD(df1,14)

    return df1[:-1]

# %%
df3 = pd.read_csv("parameters.csv")
df3.set_index("parameters",inplace=True)
df2=df3.T
tickers=list(df2['symbol'])

def f(name):
    time.sleep(2)
    print('hi')
    time.sleep(2)
    print(name)
# %%

Processes=[]
if __name__ == '__main__':
    for i in range(len(tickers)):
        Processes.append(Process(target=f, args=(tickers[i],)))



for process in Processes:
    process.start()


for process in Processes:
    process.join()