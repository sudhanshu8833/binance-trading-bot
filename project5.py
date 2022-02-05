# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# generic modules
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
from urllib3.packages.six import X

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

# %%
df3 = pd.read_csv("parameters.csv")
df3.set_index("parameters",inplace=True)
df2=df3.T
import telepot
bot = telepot.Bot('1715056219:AAGxytb3U1gIt1vlVn8Jf5b4za3E1HPuOd4')
bot.getMe()

# %%
client = Client((df2['binance_api_key'][0]), str(df2['binance_api_secret_key'][0]))


# %%


position=""


# %%
def candle(symbol, interval):


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
    High=[]
    Low=[]
    High.append(0)
    Low.append(0)
    for i in range(1,len(df)):

        High.append(max(df['High'][i],df1['Close'][i],df1['Open'][i]))
        Low.append(min(df['Low'][i],df1['Close'][i],df1['Open'][i]))

    df1['High']=np.array(High)
    df1['Low']=np.array(Low)



    new_df = indicators.SuperTrend(df1, int(df2['atr'][0]), int(df2['factor'][0]))

    return new_df


    



# %%
def ltp_price(instrument):
    prices = client.get_all_tickers()
    for i in range(len(prices)):
        if prices[i]['symbol']==str(instrument):
            
            return float(prices[i]['price'])

    


# %%

def market_order():
    global position,price1,l
    
    order = client.futures_create_order(
        symbol=str(df2['symbol_binance'][0]),
        side=Client.SIDE_BUY,
        type=Client.ORDER_TYPE_MARKET,
       
        
        quantity=float(df2['quantity'][0]))

    price1=ltp_price(df2['symbol_binance'][0])
    position='long'
    l=1
    



def market_order1():
    global position,price2,l
    
    order = client.futures_create_order(
        symbol=str(df2['symbol_binance'][0]),
        side=Client.SIDE_SELL,
        type=Client.ORDER_TYPE_MARKET,

        
        quantity=float(df2['quantity'][0]))
    price2=ltp_price(df2['symbol_binance'][0])
    position='short'
    l=1


# %%



# %%
def main():
    global quantity,df,position,price1,price2,l
    times1=time.time()
    df3 = pd.read_csv("parameters.csv")
    df3.set_index("parameters",inplace=True)
    df2=df3.T
    if df2['bot'][0]=='off':
        while True:
            position=""
            df3 = pd.read_csv("parameters.csv")
            df3.set_index("parameters",inplace=True)
            df2=df3.T    
            if df2['bot'].iloc[0]=='on':

                break
    
    # bot.sendMessage(1039725953,str(df2['symbol_binance'][0]))

    df=candle(df2['symbol_binance'][0],str(df2['time_frame'][0]))
    signal=""

    bot.sendMessage(1039725953,str(df['Close'][-1]))
    # position=position_now(df2['symbol_binance'][0])
   

    print(df2['symbol_binance'][0])
    if l==0:
  
        if df.iloc[:,8][-1]!=df.iloc[:,8][-2] and position=="":
            if df.iloc[:,8][-1]=='down':
                signal='sell'
              

            if df.iloc[:,8][-1]=='up':
                signal='buy'
    
    ltp=ltp_price(df2['symbol_binance'][0])
    
    
    if l>0:
        if df.iloc[:,8][-1]!=df.iloc[:,8][-2] and position=="":
            if df.iloc[:,8][-1]=='down':
                signal='sell'

            if df.iloc[:,8][-1]=='up':
                signal='buy'
        
        if (position=="long" and ltp>price1+price1*(float(df2['take_profit_precentage'][0]))/100/int(df2['binance_X'][0])) or(position=="long" and ltp<price1-price1*(float(df2['stop_loss_precentage'][0]))/100/int(df2['binance_X'][0])):
            signal="squareoffsell"

        if (position=="short" and ltp<price2-price2*(float(df2['take_profit_precentage'][0]))/100/int(df2['binance_X'][0])) or(position=="short" and ltp>price2+price2*(float(df2['stop_loss_precentage'][0]))/100/int(df2['binance_X'][0])):
            signal="squareoffbuy"




    if signal=='buy':
        market_order()

        bot.sendMessage(1039725953,f"New long position initiated at {price1}")

    if signal=='sell':
        market_order1()

        bot.sendMessage(1039725953,f"New short position initiated at {price2}")
    
    if signal=="squareoffsell":
        market_order1()

        position=""
        bot.sendMessage(1039725953,f"long position squared of at {price2}")
        bot.sendMessage(1039725953,f" profit of {((price2-price1)/price1)*100*int(df2['binance_X'][0])}")
    if signal=="squareoffbuy":
        market_order()

        position=""
        bot.sendMessage(1039725953,f"short position squared of at {price1}")
        bot.sendMessage(1039725953,f" profit of {((price2-price1)/price1)*100*int(df2['binance_X'][0])}")




l=0
price1=0
price2=0
while True:
    try:

        main()
        print("hello")
    except Exception as e:
        botss=str(e)
        print('something is happening')
        bot.sendMessage(1039725953,botss)

