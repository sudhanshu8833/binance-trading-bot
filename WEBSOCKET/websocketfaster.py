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
from datetime import datetime

# %%

df3 = pd.read_csv("parameters.csv")
df3.set_index("parameters",inplace=True)
df2=df3.T

import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

import os, time
time.strftime('%X %x %Z')
os.environ['TZ'] = 'Europe/London'
time.tzset()



# %%


logger.info('it started')

client = Client(df2['binance_api_key'][0], df2['binance_api_secret_key'][0])  



# %%
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


def real_candles(symbol,interval):
    global df2
    BaseClient.testnet=False
    data=client.futures_klines(symbol=symbol,interval=interval)
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

    return df


def time_management(df,times,value):
    
    for i in range(len(df)):
        # print(times,datetime.timestamp(df.index[i]))
        if times<datetime.timestamp(df.index[i]) and value=='candle':
            
            return df[:i]

        if times<datetime.timestamp(df.index[i]) and value=='ltp':

            return df['Close'][i-1]
        
# %%
def candle(symbol, interval,times):

    global df2
    while True:
        try:
            with open("data.json") as json_data_file:
                data = json.load(json_data_file) 

            instrument=symbol.lower()+'_perpetual'
            df = pd.DataFrame(data[str(instrument)])


            df.columns = ['Datetime',
                        'Open', 'High', 'Low', 'Close', 'volume']
            df.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in df.Datetime]
            df['Open']=pd.to_numeric(df["Open"], downcast="float")
            df["High"]=pd.to_numeric(df["High"], downcast="float")
            df["Low"]=pd.to_numeric(df["Low"], downcast="float")
            df["Close"]=pd.to_numeric(df["Close"], downcast="float")
            df["Volume"]=pd.to_numeric(df["volume"], downcast="float")

            ohlc_dict = {                                                                                                             
                'Open': 'first',                                                                                                    
                'High': 'max',                                                                                                       
                'Low': 'min',                                                                                                        
                'Close': 'last',                                                                                                    
                'Volume': 'sum',
            }

            df=df.resample(str(df2['time_in_mins'][0])+'T', closed='left', label='left').apply(ohlc_dict)

            df['ATR']=TA.ATR(df,int(df2['ATR'][0]))

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
            return HAdf

        except Exception as e:
            print(str(e))


# %%
def ltp_price(instrument,times):

    price=float(client.futures_symbol_ticker(symbol=instrument)['price'])

    return price

    
def market_order(instrument,side,quantity1,times):
    global position,price1,l
    # BaseClient.FUTURES_TESTNET_URL='https://fapi.binance.{}/fapi'
    # BaseClient.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'

    try:
        
        if side=='buy':
            order = client.futures_create_order(
                symbol = instrument,
                side = SIDE_BUY,
                positionSide = "LONG",
                type = ORDER_TYPE_MARKET,
                quantity = quantity1,
                )
            logger.info(f'| {datetime.fromtimestamp(times)} |ordered a buy on {instrument} with quantity {quantity1}')

        if side=='sell':
            order = client.futures_create_order(
                symbol = instrument,
                side = SIDE_SELL,
                positionSide = "SHORT",
                type = ORDER_TYPE_MARKET,
                quantity = quantity1,
                )
            logger.info(f'| {datetime.fromtimestamp(times)} |ordered a sell on {instrument} with quantity {quantity1}')
        print('ordered')
    except Exception:
        logger.error(str(traceback.format_exc()))





def squareoff1(instrument,side,quantity1,times):

    global position,price2,l
    

    try:
        if side=='buy':
            order = client.futures_create_order(
                symbol = instrument,
                side = SIDE_BUY,
                positionSide = "SHORT",
                type = ORDER_TYPE_MARKET,
                quantity = quantity1,
                )
            logger.info(f'| {datetime.fromtimestamp(times)} |squared off a sell on {instrument} with quantity {quantity1}')

        if side=='sell':
            order = client.futures_create_order(
                symbol = instrument,
                side = SIDE_SELL,
                positionSide = "LONG",
                type = ORDER_TYPE_MARKET,
                quantity = quantity1,
                )
            logger.info(f'| {datetime.fromtimestamp(times)} |squared off a buy on {instrument} with quantity {quantity1}')
    except Exception:
        logger.error(str(traceback.format_exc()))





# %%
def indecision(instrument,value):
    indecision_red=False
    indecision_green=False
    no_upper=False
    no_lower=False

    if df['Open'].iloc[value]<df['Close'].iloc[value]:
        if abs(df['Open'].iloc[value]-df['Low'].iloc[value])>abs(df['Close'].iloc[value]-df['High'].iloc[value]) and abs(df['Open'].iloc[value]-df['Low'].iloc[value])>abs(df['Open'].iloc[value]-df['Close'].iloc[value]) and abs(df['Close'].iloc[value]-df['High'].iloc[value])>.4*abs(df['Open'].iloc[value]-df['Low'].iloc[value]):
            indecision_green=True


        if abs(df['Open'].iloc[value]-df['Low'].iloc[value])<abs(df['Close'].iloc[value]-df['High'].iloc[value]) and abs(df['Open'].iloc[value]-df['Low'].iloc[value])>abs(df['Open'].iloc[value]-df['Close'].iloc[value]) and .4*abs(df['Close'].iloc[value]-df['High'].iloc[value])<abs(df['Open'].iloc[value]-df['Low'].iloc[value]):
            indecision_green=True


    if df['Open'].iloc[value]>df['Close'].iloc[value]:
        if abs(df['Close'].iloc[value]-df['Low'].iloc[value])>abs(df['Open'].iloc[value]-df['High'].iloc[value]) and abs(df['Close'].iloc[value]-df['Low'].iloc[value])>abs(df['Open'].iloc[value]-df['Close'].iloc[value]) and abs(df['Open'].iloc[value]-df['High'].iloc[value])>.4*abs(df['Open'].iloc[value]-df['Low'].iloc[value]):
            indecision_red=True


        if abs(df['Close'].iloc[value]-df['Low'].iloc[value])<abs(df['Open'].iloc[value]-df['High'].iloc[value]) and abs(df['Close'].iloc[value]-df['Low'].iloc[value])>abs(df['Open'].iloc[value]-df['Close'].iloc[value]) and .4*abs(df['Open'].iloc[value]-df['High'].iloc[value])<abs(df['Close'].iloc[value]-df['Low'].iloc[value]):
            indecision_red=True


    if df['Open'].iloc[value]>df['Close'].iloc[value] and df['High'].iloc[value]==df['Open'].iloc[value]:
        no_upper=True

    if df['Open'].iloc[value]<df['Close'].iloc[value] and df['Low'].iloc[value]==df['Open'].iloc[value]:
        no_lower=True    


    return indecision_red,indecision_green,no_upper,no_lower


# %%

def trade_signal(instrument,times):

    global ltp,position,high,low,high1,stock,price,short_for_candle,quantity3,df
    
    value1=indecision(instrument,-1)
    value2=indecision(instrument,-2)
    for i in range(len(orders)):
        if instrument==orders[i]['symbol']:
            order=orders[i]
            val=i
            break


    signal=""


    if ((value2[0] and value1[2]) or (value2[1] and value1[2])) and int(df['Stoch-k'][-1])<=int(df['Stoch-d'][-1]) and 98>=int(df['RSI'][-1])>=2:
        signal="sell"
        short_for_candle=5


    if ((value2[3] and df['Open'][-1]>df['Close'][-1]) and (df['Open'][-1]>df['Close'][-2] or df['Close'][-1]<df['Open'][-2])) and int(df['Stoch-k'][-1])<=int(df['Stoch-d'][-1]) and 98>=int(df['RSI'][-1])>=2:
        signal="sell" 
        short_for_candle=3

    if ((value2[0] and value1[3]) or (value2[1] and value1[3])) and int(df['Stoch-k'][-1])>=int(df['Stoch-d'][-1]) and 98>=int(df['RSI'][-1])>=2:
        signal="buy"

    return signal


def squareoff(order,l_s,val,times):

    global ltp
    signal=''
    if l_s=="long":
        if ltp>=order['price']+(order['price']*(float(df2['difference'][0])+order['stop_per']))/100:
            
            order['stoploss']=order['price']+(order['price']*order['stop_per'])/100
            order['stop_per']+=float(df2['step'][0])




        if ltp<=order['stoploss'] or times>=float(order["time"])+60*int(df2['time_in_mins'][0])*5:
            signal="squareoffsell"
            quantity3=order['quantity']



    elif l_s=="short":
        if ltp<=order['price']-(order['price']*(float(df2['difference'][0])+order['stop_per']))/100:
            
            order['stoploss']=order['price']-(order['price']*order['stop_per'])/100
            order['stop_per']+=float(df2['step'][0])


        if ltp>=order['stoploss'] or times>=float(order["time"])+60*int(df2['time_in_mins'][0])*int(order['short_for_candle']):
            signal="squareoffbuy"
            quantity3=order['quantity']


    return signal    

def get_precision(symbol):
    global info
    for x in info['symbols']:
        if x['symbol'] == symbol:
            return x['quantityPrecision'],x['pricePrecision']


# %%
def main():
    global df2,tickers,df,ltp,orders,j,short_for_candle,info,orders,times,time1
    
    for ticker in tickers:
        try:

            df=candle(ticker,df2['time_frame'][0],time.time())



            ltp=ltp_price(ticker,time.time())
            # print(ltp)
            signal=trade_signal(ticker,time.time())

            if signal=='buy':
                invest=float(df2['investment'][0])
                quantity=(invest)/ltp
                quantity=quantity*int(df2['binance_X'][0])
                precision=int(get_precision(str(ticker))[0])
                quantity=round(quantity,precision)
                market_order(ticker,'buy',quantity,time.time())

                order={}

                order['quantity']=quantity
                order['symbol']=ticker
                order['type']='long'
                order['time']=time.time()
                order['price']=df['Close'][-1]
                order['stoploss']=df['Close'][-1]-df['ATR'][-1]*float(df2['ATR MULTIPLY'][0])

                if abs((order['price']-order['stoploss'])/order['price'])*100>float(df2['max_loss'][0]):
                    order['stoploss']=df['Close'][-1]-(df['Close'][-1]*float(df2['max_loss'][0]))/100
                
                order['stop_per']=float(df2['step'][0])
                order['time']=time.time()
                order['quantity']=quantity
                orders.append(order)
                print(order)

            if signal=='sell':
                invest=float(df2['investment'][0])
                quantity=(invest)/ltp
                quantity=quantity*int(df2['binance_X'][0])
                precision=int(get_precision(str(ticker))[0])
                quantity=round(quantity,precision)

                market_order(ticker,'sell',quantity,time.time())

        
                order={}
                order['symbol']=ticker
                order['time']=time.time()
                order['type']='short'
                order['stop_per']=float(df2['step'][0])
                order['short_for_candle']=short_for_candle
                order['stoploss']=df['Close'][-1]+df['ATR'][-1]*float(df2['ATR MULTIPLY'][0])
                order['price']=df['Close'][-1]

                if abs((order['price']-order['stoploss'])/order['price'])*100>float(df2['max_loss'][0]):
                    order['stoploss']=df['Close'][-1]+(df['Close'][-1]*float(df2['max_loss'][0]))/100

                order['quantity']=quantity
                orders.append(order)
                print(orders)


        except Exception as e:
            # logger.error(str(traceback.format_exc()))
            print(str(e))


    # df=candle(str(tickers[0]),str(df2['time_frame'][0]),time.time())
    # value=str(df.index[-2])
    # print(value)
    
    while True:
            # print(orders)
        if len(orders)!=0:

            for i in range(len(orders)):

                try:
                    df=candle(str(tickers[0]),str(df2['time_frame'][0]),time.time())
                    ltp=ltp_price(orders[i]['symbol'],time.time())
                    # print(ltp)
                    signal=squareoff(orders[i],orders[i]['type'],i,time.time())

                    if signal=='squareoffbuy':
                        squareoff1(orders[i]['symbol'],'buy',orders[i]['quantity'],time.time())
                        orders.pop(i)
                        print(orders)
                        break

                    if signal=='squareoffsell':
                        squareoff1(orders[i]['symbol'],'sell',orders[i]['quantity'],time.time())
                        orders.pop(i)
                        print(orders)
                        break
                except Exception as e:
                    pass

                if len(orders)==0:
                    break

                if time.time()>=time1:
                    logger.info(f'| {datetime.fromtimestamp(time.time())} | System shifted')
                    j=1
                    break
            if j==1:
                break



        else:

            if time.time()>=time1:
                logger.info(f'| {datetime.fromtimestamp(time.time())} | System shifted')
                j=1

                break       



# %%




df3 = pd.read_csv("parameters.csv")
df3.set_index("parameters",inplace=True)
df2=df3.T
quantity3=0
tickers=list(df2['symbol'])
orders=[]
data={}
print(df2['time_in_mins'][0])
for ticker in tickers:
    try:

        client.futures_change_margin_type(symbol=str(ticker), marginType='ISOLATED')
        
    except Exception as e:
        print(str(e))


    try:
        client.futures_change_leverage(symbol=str(ticker),leverage=int(df2['binance_X'][0]))

    except:
        pass

df=candle(str(tickers[0]),str(df2['time_frame'][0]),time.time())
value=str(df.index[-2])

while True:
    try:
        df=candle(str(tickers[0]),str(df2['time_frame'][0]),time.time())

        value1=str(df.index[-2])
        if value1!=value:
            logger.info(f'| {datetime.fromtimestamp(time.time())} | System shifted')
            time1=datetime.timestamp(df.index[-1])
            break   
           
    except:
        pass
time1+=int(df2['time_in_mins'][0])*60
print(time1)
print(time.time())
short_for_candle=0
info = client.futures_exchange_info()


while True:
    try:
        time1+=int(df2['time_in_mins'][0])*60
        logger.info(f'| {datetime.fromtimestamp(time.time())} | System shifted')
        print(time1)
        j=0
        main()
        print({datetime.fromtimestamp(time.time())})
    except Exception as e:
        # logger.error(str(traceback.format_exc()))
        
        print(str(e))

        