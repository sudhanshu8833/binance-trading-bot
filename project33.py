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

# import telepot
# bot = telepot.Bot('1715056219:AAGxytb3U1gIt1vlVn8Jf5b4za3E1HPuOd4')
# bot.getMe()

# %%
# client = Client((df2['binance_api_key'][0]), str(df2['binance_api_secret_key'][0]))
# BaseClient.FUTURES_TESTNET_URL='https://fapi.binance.{}/fapi'
# BaseClient.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'



client = Client("GBCTCkf6qgDQSZrPJWp513J69pJ2yVC8Fntdos7REMs5kyWn4ICJ2FNKnX9CM7WW", "v0gKOvAfruQaXGbk77W1CsIWf9CVR9kL0U2DEyru2pUwAapXrfyfAMGrEZIdSyaN")  #sudhanshu real api



# client = Client("9358c5e083cedf0f310b4dd17c1c2be8760b2752a8ca890046c994da91a37b9c", "fa8e91b3077b01fc9f7fba888495b1a9f919a033e13cc51221fdb8e275c25496")  #elclis

# BaseClient.FUTURES_TESTNET_URL='https://fapi.binance.{}/fapi'
# BaseClient.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'




# client = Client("cc9feab03d1264ed67c07738cdd42502dd80a8b67fedaf2e5f9b6e9c55a2faad", "e30d3db72358639f29b6280bf1c54fd564e7b1eb5cb13f020739fd197f396e1b") #sudhanshu# client = Client("cc9feab03d1264ed67c07738cdd42502dd80a8b67fedaf2e5f9b6e9c55a2faad", "e30d3db72358639f29b6280bf1c54fd564e7b1eb5cb13f020739fd197f396e1b")
# %%
# while True:
#     data=client.get_all_tickers()
#     symbol=[]
#     prices=[]

#     df=pd.DataFrame()
#     for dat in data:
#         symbol.append(dat['symbol'])
#         prices.append(dat['price'])
        

#     df['symbols']=np.array(symbol)
#     df['price']=np.array(prices)
#     df.to_csv('data.csv')


# %%
# import json

# while True:
#     try:
#         data=client.get_all_tickers()
#         # Data to be written
#         dictionary =data

#         # Serializing json 
#         json_object = json.dumps(dictionary, indent = 2)
        
#         # Writing to sample.json
#         with open("sample.json", "w") as outfile:
#             outfile.write(json_object)
#     except Exception as e:
#         print(e)


# %%

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

# %%
def candle(symbol, interval):

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




# %%
def ltp_price(instrument):
    price=float(client.futures_symbol_ticker(symbol=instrument)['price'])

    return price

    
def market_order(instrument,side,quantity1):
    global position,price1,l
    # BaseClient.FUTURES_TESTNET_URL='https://fapi.binance.{}/fapi'
    # BaseClient.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'

    try:
        
        if side=='buy':
            # order = client.futures_create_order(
            #     symbol=str(instrument),
            #     side=Client.SIDE_BUY,
            #     type=Client.ORDER_TYPE_MARKET,

            #     quantity=quantity1)
            logger.info(f'ordered a buy on {instrument}')

        if side=='sell':
            # order = client.futures_create_order(
            #     symbol=str(instrument),
            #     side=Client.SIDE_SELL,
            #     type=Client.ORDER_TYPE_MARKET,

            #     quantity=quantity1)
            logger.info(f'ordered a sell on {instrument}')
        print('ordered')
    except Exception as e:
        logger.error(str(traceback.format_exc()))



def squareoff(instrument,side,quantity1):

    global position,price2,l
    
    try:

        if side=='buy':
            # order = client.futures_create_order(
            #     symbol=str(instrument),
            #     side=Client.SIDE_BUY,
            #     type=Client.ORDER_TYPE_MARKET,
            #     quantity=quantity1)
            logger.info(f'squared off a sell on {instrument}')

        if side=='sell':
            # order = client.futures_create_order(
            #     symbol=str(instrument),
            #     side=Client.SIDE_SELL,
            #     type=Client.ORDER_TYPE_MARKET,

            #     quantity=quantity1)
            logger.info(f'squared off a buy on {instrument}')
        print('close position')
    except Exception as e:
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

def trade_signal(instrument,l_s,value):

    global ltp,position,high,low,high1,stock,times,price,short_for_candle,quantity3,df
    
    value1=indecision(instrument,-1)
    value2=indecision(instrument,-2)
    for i in range(len(orders)):
        if instrument==orders[i]['symbol']:
            order=orders[i]
            val=i
            break


    signal=""
    if l_s=="" and value=="":
    
        if ((value2[0] and value1[2]) or (value2[1] and value1[2])) and int(df['Stoch-k'][-1])<=int(df['Stoch-d'][-1]) and 98>=int(df['RSI'][-1])>=2:
            signal="sell"
            short_for_candle=5


        if ((value2[3] and df['Open'][-1]>df['Close'][-1]) and (df['High'][-1]>df['Close'][-2] or df['Low'][-1]<df['Open'][-2])) and int(df['Stoch-k'][-1])<=int(df['Stoch-d'][-1]) and 98>=int(df['RSI'][-1])>=2:
            signal="sell" 
            short_for_candle=3

        if ((value2[0] and value1[3]) or (value2[1] and value1[3])) and int(df['Stoch-k'][-1])>=int(df['Stoch-d'][-1]) and 98>=int(df['RSI'][-1])>=2:
            signal="buy"



    elif l_s=="long" and value=="hi":
        if ltp>=order['price']+(order['price']*(float(df2['difference'][0])+order['stop_per']))/100:
            
            order['stoploss']=order['price']+(order['price']*order['stop_per'])/100
            order['stop_per']+=1


            
        
        if ltp<=order['stoploss'] or time.time()>=float(order["time"])+60*int(df2['time_in_mins'][0])*5:
            signal="squareoffsell"
            quantity3=order['quantity']
            orders.pop(val)
            position[instrument]=''


    elif l_s=="short" and value=="hi":
        if ltp<=order['price']-(order['price']*(float(df2['difference'][0])+order['stop_per']))/100:
            
            order['stoploss']=order['price']-(order['price']*order['stop_per'])/100
            order['stop_per']+=1


            
        
        if ltp>=order['stoploss'] or time.time()>=float(order["time"])+60*int(df2['time_in_mins'][0])*int(order['short_for_candle']):
            signal="squareoffbuy"
            quantity3=order['quantity']
            orders.pop(val)
            position[instrument]=''
    return signal    

def get_precision(symbol):
    global info
    for x in info['symbols']:
        if x['symbol'] == symbol:
            return x['quantityPrecision'],x['pricePrecision']


# %%
def main():
    global df2,tickers,df,ltp,orders,j,short_for_candle,info
    
    for ticker in tickers:
        try:
            df=candle(ticker,df2['time_frame'][0])
            
            ltp=ltp_price(ticker)
            
            signal=trade_signal(ticker,position[ticker],"")
            if signal=='buy':
                invest=float(df2['investment'][0])
                quantity=(invest)/ltp
                quantity=quantity*int(df2['binance_X'][0])
                precision=int(get_precision(str(ticker))[0])
                quantity=round(quantity,precision)
                market_order(ticker,'buy',quantity)

                order={}
                position[ticker]='long'
                order['quantity']=quantity
                order['symbol']=ticker
                order['type']='long'
                order['time']=time.time()
                order['stoploss']=df['Close'][-1]-df['ATR'][-1]*float(df2['ATR MULTIPLY'][0])
                order['price']=df['Close'][-1]
                order['stop_per']=1
                order['time']=time.time()
                order['quantity']=quantity
                orders.append(order)


            if signal=='sell':
                invest=float(df2['investment'][0])
                quantity=(invest)/ltp
                quantity=quantity*int(df2['binance_X'][0])
                precision=int(get_precision(str(ticker))[0])
                quantity=round(quantity,precision)

                market_order(ticker,'sell',quantity)

                position[ticker]='short'
                order={}
                order['symbol']=ticker
                order['time']=time.time()
                order['type']='short'
                order['stop_per']=1
                order['short_for_candle']=short_for_candle
                order['stoploss']=df['Close'][-1]+df['ATR'][-1]*float(df2['ATR MULTIPLY'][0])
                order['price']=df['Close'][-1]
                order['quantity']=quantity
                orders.append(order)


            if signal=='squareoffbuy':
                squareoff(ticker,'buy',quantity3)

            if signal=='squareoffsell':
                squareoff(ticker,'sell',quantity3)
        except Exception as e:
            logger.error(str(traceback.format_exc()))

    df=candle(str(tickers[0]),str(df2['time_frame'][0]))
    value=str(df.index[-2])
    while True:
        for ticker in tickers:
            try:
                df=candle(str(ticker),str(df2['time_frame'][0]))
                ltp=ltp_price(ticker)
                print(ltp)
                signal=trade_signal(ticker,position[ticker],"hi")

                if signal=='squareoffbuy':
                    squareoff(ticker,'buy',quantity3)

                if signal=='squareoffsell':
                    squareoff(ticker,'sell',quantity3)

                value1=str(df.index[-2])

                if value1!=value:
                    logger.info('System shifted')
                    j=1
                    break
            except Exception as e:
                logger.error(str(traceback.format_exc()))

        if j==1:
            break          


# %%




df3 = pd.read_csv("parameters.csv")
df3.set_index("parameters",inplace=True)
df2=df3.T
quantity3=0
tickers=list(df2['symbol'])
orders=[]



position={}
for ticker in tickers:
    try:
        position[ticker]=""
        client.futures_change_margin_type(symbol=str(ticker), marginType='ISOLATED')
        client.futures_change_leverage(symbol=str(ticker),leverage=int(df2['binance_X'][0]))
    except Exception as e:
        print(str(e))
short_for_candle=0
info = client.futures_exchange_info()


# df=candle('BTCUSDT',df2['time_frame'][0])
# order={}
# order['symbol']='BTCUSDT'
# order['time']=time.time()
# order['type']='short'
# order['stop_per']=1
# order['short_for_candle']=short_for_candle
# order['stoploss']=df['Close'][-1]+df['ATR'][-1]
# order['price']=df['Close'][-1]
# order['quantity']=0.01
# orders.append(order)
# position['BTCUSDT']='short'

while True:
    try:

        j=0
        main()

    except Exception:

        logger.error(str(traceback.format_exc()))



# %%



