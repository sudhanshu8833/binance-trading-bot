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

import os, time
time.strftime('%X %x %Z')
os.environ['TZ'] = 'Europe/London'
time.tzset()
# %%

df3 = pd.read_csv("parameters.csv")
df3.set_index("parameters",inplace=True)
df2=df3.T



# import telepot
# bot = telepot.Bot('1715056219:AAGxytb3U1gIt1vlVn8Jf5b4za3E1HPuOd4')
# bot.getMe()

# %%
# client = Client((df2['binance_api_key'][0]), str(df2['binance_api_secret_key'][0]))
# BaseClient.FUTURES_TESTNET_URL='https://fapi.binance.{}/fapi'
# BaseClient.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'



client = Client("GBCTCkf6qgDQSZrPJWp513J69pJ2yVC8Fntdos7REMs5kyWn4ICJ2FNKnX9CM7WW", "v0gKOvAfruQaXGbk77W1CsIWf9CVR9kL0U2DEyru2pUwAapXrfyfAMGrEZIdSyaN")  #sudhanshu real api

import websocket
import _thread
import time
import json
instrument='btcusdt'
interval='1m'



import pandas as pd
data={}




# tickers=['btcusdt_perpetual','ethusdt_perpetual','bnbusdt_perpetual','adausdt_perpetual']
df3 = pd.read_csv("parameters.csv")
df3.set_index("parameters",inplace=True)
df2=df3.T
quantity3=0
symbols=list(df2['symbol'])

tickers=[]
for ticker in symbols:
    tickers.append(ticker.lower()+'_perpetual')

socket='wss://fstream.binance.com/stream?streams='
for ticker in tickers:
    socket=socket+str(ticker)+'@continuousKline_30m/'

socket=socket[:-1]


for ticker in tickers:
    data2=ticker.split('_')
    data1=client.futures_klines(symbol=data2[0].upper(),interval='30m')
    for i in range(len(data1)):
        data1[i]=data1[i][:6]
    data[ticker]=data1[:-1]

# for ticker in tickers:
#     data[ticker]=[]

with open("data.json", "w") as outfile:
    json.dump(data,outfile)

def on_message(ws, message):
    json_message=json.loads(message)
    for ticker in tickers:
        if ticker in json_message['stream']:
            candle=json_message['data']['k']
            candle_closed=candle['x']
            if candle_closed==True:
                candle_made=[candle['t'],candle['o'],candle['h'],candle['l'],candle['c'],candle['v']]
                data[ticker].append(candle_made)
                with open("data.json", "w") as outfile:
                    json.dump(data,outfile)


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")


if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(socket,on_message=on_message,on_close=on_close)
    ws.run_forever()