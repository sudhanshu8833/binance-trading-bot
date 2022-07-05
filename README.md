# binance-trading-bot

This project was inspired by the observation that all cryptocurrencies pretty much behave in the same way. When one spikes, they all spike, and when one takes a dive, they all do. Pretty much. Moreover, all coins follow Bitcoin's lead; the difference is their phase offset.

So, if coins are basically oscillating with respect to each other, it seems smart to trade the rising coin for the falling coin, and then trade back when the ratio is reversed.





# DRAGON STRATEGY

THIS STRATEGY CONSISTS OF BULLISH AND BEARISH PATTERNS.

OUR SETUP COMES FROM HEIKIN ASHI CHART.
WE SHALL USE DIFFERENT TYPES OF CANDLES IN HEIKIN ASHI CHART, I WILL LIST THEM BELOW..

INDECISION RED CANDLE

INDECISION GREEN CANDLE

NO UPPER WICK RED CANDLE

NO LOWER WICK GREEN CANDLE

EXPLANATION
WHENEVER AN INDECISION CAN CANDLE IS FORMED ON HEIKIN ASHI.

AN INDECISION CANDLE IS CANDLE THAT HAS UPPER WICK AND LOWER WICK. THE LOWER WICK SHOULD BE GREATER OR EQUAL THE DIFFERENCE BETWEEN THE CLOSE VALUE OF HEIKIN ASHI CANDLE AND OPEN VALUE HEIKIN ASHI CANDLE.

O=OPEN OF HEIKIN ASHI H=HIGH OF HEIKIN ASHI C=CLOSE OF HEIKIN ASHI L=LOW OF HEIKIN ASHI

DRAGON STRATEGY

FORMULA OF INDECISION GREENCANDLE=

LOWER WICK IS GREATER OR EQUAL (O - C ) (SUBSTRACTION) WHERE C -L IS NOT EQUAL TO ZERO ( CLOSE MINUS LOW) THE UPPER WICK SHOULD GREATER OR EQUAL 40% OF C - L

EXAMPLE
IF C - L = 10, UPPER WICK VALUE SHOULD BE GREATER OR EQUAL 41% OF 10 WHICH IS 4 ALSO VICEVERSA IF UPPER WICK IS TALLER THAN LOWER WICK . THE LOWER WICK SHOULD BE AT LEAST 40% OF UPPER WICK

FORMULA FOR HEIKIN ASHI RED INDECISION CANDLE =

LOWER WICK OF HEIKIN ASHI IS GREATER OR EQUAL CLOSE - OPEN. WHERE C -L IS NOT EQUAL TO ZERO ( CLOSE MINUS LOW) CANNOT BE ZERO
THE UPPER WICK SHOULD GREATER OR EQUAL 41% OF C – L

DRAGON STRATEGY

EXAMPLE
IF O - L = 10, UPPER WICK VALUE SHOULD BE GREATER OR EQUAL 41% OF
10 WHICH IS 4.1 ALSO VICEVERSA . THE LOWER WICK SHOULD BE AT LEAST 41% OF UPPER WICK.

DRAGON STRATEGY

FORMULA FOR NO UPPER WICK RED CANDLE
NO UPPER WICK RED/ BEARISH CANDLE IS A RED HEIKIN ASHI CANDLE WHICH ITS OPEN VALUE IS EXACTLY EQUAL TO THE HIGH OF THE CANDLE.

DRAGON STRATEGY

FORMULA FOR NO LOWER WICK GREEN CANDLE
A NO LOWER WICK GREEN CANDLE IS A GREEN/ BULLISH HEIKIN ASHI CANDLE WHICH IS OPEN VALUE IS EXACTLY EQUAL TO LOW VALUE OF THE HEIKIN ASHI CANDLE. O=L

THIS IS A NO LOWER WICK GREEN HEIKIN ASHI CANDLE ALSO WE SHALL USE THE FOLLOWING INDICATORS FROM TRADING VIEW NAMELY:

STOCHASTIC RSI
RELATIVE STRENGTH LINE
AVERAGE TRUE RANGE( ATR PIPS BY LAZY BEAR { FIXED BY ELIXIUM} )


#Tool Setup
Install Python dependencies
https://www.python.org/downloads/
Install Pip
https://pip.pypa.io/en/stable/installation/
To download all the requirements :
Run the following line in the terminal: pip install -r requirements.txt.

Run program
You would be able to see all the orders that has been generated.

python3 project33new.py
