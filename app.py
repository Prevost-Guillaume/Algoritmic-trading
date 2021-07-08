#################################################################################################################################################
#                                                                                                                                               #
#                                                        IMPORTING PACKAGES                                                                     #
#                                                                                                                                               #
#################################################################################################################################################
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import SGDRegressor
from scipy import signal
import random

import datetime

import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs.scatter import Line

import pyAesCrypt
key = st.secrets["key"]
for filename in ['Indicateurs.py', 'main.py', 'strategy_functions.py', 'regressor.sav', "data/all_data_3.p"]:
    pyAesCrypt.decryptFile(filename+".aes", filename, key)


from Indicateurs import *
from strategy_functions import *



#################################################################################################################################################
#                                                                                                                                               #
#                                                               MAIN                                                                            #
#                                                                                                                                               #
#################################################################################################################################################

### Get data
data = getData('data/all_data_3.p')
date_split = pd.Timestamp('2017-01-01')

train_dic, test_dic = traintestdicosplit(data, date_split=date_split)
x_train, y_train, x_test, y_test = extractXY(train_dic, test_dic, rate_of_0=.5)

x_train = preprocessing(x_train)
x_test = preprocessing(x_test)


if False:
    # Checkpoint if necessary
    data = [x_train, x_test, y_train, y_test]
    f = open('data/train_test_data.p', 'wb')
    pickle.dump(data, f)
    f.close()
        
    f = open('data/train_test_data.p', 'rb')
    x_train, x_test, y_train, y_test = pickle.load(f)
    print(x_train.columns)

    f.close()
    


### Modellisation
##model = SGDRegressor(random_state=0)
##model.fit(x_train, y_train)
##pickle.dump(model, open('regressor.sav', 'wb'))
model = pickle.load(open('regressor.sav', 'rb'))


def get_trade_df(ticker):

    df = test_dic[ticker]
    df = df.drop(["y"], axis=1)
    x = preprocessing(df)

    # Make predictions
    df['pred'] = model.predict(x)

    # filter prediction
    b, a = signal.butter(3, 0.1, btype='lowpass')
    df['filteredpred'] = signal.filtfilt(b, a, df['pred'])
    df["filteredpred"] = 10*df["filteredpred"]/max(df["filteredpred"])


    # ML indicator moving average strategy
    df['ma10d'] = pd.Series(df["filteredpred"].rolling(window=6, center=False).mean())
    df['diff'] = (df['ma10d'] - df['filteredpred']).apply(np.sign)
    df['signal_ma'] = derivate(df['diff'])/2
    df['signal_ma'] = rectifieSignal(df['signal_ma'], shift=True)

    # Basic moving average strategy
    df['ma10d'] = pd.Series(df["Open"].rolling(window=10, center=False).mean())
    df['ma20d'] = pd.Series(df["Open"].rolling(window=20, center=False).mean())
    df['diff'] = (df['ma10d'] - df['ma20d']).apply(np.sign)
    df['signal2'] = derivate(df['diff'])/2
    df['signal2'] = rectifieSignal(df['signal2'], shift=True)


    # Backtest
    df['capital_ma'] = backtest(df['Open'], df['signal_ma'])
    df['capital_naive'] = backtest(df['Open'], df['signal2'])


    # Analyse strategy
    stats = getStats(df['signal_ma'], df['capital_ma'], df['Open'])


    return df, stats






#####################################################################################################################################################
#                                                                                                                                                   #
#                                                           DASHBOARD WITH STREAMLIT                                                                #
#                                                                                                                                                   #
#####################################################################################################################################################
st.title('Algoritmic trading')

###############
st.write("Automation of trading")
st.write("When to buy and sell ?")

### SIDEBAR PARAMETERS ###
st.sidebar.title('Parameters')
ticker = st.sidebar.selectbox('Select ticker', [i for i in train_dic]+[i for i in test_dic])

df, stats = get_trade_df(ticker)
df['date'] = df.index
df['return (our strategy)'] = df['capital_ma']
df['return (moving average strategy)'] = df['capital_naive']
indics = ['Open', 'Volume', 'MACD', 'RSI', 'ADL', 'CO',
       'WilliamR', 'Momentum', 'ROC', 'Volatility', '20d', '50d', 'de20d',
       'de50d', 'lowfiltered3', '20d-50d', 'de20d-de50d', 'Open-lowfiltered3',
       'BB_UPPER', 'BB_LOWER', 'BB_MIDDLE', 'BB_WIDTH', 'Open-BB_UPPER','Open-BB_LOWER',
       'Open-BB_MIDDLE', 'pivots_0', 'pivots_1', 'logOpen', 'loglowfiltered3',
       'filteredpred', 'return (our strategy)','return (moving average strategy)']


indics = st.sidebar.multiselect('Select indicators',
                                indics,
                                default=['Open','return (our strategy)','return (moving average strategy)'])
signals = st.sidebar.checkbox('Plot signals')


### PLOT CHART ###
df_0 = df[df['signal_ma'] < 0]
df_1 = df[df['signal_ma'] > 0]

##fig = px.line(df, x="date", y=['Open']+indics,
##              hover_data={"date": "|%B %d, %Y"},
##              title='custom tick labels')
##fig.add_trace(px.scatter(df_0, x='date', y='Open', color="red", marker_symbol='triangle-down'))
##fig.add_trace(px.scatter(df_1, x='date', y='Open', color="green", marker_symbol='triangle-up'))

##fig.add_trace(go.Figure(go.Scatter(mode="markers", x=df_0['date'], y=df_0['Open'], marker_symbol=6,
##                           marker_line_color="midnightblue", marker_color="lightskyblue")))

if signals:
    trace1 = go.Scatter(mode='markers', name='Sell', x=df_0['date'], y=df_0['Open'], marker_color='red', marker_symbol=6)
    trace2 = go.Scatter(mode='markers', name='Buy', x=df_1['date'], y=df_1['Open'], marker_color='green', marker_symbol=5)
traces = []
for indic in indics:
    traces.append(go.Scatter(name=indic, x=df['date'], y=df[indic]))

fig = make_subplots(rows=1, cols=1)
if signals:
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=1)
for t in traces:
    fig.add_trace(t, row=1, col=1)

fig.update_layout(title="Trading simulation results", width=800, height=500)

##fig.update_xaxes(
##    dtick="M1",
##    tickformat="%Y")
##
##fig = go.Figure([go.Scatter(x=df.index, y=df['Open'])])
st.plotly_chart(fig)


### STATS ###

expander = st.beta_expander("Statistics")
for s in stats:
    expander.write(s+' : '+str(stats[s]))







    
