import pandas as pd
pd.options.mode.chained_assignment = None
import quandl as qd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from matplotlib import style
import datetime, time
#import pickle
##to serialize an object; in this case, we use this midule for saving the trained model

def time_series(df, steps):
    df_t = np.empty([steps,steps])
    for i in range(steps):
        if i == 0:
            df_t[i,:] = np.array(df['Adj. Close'][-steps:])
        else:     
            df_t[i,:] = np.array(df['Adj. Close'][-(steps+i):-i])
    return df_t
    df_t
data = qd.get("WIKI/GOOGL")

#import pandas_datareader as pdr
#apple = pdr.get_data_yahoo('AAPL')
## alternative for shorter data

data_Adj = data[['Adj. Open', 'Adj. High', 'Adj. Low' ,'Adj. Close', 'Adj. Volume' ]]

data_Adj.loc[:,'HL_PCT'] = ((data_Adj['Adj. High']-data_Adj['Adj. Close'])/data_Adj['Adj. Close']) * 100.0
data_Adj.loc[:,'PCT_change'] = ((data_Adj['Adj. Close']-data_Adj['Adj. Open'])/data_Adj['Adj. Open'] ) * 100.0
df = data_Adj[['Adj. Close', 'HL_PCT', 'PCT_change' , 'Adj. Volume' ]]
#pd.plotting.scatter_matrix(df)

forecast_out = int(0.008*len(df))
##1% of the data prediction 

df['label'] = df['Adj. Close'].shift(-forecast_out*2)

x_unscale = np.array(df.drop(['label'],1))
x_all = preprocessing.scale(x_unscale)
x_forecast = x_all[-forecast_out*2:-forecast_out]
## data used for prediction
x = x_all[:-forecast_out*2]
## x data use to build model 

y = np.array(df['label'][:-forecast_out*2])


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
clf = LinearRegression(n_jobs = 5)
clf.fit(x_train, y_train)

#with open ('linearregression.pickle','wb') as f:
#    pickle.dump(clf, f)
    
#pickle_in = open('linearregression.pickle','rb')    
#clf = pickle.load(pickle_in)
#
accuracy = clf.score(x_test, y_test)

print ('The accuracy for the model:', accuracy)

y_forecast = clf.predict(x_forecast)

df.loc[-forecast_out:,'Forecast'] = y_forecast

df['Adj. Close'][-forecast_out*10:].plot()
df['label'][-forecast_out*10:].plot()
df['Forecast'][-forecast_out*10:].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


