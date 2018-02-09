import pandas as pd
pd.options.mode.chained_assignment = None
#==============================================================================
# Google "datafram example"
# calling example for specific value:
# print data.loc['20130102':'20130102','Open']
#==============================================================================

import quandl as qd
import math
import numpy as np
#for arrays
from sklearn import preprocessing, cross_validation, svm 
# important mechine learning package
from sklearn.linear_model import LinearRegression 
#for regression process
import matplotlib.pyplot as plt
#plt.plot(np.arange(0, len(x[:,0])), x[:,0])#example
#plt.savefig('test.pdf)
from matplotlib import style
style.use('ggplot')
#for plot style

import datetime, time
import pickle
#to serialize an object; in this case, we use this midule for saving the trained model
data = qd.get("WIKI/COST")

#import pandas_datareader as pdr
#apple = pdr.get_data_yahoo('AAPL')
## alternative for short data

data_Adj = data[["Adj. Open", "Adj. High", "Adj. Low" ,"Adj. Close", "Adj. Volume" ]]
data_Adj.loc[:,"HL_PCT"] = ((data_Adj["Adj. High"]-data_Adj["Adj. Close"])/data_Adj["Adj. Close"] ) * 100.0
data_Adj.loc[:,"PCT_change"] = ((data_Adj["Adj. Close"]-data_Adj["Adj. Open"])/data_Adj["Adj. Open"] ) * 100.0
df = data_Adj[["Adj. Close", "HL_PCT", "PCT_change" , "Adj. Volume" ]]

#df["Adj. Close"].plot()
##df plot function

forecast_col = 'Adj. Close'
#df.fillna('-99999999, inplace = True')
## fill empty data row if ther is any

#regression for forcast
forecast_out = int(math.ceil(0.01*len(df))) 
print (forecast_out)
#shift for forecasting
df['label'] = df[forecast_col].shift(-forecast_out)
# make labe col to shit adj. close up by 0.1*len(df)
df.dropna(inplace=True)
# print df.tail() 
 
#video p4

x_unscale = np.array(df.drop(['label'],1))
y = np.array(df['label'])
x = preprocessing.scale(x_unscale)
#Scaled data has zero mean and unit variance
x_lately = x[-forecast_out:]
#x = x[:-forecast_out]

#plt.plot(np.arange(0, len(x_unscale[:,0])), x_unscale[:,0],np.arange(0, len(y)), y)

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size = 0.2)
clf = LinearRegression(n_jobs = 5)
#n_job = how many jobs at the same time
#other clf = svm.SVR(kernel = 'poly')
#put the algorithm
clf.fit(x_train, y_train)
# video 6pickle_in 
with open ('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)
    
pickle_in = open('linearregression.pickle','rb')    
clf = pickle.load(pickle_in)
#
clf.score(x_test, y_test)
accuracy = clf.score(x_test, y_test)

#print accuracy

#video 5

forecast_set = clf.predict(x_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
# from video : last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    
data_Adj['Adj. Close'][-800:].plot()
#data_Adj['Adj. Close'].plot() compare the predication with the actual price
#data_Adj['Adj. Close'][-forecast_out:].plot()
#df['Forecast'][-forecast_out:].plot()
df['Adj. Close'][-800:].plot()
df['Forecast'][-800:].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.xlabel('Price')
plt.show()


