import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, cross_decomposition, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import time
from datetime import datetime,timedelta
from SmartApi import SmartConnect 
import pyotp
from logzero import logger
from SearchScrip import getNSEScripToken    
import plotly.graph_objects as go
import pandas as pd
import config
style.use('ggplot')

api_key = config.api_key
username = config.username
pwd = config.password
smartApi = SmartConnect(api_key)
try:
    token = config.token
    totp = pyotp.TOTP(token).now()
except Exception as e:
    logger.error("Invalid Token: The provided token is not valid.")
    raise e

# correlation_id = "abcde"
data = smartApi.generateSession(username, pwd, totp)

if data['status'] == False:
    logger.error(data)

else:
    # login api call
    # logger.info(f"You Credentials: {data}")
    authToken = data['data']['jwtToken']
    refreshToken = data['data']['refreshToken']
    # fetch the feedtoken
    feedToken = smartApi.getfeedToken()
    # fetch User Profile
    res = smartApi.getProfile(refreshToken)
    smartApi.generateToken(refreshToken)
    res = res['data']['exchanges']
    # if res:
    #     print("Login Success")
    # print(res)

#Historic api
    try:
        historicParam={
        "exchange": "NSE",
        "symboltoken": getNSEScripToken("RELIANCE"),
        "interval": "ONE_DAY",
        "fromdate": "2019-01-01 00:00",
        "todate": "2024-04-19 00:00"
        }
        res=smartApi.getCandleData(historicParam)['data']
        # print(res)
        # df=pd.DataFrame.from_dict(res,columns=['date','open','high','low','close','volumn'])

        df = pd.DataFrame.from_dict(res)
        df.columns = ['Date','open','high','low','close','volumn']
        
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df.set_index(df["Date"], inplace=True)
        
        print(df)


        fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                             open=df['open'],
                                             high=df['high'],
                                             low=df['low'],
                                             close=df['close'],
                                             )])



        fig.update_layout(xaxis_rangeslider_visible=False)
        # fig.update_xaxes(type="category")
        # fig.show()

    except Exception as e:
        logger.exception(f"Historic Api failed: {e}")
    # #logout
#     # try:
#     #     logout=smartApi.terminateSession('Your Client Id')
#     #     logger.info("Logout Successfull")
#     # except Exception as e:
#     #     logger.exception(f"Logout failed: {e}")


df["HL_PCT"] = (df["high"]-df["close"])/df["close"] * 100




df["PCT_Change"] = (df["close"]-df["open"])/df["open"] * 100



df = df.loc[:,["close","HL_PCT",'PCT_Change',"volumn"]]
print("print line 129")
print(df)

forecast_col = "close"
df.fillna(-9999,inplace= True)


forecast_out = int(math.ceil(0.01*len(df)))
print(f"Forecast : {forecast_out}")
df["label"] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)



X=np.array(df.drop(["label"],axis=1))
y = np.array(df["label"])
x = preprocessing.scale(X)
print(len(X),len(y))
X_lately = X[-forecast_out:]




X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3) 




clf = LinearRegression().fit(X_train,y_train)




accuracy = clf.score(X_test,y_test)
print("Accuracy of Model: ",accuracy)





forecast_set = clf.predict(X_lately)



df["Forecast"] = np.NaN
# df.index = df["Date"]
# df.set_index('Date', inplace=True)
# print(df.index)
last_date = df.iloc[-1].name
print(f"Last date: {last_date}")
last_unix = last_date
one_day = 1
next_day = last_unix + timedelta(days=1)


forecast_set = forecast_set
forecast_set


for i in forecast_set:
    # next_date = datetime.fromtimestamp(next_day)
    next_day = next_day + timedelta(days=1)
    df.loc[next_day] = [np.NaN for _ in range(len(df.columns)-1)] + [i]



df["close"].plot()
df["Forecast"].plot()
plt.plot(df["close"],df["Forecast"])
plt.legend(loc=4)
plt.xlabel ("Date")
plt.ylabel("Price")

print(df.tail(10))
plt.show()




