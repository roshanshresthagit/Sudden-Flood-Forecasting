#!/usr/bin/env python
# coding: utf-8

# In[6]:


#importing libries
import pandas as pd
import requests
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error as mse
from datetime import timedelta, datetime


def maikholaForecasting():

    #token
    print('Getting token')
    auth={"username":"","password":""}
    payload=json.dumps(auth)
    
    try:
        response = requests.post("https://daq.wscada.net/auth/login", headers={'content-type':'application/json'},data=payload)
        if response.status_code == 200:
            bearer_token = response.json().get('token')
        else:
            print(f'failed to obtain token. Status code: {response.json()}')
    except requests.exceptions.RequestException as e:
        print(f'An error occured:{e}')
    
    date_to=datetime.now()
    #api_data 
    print('Getting API')
    date_to_end= date_to.strftime('%Y-%m-%dT%H:00:00')
    MaiKhola_api=f'https://daq.wscada.net/api/analysis/more?stations=220&parameters=158&date_from=2021-09-06T17:00:00&date_to={date_to_end}'
    MaiBeni_api=f'https://daq.wscada.net/api/analysis/more?stations=219&parameters=175&date_from=2021-09-06T17:00:00&date_to={date_to_end}'
    Nayabazar_api =f'https://daq.wscada.net/api/analysis/more?stations=217&parameters=175&date_from=2021-09-06T17:00:00&date_to={date_to_end}'
    Pashupatinagar_api=f'https://daq.wscada.net/api/analysis/more?stations=218&parameters=175&date_from=2021-09-06T17:00:00&date_to={date_to_end}'
    SandakpurValley_api=f'https://daq.wscada.net/api/analysis/more?stations=213&parameters=175&date_from=2021-09-06T17:00:00&date_to={date_to_end}'
    
    headers = {'Authorization': f'Bearer {bearer_token}'}

    response1 = requests.get(MaiKhola_api, headers=headers)
    response2 = requests.get(MaiBeni_api, headers=headers)
    response3 = requests.get(Nayabazar_api, headers=headers)
    response4 = requests.get(Pashupatinagar_api, headers=headers)
    response5 = requests.get(SandakpurValley_api, headers=headers)
    
    print('response of Mai Khola api Data')
    if response1.status_code == 200:
        data1 = response1.json() 
        df1 = pd.DataFrame(data1)
    else:
        print(f"Failed to retrieve data. Status code: {response1.status_code}")
    
    print('response Data of Mai Beni api data')
    if response2.status_code == 200:
        data2 = response2.json() 
        df2 = pd.DataFrame(data2)
    else:
        print(f"Failed to retrieve data. Status code: {response2.status_code}")

    print('response of NayaBazar api data')
    if response3.status_code == 200:
        data3 = response3.json() 
        df3 = pd.DataFrame(data3)
    else:
        print(f"Failed to retrieve data. Status code: {response3.status_code}")

    
    print('response of Pashupatinagar api data')
    if response4.status_code == 200:
        data4 = response4.json() 
        df4 = pd.DataFrame(data4)
    else:
        print(f"Failed to retrieve data. Status code: {response4.status_code}")

    
    print('response of Sandakpur valley api data')
    if response5.status_code == 200:
        data5 = response5.json() 
        df5 = pd.DataFrame(data5)
    else:
        print(f"Failed to retrieve data. Status code: {response5.status_code}")

    print('data of each staion')    
    maikhola_data=df1.at[0,'parameters'][0]['data']
    maibeni_data=df2.at[0,'parameters'][0]['data']
    nayabazar_data=df3.at[0,'parameters'][0]['data']
    pashupatinagar_data=df4.at[0,'parameters'][0]['data']
    sandakpur_data=df5.at[0,'parameters'][0]['data']

    print('mai khola data')
    Mai_khola=pd.DataFrame(maikhola_data)
    Mai_khola['time']=pd.to_datetime(Mai_khola['time'])
    Mai_khola.set_index('time',inplace=True)
    Mai_khola_data=Mai_khola.resample('H').mean()
    Mai_khola_data=Mai_khola_data.rename(columns={'value':'Mai_Khola_Waterlevel'})

    print('mai beni data')
    Maibeni=pd.DataFrame(maibeni_data)
    Maibeni['time']=pd.to_datetime(Maibeni['time'])
    Maibeni.set_index('time',inplace=True)
    Mai_Beni_data=Maibeni.resample('H').sum()
    Mai_Beni_data=Mai_Beni_data.rename(columns={'value':'Mai_Beni_Rainfall'})

    print('nayabazar data')
    Naya_bazar=pd.DataFrame(nayabazar_data)
    Naya_bazar['time']=pd.to_datetime(Naya_bazar['time'])
    Naya_bazar.set_index('time',inplace=True)
    Naya_bazar_data=Naya_bazar.resample('H').sum()
    Naya_bazar_data=Naya_bazar_data.rename(columns={'value':'Nayabazar_Rainfall'})

    print('pashupatinagar data')
    Pashupatinagar=pd.DataFrame(pashupatinagar_data)
    Pashupatinagar['time']=pd.to_datetime(Pashupatinagar['time'])
    Pashupatinagar.set_index('time',inplace=True)
    Pashupatinagar_data=Pashupatinagar.resample('H').sum()
    Pashupatinagar_data=Pashupatinagar_data.rename(columns={'value':'Pashupatinagar_Rainfall'})

    print('sandakpur data')
    Sandakpur=pd.DataFrame(sandakpur_data)
    Sandakpur['time']=pd.to_datetime(Sandakpur['time'])
    Sandakpur.set_index('time',inplace=True)
    Sandakpur_data=Sandakpur.resample('H').sum()
    Sandakpur_data=Sandakpur_data.rename(columns={'value':'Sandakpur_Valley_Rainfall'})
    
    #combined data
    data=pd.concat([Mai_khola_data, Mai_Beni_data, Naya_bazar_data, Pashupatinagar_data, Sandakpur_data],axis=1)

    #data cleaning
    import numpy as np
    data['Mai_Khola_Waterlevel']=data['Mai_Khola_Waterlevel'].apply(lambda x: x if x>=0 else np.nan)
    data['Pashupatinagar_Rainfall']=data['Pashupatinagar_Rainfall'].fillna(0)
    data['Mai_Khola_Waterlevel']=data['Mai_Khola_Waterlevel'].fillna(method='ffill')

    dataset=data.copy()

    #windowing data
    def df_to_X_y2(df, window_size=4):
        print('windowing data')
        df_as_np = df.to_numpy()
        X = []
        y = []
        for i in range(len(df_as_np)-window_size):
            row = [r for r in df_as_np[i:i+window_size]]
            X.append(row)
            label = df_as_np[i+window_size][0] 
            y.append(label)
        return np.array(X), np.array(y)

    X2,y2=df_to_X_y2(dataset)

    print('spliting data for training testing and validation')
    n=len(dataset)
    X2_train,y2_train=X2[0:int(n*0.8)],y2[0:int(n*0.8)]
    X2_val,y2_val=X2[int(n*0.8):],y2[int(n*0.8):]
    X2_train.shape,y2_train.shape,X2_val.shape,y2_val.shape

    print('DATA PREPROCESSING OR SCALING ONLY THE WATERLEVEL DATA')
    waterlevel_training_mean=np.mean(dataset.iloc[:,0])
    waterlevel_training_std=np.std(dataset.iloc[:,0])

    def preprocess(X):
        X[:,:,0]=(X[:,:,0]-waterlevel_training_mean/waterlevel_training_std)
        return X

    preprocess(X2_train)
    preprocess(X2_val)

    print('MODEL ARCHITECTURE')
    model=Sequential()
    model.add(InputLayer((4,5)))
    model.add(LSTM(64,return_sequences=True))
    model.add(LSTM(32,return_sequences=False))
    model.add(Dense(8,'relu'))
    model.add(Dense(1,'linear'))
    model.summary()    

    #SAVING THE MODEL 
    cp=ModelCheckpoint('finalmodel/',save_best_only=True)
    model.compile(loss=MeanSquaredError(),optimizer=Adam(learning_rate=0.001),metrics=[RootMeanSquaredError()])

    #MODEL TRAINNING
    history=model.fit(X2_train,y2_train,validation_data=(X2_val,y2_val),epochs=10,callbacks=[cp])

    print('forecasting 3 hours looking 4 hours into past')
    def forecast_future(model, input_sequence, forecast_steps=1, window_size=4):
        forecasted_values = []
        for _ in range(forecast_steps):
            prediction = model.predict(np.expand_dims(input_sequence, axis=0))[0][0]
            forecasted_values.append(prediction)
            input_sequence = np.concatenate((input_sequence[1:], np.array([[(prediction-waterlevel_training_mean/waterlevel_training_std), 0, 0, 0, 0]])))
        return forecasted_values

    print('last sequence in X2_test as the starting point for forecasting')
    input_sequence = X2_val[-1]
    forecasted_values = forecast_future(model, input_sequence, forecast_steps=3)

    print('datetime of predicated value')
    maxindex=data.index.max()
    next_index = maxindex + pd.DateOffset(hours=1)
    new_index = pd.date_range(start=next_index, periods=3, freq='H')
    new=pd.DataFrame({'time':new_index,'waterlevel':forecasted_values})
    new['time']=pd.to_datetime(new['time']).dt.strftime('%Y-%m-%dT%H:%M:%S')

    print('posting data into comparision through api')
    forecast_api='http://forecast.wscada.net/import/'

    payload=[]
    for index, row in new.iterrows():
        payload.append({
            "time":row["time"],
            "origin_code":"500232",    
            "parameter_code":"ML_WL_F",
            "value":row['waterlevel']
        })
    try:
        response = requests.post(forecast_api, json=payload)
        if response.status_code==200:
            print(f'Successfully posted data for index {index}: {payload}')
        else:
            print(f'Failed to post data for index {index}: {payload}')
    except Exception as e:
        print(f'Error posting data for index {index}: {e}')
            
    
        


# In[7]:


maikholaForecasting()


# In[ ]:




