#Importing the Libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from yahoofinancials import YahooFinancials
import requests
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")
sns.set_style('whitegrid')
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import plotly.graph_objs as go
import time


st.title('ForeStock')
#Taking User Input

st.sidebar.header('Enter the Data')
with st.sidebar:
    ticker  = st.sidebar.text_input('Enter Stock Ticker:')
    start_viz = st.sidebar.text_input("Enter Start Date of the Data to be fetched (YYYY-MM-DD):")
    end_viz = st.sidebar.text_input("Enter End Date of the Data to be fetched (YYYY-MM-DD):")
    interval_op = ["daily","weekly","monthly","yearly"]
    interval_viz = st.sidebar.selectbox("Select Interval of Data to be fetched:",options=interval_op)
    time_op = ["1Y","3Y","5Y","10Y","15Y","20Y"]
    # time_float = st.sidebar.number_input('Enter the future time period in Days: ')
    time_float = st.sidebar.selectbox('Select the Forecast Period: ',options = time_op)
    with st.form(key = 'Form1'):
        submitted = st.form_submit_button("Enter")

if submitted:
    #Reading the Data
    
    yahoo_financials = YahooFinancials(ticker)
    data = yahoo_financials.get_historical_price_data(start_date='2000-01-01', 
                                                    end_date='2012-12-31', 
                                                    time_interval='daily')
    df = pd.DataFrame(data[ticker]['prices'])
    df1 =df
    df = df.drop('date', axis=1)
    df.rename(columns = {'formatted_date':'Date'}, inplace = True)
    df['Deviation'] = df['close'] - df['open']
    df1['Deviation'] = df1['close'] - df1['open']
    df1 = df1.drop('date', axis=1).set_index('formatted_date')

    #Reading Vizualizations Data
    
    yahoo_financials_viz = YahooFinancials(ticker)
    data_viz = yahoo_financials_viz.get_historical_price_data(start_date=start_viz, 
                                                    end_date=end_viz, 
                                                    time_interval=interval_viz)
    df_viz = pd.DataFrame(data_viz[ticker]['prices'])
    df_viz = df_viz.set_index('formatted_date')
    df_viz['Deviation'] = df_viz['close'] - df_viz['open']
    ma100 = df_viz['close'].rolling(100).mean()
    ma200 = df_viz['close'].rolling(200).mean()

    #Reading NLP Data
    yahoo_financials_sent = YahooFinancials(ticker)
    data_sent = yahoo_financials_sent.get_historical_price_data(start_date='2008-10-02', 
                                                    end_date='2020-02-13', 
                                                    time_interval='daily')
    df_sent = pd.DataFrame(data_sent[ticker]['prices'])
    df_sent = df_sent.drop('date', axis=1)
    df_sent.rename(columns = {'formatted_date':'Date'}, inplace = True)
    df_sent['Deviation'] = df_sent['close'] - df_sent['open']

    #Visualizations
    with st.spinner('Loading.... Please Wait...'):
        st.subheader(f'{ticker} {interval_viz} Hisorical Price Movements between {start_viz} & {end_viz}')
        fig = plt.figure(figsize=(8, 4))
        plt.plot(df_viz['close'])
        plt.plot(ma100)
        plt.plot(ma200,'g')
        plt.xticks([1,100,300,500,700, 900, 1100, 1300, 1500])
        plt.xlabel('Date')
        plt.ylabel('Closing Price ($)')
        plt.legend(['Close','100 Days Moving Average','200 Days Moving Average'])
        st.plotly_chart(fig)


    with st.spinner('Loading.... Please Wait...'):
        st.subheader(f'{ticker} {interval_viz} Gains/Losses between {start_viz} & {end_viz}')
        fig1 = plt.figure(figsize=(6, 4))
        # fig1 = go.Figure(data = [go.Line(df_viz['Deviation'])])
        plt.plot(df_viz['Deviation'])
        plt.xticks([1,100,300,500,700, 900, 1100, 1300, 1500])
        plt.xlabel('Date')
        plt.ylabel('Closing Price Deviations ($)')
        st.plotly_chart(fig1)

    # import chart_studio.plotly as py

    with st.spinner('Loading.... Please Wait...'):
        st.subheader(f'{ticker} {interval_viz} High and Low Movements between {start_viz} & {end_viz}')

        fig2 = go.Figure(data = [go.Candlestick(x=df_viz.index,
                        open=df_viz['open'],
                        high=df_viz['high'],
                        low=df_viz['low'],
                        close=df_viz['close'])])
        # fig2.update_layout(paper_bgcolor="light grey")

        st.plotly_chart(fig2)


    #NLP Model
    with st.spinner('Loading.... Please Wait...'):
        tweets = pd.read_csv("us_equities_news_dataset.csv")
        ticker_list = tweets['ticker'].unique().tolist()

        if ticker_list.count(ticker) == 1:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            def negativityCheck(sentence):
                if(analyzer.polarity_scores(sentence)['compound'] == 0):
                    return 0
                elif(analyzer.polarity_scores(sentence)['compound']>0):
                    return 1
                else:
                    return -1
            sentences = tweets["title"]
            tweets["negative_sentiment"] = pd.DataFrame(sentences.apply(negativityCheck).tolist())
            negativeTweets = tweets.loc[tweets["negative_sentiment"]==-1]
            positiveTweets = tweets.loc[tweets["negative_sentiment"]==1]   

            negativeTweets_Comp = negativeTweets.loc[negativeTweets['ticker'] ==ticker]
            import datetime
            negativeTweets_Comp['Date'] = pd.to_datetime(negativeTweets_Comp['release_date']).dt.strftime('%Y-%m-%d')
            positiveTweets_Comp = positiveTweets.loc[positiveTweets['ticker'] ==ticker]
            positiveTweets_Comp['Date'] = pd.to_datetime(positiveTweets_Comp['release_date']).dt.strftime('%Y-%m-%d')
            influence_neg= pd.merge(negativeTweets_Comp,df_sent,on='Date',how="inner")
            influence_pos= pd.merge(positiveTweets_Comp,df_sent,on='Date',how="inner")
            DateCount_neg= negativeTweets_Comp["Date"].value_counts()
            DateCount_neg = pd.DataFrame(DateCount_neg)
            DateCount_neg["No."] = DateCount_neg.index
            DateCount_neg.rename(columns = {'Date':'Count'}, inplace = True)
            DateCount_neg.rename(columns = {'No.':'Date'}, inplace = True)
            DateCount_pos= positiveTweets_Comp["Date"].value_counts()
            DateCount_pos = pd.DataFrame(DateCount_pos)
            DateCount_pos["No."] = DateCount_pos.index
            DateCount_pos.rename(columns = {'Date':'Count'}, inplace = True)
            DateCount_pos.rename(columns = {'No.':'Date'}, inplace = True)
            InfluenceByNo_neg= pd.merge(DateCount_neg,df_sent,on="Date",how="inner")
            InfluenceByNo_pos= pd.merge(DateCount_pos,df_sent,on="Date",how="inner")
            InfluenceByNo_neg["dateTime"] = pd.to_datetime(InfluenceByNo_neg["Date"])
            InfluenceByNo_neg["Count"] = InfluenceByNo_neg["Count"]/np.median(InfluenceByNo_neg["Count"])
            InfluenceByNo_neg = InfluenceByNo_neg.sort_values(by="dateTime") 
            InfluenceByNo_pos["dateTime"] = pd.to_datetime(InfluenceByNo_pos["Date"])
            InfluenceByNo_pos["Count"] = InfluenceByNo_pos["Count"]/np.median(InfluenceByNo_pos["Count"])
            InfluenceByNo_pos = InfluenceByNo_pos.sort_values(by="dateTime")  
            correlation_neg = InfluenceByNo_neg['Count'].corr(InfluenceByNo_neg['close'])
            correlation_pos = InfluenceByNo_pos['Count'].corr(InfluenceByNo_pos['close'])
            print(f'The Negative news correlation is : {correlation_neg}')
            print(f'The Positive news correlation is : {correlation_pos}')

            def susceptance_pos(correlation):
                if correlation > 0.7:
                    st.info(f'The Price Movements of {ticker} has High susceptance to Positive News')
                elif correlation < 0.3:
                    st.info(f'The Price Movements of {ticker} has Low susceptance to Positive News')
                else:
                    st.info(f'The Price Movements of {ticker} has Medium susceptance to Positive News')

            def susceptance_neg(correlation):
                if correlation > 0.7:
                    st.info(f'The Price Movements of {ticker} has High susceptance to Negative News')
                elif correlation < 0.3:
                    st.info(f'The Price Movements of {ticker} has Low susceptance to Negative News')
                else:
                    st.info(f'The Price Movements of {ticker} has Medium susceptance to Negative News')

            st.subheader(f'Influence of  News on {ticker} Stock Price')
            fig0, ax = plt.subplots(figsize=(6,4))
            #Positive News
            openingValues_pos= InfluenceByNo_pos["open"]
            closingValues_pos = InfluenceByNo_pos["close"]
            countTweet_pos = InfluenceByNo_pos["Count"]
            dates_pos = InfluenceByNo_pos["dateTime"]
            #Negative News
            openingValues_neg= InfluenceByNo_neg["open"]
            closingValues_neg = InfluenceByNo_neg["close"]
            countTweet_neg = InfluenceByNo_neg["Count"]
            dates_neg = InfluenceByNo_neg["dateTime"]

            ax.plot(dates_pos,closingValues_pos,label="close")
            ax.plot(dates_pos,countTweet_pos,label="amount of positive news")
            ax.plot(dates_neg,countTweet_neg,label="amount of negative news")
            plt.xlabel('Year')
            plt.ylabel('Volume of News')
            ax.legend()
            st.plotly_chart(fig0)
            susceptance_pos(correlation_pos)
            susceptance_neg(correlation_neg)
        else:
            st.subheader("No Sentiment Data Found")


    #Model Training
    with st.spinner('Loading.... Please Wait...'):
        close_prices = df['close']
        values = close_prices.values
        training_data_len = math.ceil(len(values)* 0.7)

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(np.reshape(values,(-1,1)))
        train_data = scaled_data[0: training_data_len, :]

        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_val = x_train[0:598]
        y_val = y_train[0:598]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        test_data = scaled_data[training_data_len-60: , : ]
        x_test = []
        y_test = values[training_data_len:]

        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])


        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        #Running the Model
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.layers import Dropout

        model = tf.keras.models.load_model('Final Model.h5')

        # Show the model architecture
        #new_model.summary()

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        from sklearn.metrics import mean_absolute_error,mean_squared_error
        mse = mean_squared_error(y_test,predictions)
        mae = mean_absolute_error(y_true=y_test,y_pred=predictions)
        rmse = np.sqrt(mse)
        print(mse,mae,rmse) 

        #Model Viz
        st.subheader(f'Model Performance for {ticker} Stock Price')
        data_close = df.filter(['close'])
        train = data_close[:training_data_len]
        validation = data_close[training_data_len:]
        validation['Predictions'] = predictions
        fig4 = plt.figure(figsize=(6,4))
        # plt.title('Model')
        plt.xticks([1,100,300,500,700, 900])
        plt.xlabel('Date')
        plt.ylabel('Close Price USD ($)')
        plt.plot(train)
        plt.plot(validation[['close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        st.plotly_chart(fig4)

    #Future Trends
    with st.spinner('Loading.... Please Wait...'):
        val = pd.DataFrame(df1['close'][training_data_len:])
        val['Predicted'] = predictions


        if time_float == '1Y':
            time = 365
        elif time_float == '3Y':
            time  = 1095
        elif time_float == '5Y':
            time = 1825
        elif time_float == '10Y':
            time = 3650
        elif time_float == '15Y':
            time = 5475
        else:
            time = 7300

        val = val.append(pd.DataFrame(columns=val.columns,index=pd.date_range(start=val.index[-1], periods=time+1, freq='D', closed='right')))

        upcoming_prediction = pd.DataFrame(columns=['close'],index=val.index)
        upcoming_prediction.index=pd.to_datetime(upcoming_prediction.index)

        curr_seq = x_test[-1:]
        for i in range(-time,0):
            up_pred = model.predict(curr_seq)
            upcoming_prediction.iloc[i] = up_pred
            curr_seq = np.append(curr_seq[0][1:],up_pred,axis=0)
            curr_seq = curr_seq.reshape(x_test[-1:].shape)

        upcoming_prediction[['close']] = scaler.inverse_transform(upcoming_prediction[['close']])

        #Forecast Visualization
        st.subheader(f'{ticker} {time/365} Years Future Projections')
        fig5 = plt.figure(figsize=(6, 4))
        plt.plot(val['close'].values)
        plt.plot(upcoming_prediction['close'].values)
        # plt.xticks([1,100,300,500,700, 900, 1100, 1300, 1500])
        plt.xlabel('Date')
        plt.ylabel('Closing Price ($)')
        plt.legend(['Close','Future Predicted'])
        st.plotly_chart(fig5)
        

