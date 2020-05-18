import pandas as pd

#Specify the ratio we want to predict
#(Choose from: 'future_1day','future_2days', 'future_3days',  'future_4days', 'future_5days', 'future_10days', 'future_15days')

#Open historical bitcoin data from BitStamp (World's longest standing crypto-exchange).
#This dataset contains historical data from 1-1-2012 to 22-4-2020.
btc_usd_df = pd.read_csv("C:/Users/Yme/Desktop/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv")

#Check for missings. There are a lot!
for col in btc_usd_df.columns:
    print(f"Col name: {col}, nr missing: {btc_usd_df[col].isnull().sum()}")

#The kaggle link where the data was obtained from
#(https://www.kaggle.com/mczielinski/bitcoin-historical-data?select=bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv)
#Explains that timestamps without any trades or activity have their data fields filled with NaNs.
#However, since we will only be using one time point every 30 minutes,
#And this data is on a 1-minute time interval it won't be too much of an issue, so we replace the missings with the ffill strategy
btc_usd_df.fillna(method='ffill', inplace=True)

#Create Date_time objects for every Timestamp in the dataset
btc_usd_df['Date_time'] = pd.to_datetime(btc_usd_df['Timestamp'], unit='s')
#Set index to date_time
btc_usd_df.set_index(btc_usd_df['Date_time'], inplace=True)

#Since the paper I am using took 5 years of historical data to train/test their network, I will do the same.
#The time interval I chose for this purpose is from February 2015 - February 2020 plus 15 extra days
#To calculate future prices.
btc_usd_df = btc_usd_df.iloc[1617376:4310176, :]

#The paper uses time-intervals of 30 minutes. This dataset contains data with every minute. Therefore,
#We use the date-time column to filter out the observations with whole hours (0 minutes) and half hours (30 minutes)
#We avoid looping over the dataframe because this is very expensive. We use a list-comprehension and assign the list
#to a minutes column, which we use to filter the observations.
whole_half_hour = [0,30]
minutes = [x.minute for x in btc_usd_df['Date_time']]
btc_usd_df['minutes'] = minutes
btc_usd_df = btc_usd_df.loc[btc_usd_df["minutes"].isin(whole_half_hour)]

#Drop unneccesary columns
btc_usd_df.drop(columns=['Timestamp', 'minutes', 'Date_time', 'Weighted_Price'], inplace=True)

#%%
#The original paper uses close prices of the previous day to compare whether or not a stock went up or down.
#Since bitcoin never closes, we will simply use the "current price" (e.g. the price on the current time-step)
#To predict whether or not the price will be higher or lower at the time-interval we are predicting for.
#To let the neural network learn these patterns, we need a "future" column. We obtain this column by shifting up the current prices
#To match the "current" data with the "future" price for out network to learn.

#There are two timesteps in one hour, so 1 day (24 hour) means 48 timesteps in the future
btc_usd_df['future_1day'] = btc_usd_df['Close'].shift(-48)

#There are two timesteps in one hour, so 2 days (48 hours) means 96 timesteps in the future
btc_usd_df['future_2days'] = btc_usd_df['Close'].shift(-96)

#There are two timesteps in one hour, so 3 days (72 hours) means 144 timesteps in the future
btc_usd_df['future_3days'] = btc_usd_df['Close'].shift(-144)

#There are two timesteps in one hour, so 4 dayd (96 hours) means 192 timesteps in the future
btc_usd_df['future_4days'] = btc_usd_df['Close'].shift(-192)

#There are two timesteps in one hour, so 5 days (120 hours) means 240 timesteps in the future
btc_usd_df['future_5days'] = btc_usd_df['Close'].shift(-240)

#There are two timesteps in one hour, so 10 days (240 hours) means 480 timesteps in the future.
btc_usd_df['future_10days'] = btc_usd_df['Close'].shift(-480)

#There are two timesteps in one hour, so 15 days (360 hours) means 720 timesteps in the future.
btc_usd_df['future_15days'] = btc_usd_df['Close'].shift(-720)

#Drop rows from march (720) from March that were used to generate "future" prices
btc_usd_df = btc_usd_df.iloc[:-720 ,:]


