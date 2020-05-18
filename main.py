from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow_core.python.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, BatchNormalization
import pickle

RATIOS_TO_PREDICT = [(1, 'future_1day'), (2, 'future_2days'), (3, 'future_3days'), (4, 'future_4days'), (5,'future_5days'), (10, 'future_10days'), (15,'future_15days')]
BATCH_SIZE = 106
LOOKBACK = 1440
DROPOUT = 0.2

#Main loop executing all 28 models
for day_number, ratio in RATIOS_TO_PREDICT:
    btc_usd_df = pd.read_csv("pre_processed_btc_usd.csv")
    btc_usd_df.set_index('Date_time')
    btc_usd_df.drop(columns=['Date_time'], inplace=True)

    ratios_to_remove = ['future_1day', 'future_2days', 'future_3days', 'future_4days', 'future_5days', 'future_10days', 'future_15days']
    ratios_to_remove.remove(ratio)

    #Drop all ratios we are not predicting
    btc_usd_df.drop(columns=ratios_to_remove, inplace=True)

    #Create a classify function for the mapper, that outputs 1 if future price > current price,
    #else 0.
    def classify(current, future):
        if float(future) > float(current):
            return 1
        elif float(future) <= float(current):
            return 0

    btc_usd_df['target'] = list(map(classify, btc_usd_df['Close'], btc_usd_df[ratio]))
    btc_usd_df.drop(columns=[ratio], inplace=True)

    y  = btc_usd_df['target'].values
    x = btc_usd_df.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']].values

    stdsc = StandardScaler()
    x_scaled = stdsc.fit_transform(x)

    #Use Keras TimeSeriesGenerator to generate the time series data
    train_gen = TimeseriesGenerator(x_scaled, y, length=LOOKBACK, batch_size=BATCH_SIZE, start_index=0, end_index=80136, shuffle=True)
    test_gen = TimeseriesGenerator(x_scaled, y, length=LOOKBACK, batch_size=BATCH_SIZE, start_index=80137, end_index=None)


    #CHOLLET SIMPLERNN MODEL
    chollet_RNN = Sequential()
    chollet_RNN.add(SimpleRNN(32,
                        dropout=DROPOUT,
                        recurrent_dropout=DROPOUT,
                        return_sequences=True,
                        input_shape=(None, 6)))
    chollet_RNN.add(SimpleRNN(64,
                        dropout=DROPOUT,
                        recurrent_dropout=DROPOUT))
    chollet_RNN.add(Dense(1, activation='sigmoid'))
    chollet_RNN.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    history_chollet_RNN = chollet_RNN.fit_generator(train_gen,
                                        epochs=100,
                                        validation_data=test_gen,
                                        verbose=0)
    #Write history object to pickle, for plotting later
    with open(f'history_objects/{ratio}_chollet_RNN', 'wb') as f:
        pickle.dump(history_chollet_RNN.history, f)
    print(f'{ratio}_chollet_RNN model finished, history object succesfully written to pickle.')


    # CHOLLET GRU MODEL
    chollet_GRU = Sequential()
    chollet_GRU.add(GRU(32,
                        dropout=DROPOUT,
                        recurrent_dropout=DROPOUT,
                        return_sequences=True,
                        input_shape=(None, 6)))
    chollet_GRU.add(GRU(64,
                        dropout=DROPOUT,
                        recurrent_dropout=DROPOUT))
    chollet_GRU.add(Dense(1, activation='sigmoid'))
    chollet_GRU.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    history_chollet_GRU = chollet_GRU.fit_generator(train_gen,
                                                    epochs=100,
                                                    validation_data=test_gen,
                                                    verbose=0)
    #Write history object to pickle, for plotting later
    with open(f'history_objects/{ratio}_chollet_GRU', 'wb') as f2:
        pickle.dump(history_chollet_GRU.history, f2)
    print(f'{ratio}_chollet_GRU model finished, history object succesfully written to pickle. ')


    # CHOLLET_LSTM MODEL
    chollet_LSTM = Sequential()
    chollet_LSTM.add(LSTM(32,
                          dropout=DROPOUT,
                          recurrent_dropout=DROPOUT,
                          return_sequences=True,
                          input_shape=(None, 6)))
    chollet_LSTM.add(LSTM(64,
                          dropout=DROPOUT,
                          recurrent_dropout=DROPOUT))
    chollet_LSTM.add(Dense(1, activation='sigmoid'))
    chollet_LSTM.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    history_chollet_LSTM = chollet_LSTM.fit_generator(train_gen,
                                                      epochs=100,
                                                      validation_data=test_gen,
                                                      verbose=0)
    #Write history object to pickle, for plotting later
    with open(f'history_objects/{ratio}_chollet_LSTM', 'wb') as f3:
        pickle.dump(history_chollet_LSTM.history, f3)
    print(f'{ratio}_chollet_LSTM model finished, history object successfully written to pickle.')


    # SENTDEX MODEL
    sentdex_LSTM = Sequential()
    sentdex_LSTM.add(LSTM(128,
                          dropout=DROPOUT,
                          recurrent_dropout=DROPOUT,
                          return_sequences=True,
                          input_shape=(None, 6)))
    sentdex_LSTM.add(BatchNormalization())
    sentdex_LSTM.add(LSTM(128,
                          dropout=DROPOUT,
                          recurrent_dropout=DROPOUT,
                          return_sequences=True))
    sentdex_LSTM.add(BatchNormalization())
    sentdex_LSTM.add(LSTM(128,
                          dropout=DROPOUT,
                          recurrent_dropout=DROPOUT))
    sentdex_LSTM.add(BatchNormalization())
    sentdex_LSTM.add(Dense(32, activation='relu'))
    sentdex_LSTM.add(Dropout(DROPOUT))
    sentdex_LSTM.add(Dense(1, activation='sigmoid'))
    sentdex_LSTM.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    history_sentdex_LSTM = sentdex_LSTM.fit_generator(train_gen,
                                                      epochs=100,
                                                      validation_data=test_gen,
                                                      verbose=0)
    #Write history object to pickle, for plotting later
    with open(f'history_objects/{ratio}_sentdex_LSTM', 'wb') as f4:
        pickle.dump(history_sentdex_LSTM.history, f4)
    print(f'{ratio}_sentdex_LSTM model finished, history object successfully written to pickle.')


