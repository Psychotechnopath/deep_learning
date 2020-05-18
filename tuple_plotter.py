import pickle
import os
BEST_chollet_RNN_RESULTS = []
BEST_chollet_GRU_RESULTS = []
BEST_chollet_LSTM_RESULTS = []
BEST_sentdex_LSTM_RESULTS = []

#Read in all results, append the validation accuracy of the last epoch (which corresponds to validation accuracy of the model)
for i in os.listdir('history_objects')[1:]:
    if 'chollet_RNN' in i:
        with open(f'history_objects/{i}', 'rb') as f_RNN:
            result_chollet_rnn = pickle.load(f_RNN)
            BEST_chollet_RNN_RESULTS.append(result_chollet_rnn['val_accuracy'][99])
    elif 'chollet_GRU' in i:
        with open(f'history_objects/{i}', 'rb') as f_GRU:
            result_chollet_gru = pickle.load(f_GRU)
            BEST_chollet_GRU_RESULTS.append(result_chollet_gru['val_accuracy'][99])
    elif 'chollet_LSTM' in i:
        with open(f'history_objects/{i}', 'rb') as f_LSTM:
            result_chollet_lstm = pickle.load(f_LSTM)
            BEST_chollet_LSTM_RESULTS.append(result_chollet_lstm['val_accuracy'][99])
    else:
        with open(f'history_objects/{i}', 'rb') as f_LSTM2:
            result_sentdex_lstm = pickle.load(f_LSTM2)
            BEST_sentdex_LSTM_RESULTS.append(result_sentdex_lstm['val_accuracy'][99])


#Results are added to list in this order in previous for-loop.
tup = (10, 15, 1, 2, 3, 4, 5)
BEST_chollet_RNN_RESULTS = sorted(list(zip(tup, BEST_chollet_RNN_RESULTS)), key=lambda x: x[0])
BEST_chollet_GRU_RESULTS = sorted(list(zip(tup, BEST_chollet_GRU_RESULTS)), key=lambda x: x[0])
BEST_chollet_LSTM_RESULTS = sorted(list(zip(tup, BEST_chollet_LSTM_RESULTS)), key=lambda x:x[0])
BEST_sentdex_LSTM_RESULTS = sorted(list(zip(tup, BEST_sentdex_LSTM_RESULTS)), key=lambda x:x[0])

#Make the actual plot
from matplotlib import pyplot as plt
plt.figure()
plt.plot(*zip(*BEST_chollet_RNN_RESULTS), '-o', label='RNN Chollet')
plt.plot(*zip(*BEST_chollet_GRU_RESULTS), '-o', label='GRU Chollet')
plt.plot(*zip(*BEST_chollet_LSTM_RESULTS), '-o', label='LSTM Chollet')
plt.plot(*zip(*BEST_sentdex_LSTM_RESULTS), '-o', label='LSTM Sentdex')
plt.title('Testing accuracy of different NN configurations on different time intervals')
plt.ylabel("Test accuracy")
plt.xlabel("Prediction interval (days)")
plt.legend()
plt.show()


