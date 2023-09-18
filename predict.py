import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import mplfinance as mpf

N_STEPS = 8
FEATURES = 2
FEATURE_OFFSET = 0

model_gru = keras.models.load_model("gru_model.keras")
dataset = pd.DataFrame()

def test_X_sequence(sequence, n_steps):
    X = list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x = sequence[i:end_ix]
        
        seq_x = seq_x.flatten().reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(np.array([np.min(seq_x), np.max(seq_x)]).reshape(-1, 1))
        seq_x = scaler.transform(seq_x)
        seq_x = np.array(seq_x).reshape(-1, 2)
        X.append(seq_x)
    return np.array(X)

def process_test_file(data_file_name):
    global dataset

    dataset = pd.read_csv(
        data_file_name, index_col="Date", parse_dates=["Date"]
    )

    

    testset_total = dataset.loc[:,["Open", "Close"]].to_numpy()
    return test_X_sequence(testset_total, N_STEPS)

X_test = process_test_file("test.txt")
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], FEATURES)
GRU_predicted_result = model_gru.predict(X_test)

print(dataset.head(10))

def make_selling_signal(signals, price):
    signal   = []
    i = 0
    for value in price:
        if i >= N_STEPS and signals[i - N_STEPS] > 1.0:
            signal.append(value)
        else:
            signal.append(np.nan)
        i = i + 1
    return signal

def make_buying_signal(signals, price):
    signal   = []
    i = 0
    for value in price:
        if i >= N_STEPS and signals[i - N_STEPS] < -1.0:
            signal.append(value)
        else:
            signal.append(np.nan)
        i = i + 1
    return signal


selling_signal = make_selling_signal(GRU_predicted_result, dataset['Open'])
buying_signal = make_buying_signal(GRU_predicted_result, dataset['Open'])
apds = [    mpf.make_addplot(selling_signal, type='scatter', markersize=200, marker='^', color='r'),
            mpf.make_addplot(buying_signal, type='scatter', markersize=200, marker='^', color='b')
        ]

mpf.plot(dataset, type='candle',mav=(3,6,9), addplot=apds)
