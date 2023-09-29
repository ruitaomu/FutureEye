import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import mplfinance as mpf
import config

settings = config.load_settings()

N_STEPS = settings["N_STEPS"]
FEATURES_SET = settings["FEATURES_SET"]
FEATURES = len(FEATURES_SET)

#When OFFSET == 0, the predicted results are real happenning. Otherwise, it simulated by ideal situation
#OFFSET = 0
OFFSET = settings["FEATURE_OFFSET"] - 1

dataset = pd.DataFrame()

def test_X_sequence(sequence, n_steps):
    X = list()
    dummy = np.full((n_steps, FEATURES), np.nan).reshape(-1, FEATURES)
    for i in range(n_steps):
        X.append(dummy)

    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x = sequence[i:end_ix]
        
        seq_x = seq_x.flatten().reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(np.array([np.min(seq_x), np.max(seq_x)]).reshape(-1, 1))
        seq_x = scaler.transform(seq_x)
        seq_x = np.array(seq_x).reshape(-1, FEATURES)
        X.append(seq_x)
    return np.array(X)

def process_test_file(data_file_name):
    global dataset

    dataset = pd.read_csv(
        data_file_name, index_col="Date", parse_dates=["Date"]
    )

    

    testset_total = dataset.loc[:,FEATURES_SET].to_numpy()
    return test_X_sequence(testset_total, N_STEPS)

X_test = process_test_file(settings["TEST_FILE_NAME"])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], FEATURES)
model = keras.models.load_model(settings["MODEL_FILE_NAME"])

signals_h = []
signals_l = []
for i in range(len(dataset)):
    if np.all(np.isnan(X_test[i])) or i >= len(dataset) - OFFSET:
        signals_l.append(np.nan)
        signals_h.append(np.nan)
        continue

    sample = X_test[i+OFFSET]
    sample = sample.reshape(1,sample.shape[0],sample.shape[1])
    predicted_result = model.predict(sample)
    if predicted_result[0] < 0.01:
        signals_l.append(dataset['Open'][i])
    else:
        signals_l.append(np.nan)

    if predicted_result[0] > 0.99:
        signals_h.append(dataset['Open'][i])
    else:
        signals_h.append(np.nan)

apds = [    
        mpf.make_addplot(signals_l, type='scatter', markersize=100, marker='^', color='b'),
        mpf.make_addplot(signals_h, type='scatter', markersize=100, marker='^', color='r'),
        ]
mpf.plot(dataset, type='candle',mav=(3,6,9), addplot=apds)
