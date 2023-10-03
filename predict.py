import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import mplfinance as mpf
import config


buy_price_stack = []
EXECUTION_BUYING_ADJ_MA = True
EXECUTION_SELLING_ADJ_MA = True
EXECUTION_BUYING_ADJ_ONLYONCE = False

def test_X_sequence(sequence, n_steps, features):
    X = list()
    dummy = np.full((n_steps, features), np.nan).reshape(-1, features)
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
        seq_x = np.array(seq_x).reshape(-1, features)
        X.append(seq_x)
    return np.array(X)

def process_test_file(settings, if_make_index):
    global dataset

    if if_make_index:
        dataset = pd.read_csv(
        settings["TEST_FILE_NAME"], index_col="Date", parse_dates=["Date"]
        )
    else:    
        dataset = pd.read_csv(
            settings["TEST_FILE_NAME"]
        )

    features = len(settings["FEATURES_SET"])

    testset_total = dataset.loc[:,settings["FEATURES_SET"]].to_numpy()
    return test_X_sequence(testset_total, settings["N_STEPS"], features), dataset

def moving_avg(d, index, num):
    sum = 0
    for n in range(num):
        sum += d['Close'][index-n]
    return sum/num
    
def predict(if_make_index=False):
    global buy_price_stack
    
    settings = config.load_settings()
    buy_price_stack = []
    features = len(settings["FEATURES_SET"])

    #When OFFSET == 0, the predicted results are real happenning. Otherwise, it simulated by ideal situation
    if settings["FLOATING_POINT_ADJUSTMENT"]:
        OFFSET = settings["FEATURE_OFFSET"] - 1
    else:
        OFFSET = 0
    
    
    X_test, dataset = process_test_file(settings, if_make_index)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], features)
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
        signal = execution(predicted_result, dataset, i, settings["N_STEPS"])
        if signal == 0:
            signals_l.append(dataset['Open'][i])
        else:
            signals_l.append(np.nan)

        if signal == 1:
            signals_h.append(dataset['Open'][i])
        else:
            signals_h.append(np.nan)

    return dataset, signals_l, signals_h
    
def execution(predicted_result, dataset, i, n_steps):
    global buy_price_stack
    price = dataset['Open'][i]
    
    signal = -1
    if predicted_result < 0.01:
        if EXECUTION_BUYING_ADJ_MA:
            if dataset['Close'][i-1] >= moving_avg(dataset, i-1, 2) or dataset['Open'][i] >= moving_avg(dataset, i-1, 2):
                return -1
            
        if EXECUTION_BUYING_ADJ_ONLYONCE:
            if not buy_price_stack:
                signal = 0
                buy_price_stack.append(price)
            elif buy_price_stack[-1] > price:
                signal = 0
                buy_price_stack.append(price)
        else:
            signal = 0
    elif predicted_result > 0.99:
        if EXECUTION_SELLING_ADJ_MA:
            if dataset['Close'][i-1] <= moving_avg(dataset, i-1, 2) or dataset['Open'][i] <= moving_avg(dataset, i-1, 2):
                return -1
        if EXECUTION_BUYING_ADJ_ONLYONCE:
            if buy_price_stack:
                while buy_price_stack and buy_price_stack[-1] < price:
                    signal = 1
                    buy_price_stack.pop()
        else:
            signal = 1
            
    return signal 
            
            
        

if __name__ == "__main__":
    _, l, h = predict(True)
    apds = [    
            mpf.make_addplot(l, type='scatter', markersize=100, marker='^', color='b'),
            mpf.make_addplot(h, type='scatter', markersize=100, marker='^', color='r'),
            ]
    mpf.plot(dataset, type='candle',mav=(2,3,6,9), addplot=apds)
