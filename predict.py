import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler
import mplfinance as mpf
import config


buy_price_stack = []

THRESHOLD_H = 0.99999
THRESHOLD_L = 0.00001

def test_X_sequence(sequence, n_steps, features):
    X = list()
    dummy = np.full((n_steps, features), np.nan).reshape(-1, features)
    for i in range(config.N_SCOPE):
        X.append(dummy)

    for i in range(config.N_SCOPE-n_steps, len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x = sequence[i:end_ix]
        
        seq_x = seq_x.flatten().reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0,1))
        scope = sequence[i-(config.N_SCOPE-n_steps):end_ix]
        scaler.fit(np.array([np.min(scope), np.max(scope)]).reshape(-1, 1))
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
    
def predict(settings, if_make_index=False, use_model_name=None):
    global buy_price_stack
    
    buy_price_stack = []
    features = len(settings["FEATURES_SET"])
    
    #When OFFSET == 0, the predicted results are real happenning. Otherwise, it simulated by ideal situation
    if settings["FLOATING_POINT_ADJUSTMENT"]:
        OFFSET = settings["FEATURE_OFFSET"] - 1
    else:
        OFFSET = 0
    
    
    X_test, dataset = process_test_file(settings, if_make_index)
    X_test = X_test.reshape(X_test.shape[0], settings["N_STEPS"], features)
    if use_model_name == None:
        model = keras.models.load_model(settings["MODEL_FILE_NAME"])
    else:
        model = keras.models.load_model(use_model_name)

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
        execution_price = execution(settings, predicted_result, dataset, i, settings["N_STEPS"])
        if execution_price < 0:
            signals_l.append(-execution_price)
        else:
            signals_l.append(np.nan)

        if execution_price > 0:
            signals_h.append(execution_price)
        else:
            signals_h.append(np.nan)

    return dataset, signals_l, signals_h
    
last_predicted_result = 0.0
def execution(settings, predicted_result, dataset, i, n_steps):
    global buy_price_stack, last_predicted_result
    
    EXECUTION_BUYING_ADJ_MA = settings["EXECUTION_BUYING_ADJ_MA"]
    EXECUTION_SELLING_ADJ_MA = settings["EXECUTION_SELLING_ADJ_MA"]
    EXECUTION_BUYING_ADJ_ONLYONCE = settings["EXECUTION_BUYING_ADJ_ONLYONCE"]
    EXECUTION_MUST_SEE_CONFIRM = settings["EXECUTION_MUST_SEE_CONFIRM"]
    MA_ADJ_VALUE = settings["MA_ADJ_VALUE"]
    
    price = open_price = dataset['Open'][i]
    last_close = dataset['Close'][i-1]
    last_ma = moving_avg(dataset, i-1, MA_ADJ_VALUE)
    
    if EXECUTION_MUST_SEE_CONFIRM:
        if last_predicted_result < THRESHOLD_L and last_predicted_result > 0 and dataset['Close'][i-1] > dataset['Open'][i-1]:
            last_predicted_result = 0.0
            return -price
        
        if last_predicted_result > THRESHOLD_H and dataset['Close'][i-1] < dataset['Open'][i-1]:
            last_predicted_result = 0.0
            return price
        
    if predicted_result < THRESHOLD_L:
        if EXECUTION_BUYING_ADJ_MA:
            if last_close >= last_ma or open_price >= last_ma:
                return 0
            
        if EXECUTION_BUYING_ADJ_ONLYONCE:
            if not buy_price_stack:
                buy_price_stack.append(price)
            elif buy_price_stack[-1] > price:
                buy_price_stack.append(price)
            else:
                last_predicted_result = 0.0
                return 0
            
        last_predicted_result = predicted_result
        if EXECUTION_MUST_SEE_CONFIRM:
            return 0
        else:
            return -price
        
    elif predicted_result > THRESHOLD_H:
        if EXECUTION_SELLING_ADJ_MA:
            if last_close <= last_ma or open_price <= last_ma:
                return 0
            
        if EXECUTION_BUYING_ADJ_ONLYONCE:
            if buy_price_stack:
                confirm = False
                while buy_price_stack and buy_price_stack[-1] < price:
                    confirm = True
                    buy_price_stack.pop()
                if not confirm:
                    last_predicted_result = 0.0
                    return 0
            else:
                last_predicted_result = 0.0
                return 0
        
        last_predicted_result = predicted_result
        
        if EXECUTION_MUST_SEE_CONFIRM:
            return 0
        else:
            return price
    else:
        return 0 
            
            
        

if __name__ == "__main__":    
    settings = config.load_settings()
    _, l, h = predict(settings, True)
    apds = [    
            mpf.make_addplot(l, type='scatter', markersize=100, marker='^', color='b'),
            mpf.make_addplot(h, type='scatter', markersize=100, marker='^', color='r'),
            ]
    mpf.plot(dataset, type='candle',mav=(2), addplot=apds)
