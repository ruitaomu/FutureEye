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

    dataset = pd.read_csv(
        data_file_name, index_col="Date", parse_dates=["Date"]
    )

    #mpf.plot(dataset, type='candle',mav=(3,6,9), volume=True)
    #print(dataset.head())
    #print(dataset.describe())
    #mpf.add_line(mpf.LineType.VLINE,  xpd=x_position, color='red', linestyle='dashed')

    testset_total = dataset.loc[:,["Open", "Close"]].to_numpy()
    plt.figure(figsize = (18,9))
    plt.plot(dataset.loc[:,"Close"].to_numpy())
    plt.title("Prediction")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend("Close Price")
    plt.xlim(0, len(testset_total))

    return test_X_sequence(testset_total, N_STEPS)

X_test = process_test_file("test.txt")
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], FEATURES)
GRU_predicted_result = model_gru.predict(X_test)
pred_high_points = np.where(GRU_predicted_result>1.0)[0]
pred_low_points = np.where(GRU_predicted_result<-1.0)[0]
print(f"Predicted high points: {pred_high_points}")
print(f"Predicted low points: {pred_low_points}")

for t in pred_high_points:
    plt.axvline(x=t+N_STEPS-FEATURE_OFFSET, linestyle='dashed', color='red')

for t in pred_low_points:
    plt.axvline(x=t+N_STEPS-FEATURE_OFFSET, linestyle='dashed', color='blue')

#plt.xticks(pred_high_points+N_STEPS-FEATURE_OFFSET, GRU_predicted_result[GRU_predicted_result>1.0], rotation=45, fontsize=6)
#plt.xticks(pred_low_points+N_STEPS-FEATURE_OFFSET, GRU_predicted_result[GRU_predicted_result<-1.0], rotation=45, fontsize=6)
plt.show()