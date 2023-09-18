# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD
from tensorflow.random import set_seed

N_STEPS = 8
FEATURES = 2
FEATURE_OFFSET = 1
IGNORE_SAMPLE_FINISH_WITH_MINMAX = True

train_set_X_list = list()
train_set_y_list = list()
test_set_X_list = list()
test_set_y_list = list()

def add_train_set(sample_seq, value):
    n_seq = len(sample_seq)
    if (n_seq > 0):
        train_set_X_list.extend(sample_seq)
        res = np.full((n_seq,), value)
        train_set_y_list.extend(res.tolist())

def add_test_set(sample_seq, value):
    n_seq = len(sample_seq)
    if (n_seq > 0):
        test_set_X_list.extend(sample_seq)
        res = np.full((n_seq,), value)
        test_set_y_list.extend(res.tolist())

def process_data_file(data_file_name):
    high_indeces = []
    low_indeces = []
    highest_value = 0.0
    lowest_value = 0.0
    dataset = pd.DataFrame()

    set_seed(4550)
    np.random.seed(4551)

    dataset = pd.read_csv(
        data_file_name
    )

    #print(dataset.head())
    #print(dataset.describe())

    #找出给定图中所有的高低点位
    all_high_points = dataset.loc[N_STEPS:len(dataset)-FEATURE_OFFSET-1,"High"].to_numpy()
    all_low_points  = dataset.loc[N_STEPS:len(dataset)-FEATURE_OFFSET-1,"Low"].to_numpy()

    #找出全图最高和最低点的值
    highest_value = np.max(all_high_points)
    lowest_value  = np.min(all_low_points)

    #找出最高和最低点所在的位置，保存在数组中。有可能不止一个最高或最低点
    high_indeces = np.where(all_high_points == highest_value)[0]+N_STEPS
    low_indeces = np.where(all_low_points == lowest_value)[0]+N_STEPS

    #如果几个最高点或最低点紧挨着，只保留连续的第一个位置
    result = []
    for i in range(len(high_indeces)):
        if i == 0 or high_indeces[i] != high_indeces[i-1] + 1:
            result.append(high_indeces[i])

    high_indeces = np.array(result)

    result = []
    for i in range(len(low_indeces)):
        if i == 0 or low_indeces[i] != low_indeces[i-1] + 1:
            result.append(low_indeces[i])

    low_indeces = np.array(result)

    
    #根据找出的所有高低点位出现的时间点，采样在此时间点之前N_STEPS个时段的开盘和收盘价，存入列表
    scaled_high_openclose = list()
    scaled_low_openclose = list()
    unscaled_high_openclose = list()
    unscaled_low_openclose = list()


    for i in range(len(high_indeces)):
        h = dataset.loc[high_indeces[i]-N_STEPS+FEATURE_OFFSET : high_indeces[i]-1+FEATURE_OFFSET, ["Open", "Close"]].to_numpy().flatten().reshape(-1, 1)
        #设定缩放器
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(np.array([np.min(h), np.max(h)]).reshape(-1, 1)) 
        unscaled_high_openclose.append(np.array(h).reshape(-1, 2))
        h = scaler.transform(h) #缩放
        if (not IGNORE_SAMPLE_FINISH_WITH_MINMAX) or (h[N_STEPS*2-1] != 1.0 and h[N_STEPS*2-2] != 1.0):
            #当IGNORE_SAMPLE_FINISH_WITH_MINMAX为True，那么如果最高的开盘价或收盘价出现在这个采样时间段的末尾，则该段采样被忽略
            scaled_high_openclose.append(np.array(h).reshape(-1, 2))

    for i in range(len(low_indeces)):
        l = dataset.loc[low_indeces[i]-N_STEPS+FEATURE_OFFSET : low_indeces[i]-1+FEATURE_OFFSET, ["Open", "Close"]].to_numpy().flatten().reshape(-1, 1)
        #设定缩放器
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(np.array([np.min(l), np.max(l)]).reshape(-1, 1))    
        unscaled_low_openclose.append(np.array(l).reshape(-1, 2))
        l = scaler.transform(l) #缩放
        if (not IGNORE_SAMPLE_FINISH_WITH_MINMAX) or (l[N_STEPS*2-1] != 0.0 and l[N_STEPS*2-2] != 0.0):
            #当IGNORE_SAMPLE_FINISH_WITH_MINMAX为True，那么如果最低的开盘价或收盘价出现在这个采样时间段的末尾，则该段采样被忽略
            scaled_low_openclose.append(np.array(l).reshape(-1, 2))

    #将该图片里的所有高点采样数据加入训练集
    add_train_set(scaled_high_openclose, 1.00)
    #将该图片里的所有高点采样数据加入训练集
    add_train_set(scaled_low_openclose, -1.00)

    def show_sample_list(sample_list, indeces, plt_samples):
        h_arr = np.array(sample_list)
        h_arr = h_arr.reshape(-1, N_STEPS, 2)

        for i in range(len(indeces)):
            arr = h_arr[i]
            arr = np.resize(arr, dataset.shape[0]*2)
            arr = arr.reshape(-1, 2)
            arr[N_STEPS:] = 0
            arr = np.roll(arr, (indeces[i]-N_STEPS)*2)
            arr = arr.reshape(-1, 2)
            plt_samples.plot(arr)
            plt_samples.set_xlim(0, dataset.shape[0]-1)
            plt_samples.set_ylim(lowest_value, highest_value)

    def show_high_low_samples():
        fig, subplt_high_low_samples = plt.subplots()
        show_sample_list(unscaled_high_openclose, high_indeces, subplt_high_low_samples)
        show_sample_list(unscaled_low_openclose, low_indeces, subplt_high_low_samples)

    

    def show_high_low():
        fig, plt_dataset_high_low = plt.subplots()

        all_dataset_high_low = dataset.loc[:,["High","Low"]].to_numpy()
        all_dataset_high_low = all_dataset_high_low.reshape(len(all_dataset_high_low), 2)
        plt_dataset_high_low.plot(all_dataset_high_low)
        plt_dataset_high_low.set_xlim(0, dataset.shape[0]-1)
        plt_dataset_high_low.set_ylim(lowest_value, highest_value)
        plt_dataset_high_low.set_title("Low-High")

        for h in high_indeces:
            plt_dataset_high_low.axvline(x=h, linestyle='dashed', color='red')

        for l in low_indeces:
            plt_dataset_high_low.axvline(x=l, linestyle='dashed', color='blue')

    def show_open_close():
        fig, plt_dataset_open_close = plt.subplots()
        all_dataset_open_close = dataset.loc[:,["Open","Close"]].to_numpy()
        all_dataset_open_close = all_dataset_open_close.reshape(len(all_dataset_open_close), 2)
        plt_dataset_open_close.plot(all_dataset_open_close)
        plt_dataset_open_close.set_xlim(0, dataset.shape[0]-1)
        plt_dataset_open_close.set_ylim(lowest_value, highest_value)
        plt_dataset_open_close.set_title("Open-Close")

    #show_high_low_samples()    
    #show_high_low()
    #show_open_close()
    #plt.show()

file_pattern = "data/highlow-{}.txt"

#训练集的数据文件总数
num_files = 1600

#逐个处理训练集数据文件，将训练集数据加入train_set_X_list和train_set_y_list列表
for i in range(1, num_files + 1):
    file_name = file_pattern.format(i)
    process_data_file(data_file_name = file_name)



# The LSTM architecture
model_lstm = Sequential()
model_lstm.add(LSTM(units=125, activation="tanh", input_shape=(N_STEPS, FEATURES)))
model_lstm.add(Dense(units=1))
# Compiling the model
model_lstm.compile(optimizer="RMSprop", loss="mse")
model_lstm.summary()


model_gru = Sequential()
model_gru.add(GRU(units=125, activation="tanh", input_shape=(N_STEPS, FEATURES)))
model_gru.add(Dense(units=1))
# Compiling the RNN
model_gru.compile(optimizer="RMSprop", loss="mse")
model_gru.summary()

X_train = np.array(train_set_X_list)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], FEATURES)
y_train = np.array(train_set_y_list)
y_train = y_train.reshape(-1, 1)
#model_lstm.fit(X_train, y_train, epochs=200, batch_size=64)
model_gru.fit(X_train, y_train, epochs=200, batch_size=32)


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
        data_file_name
    )

    #print(dataset.head())
    #print(dataset.describe())

    testset_total = dataset.loc[:,["Open", "Close"]].to_numpy()

    plt.plot(testset_total)
    plt.title("Prediction")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.xlim(0, len(testset_total))

    return test_X_sequence(testset_total, N_STEPS)

#用训练好的模型对测试数据集进行预测
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

plt.show()
model_gru.save("gru_model.keras")
print("All done!")