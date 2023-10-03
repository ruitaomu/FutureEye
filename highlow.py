# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config as cfg
import os
import predict as predict
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, SimpleRNN
from tensorflow.keras.optimizers import SGD
from tensorflow.random import set_seed

settings = cfg.DEFAULT_SETTINGS

IGNORE_SAMPLE_FINISH_WITH_MINMAX = True

train_set_X_list = list()
train_set_y_list = list()
test_set_X_list = list()
test_set_y_list = list()

n_top_samples = 0
n_bottom_samples = 0

def add_train_set(sample_seq, value):
    global n_top_samples, n_bottom_samples
    n_seq = len(sample_seq)
    if (n_seq > 0):
        if value > 0:
            n_top_samples += n_seq
        else:
            n_bottom_samples += n_seq
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
    dataset = pd.DataFrame()
    N_STEPS = settings["N_STEPS"]
    FEATURE_OFFSET = settings["FEATURE_OFFSET"]
    FEATURES = len(settings["FEATURES_SET"])

    set_seed(4550)
    np.random.seed(4551)

    dataset = pd.read_csv(
        data_file_name, dtype={"Top": int, "Bottom": int, "High": float, "Low": float, "Open": float, "Close": float}
    )

    #print(dataset.head())
    #print(dataset.describe())

    global IGNORE_SAMPLE_FINISH_WITH_MINMAX
    AUTO_MARK = settings["AUTO_MARK"]
    if AUTO_MARK == False and 'Top' in dataset.columns and 'Bottom' in dataset.columns:
        AUTO_MARK = False
    else:
        AUTO_MARK = True

    if AUTO_MARK == True:        
        #找出给定图中所有的高低点位
        #all_high_points = dataset.loc[N_STEPS:len(dataset)-FEATURE_OFFSET-1,"High"].to_numpy()
        #all_low_points  = dataset.loc[N_STEPS:len(dataset)-FEATURE_OFFSET-1,"Low"].to_numpy()
        all_close_points  = dataset.loc[N_STEPS:len(dataset)-FEATURE_OFFSET-1,"Close"].to_numpy()

        #找出全图最高和最低点的值
        #highest_value = np.max(all_high_points)
        #lowest_value  = np.min(all_low_points)
        highest_value = np.max(all_close_points)
        lowest_value  = np.min(all_close_points)

        #找出最高和最低点所在的位置，保存在数组中。有可能不止一个最高或最低点
        #high_indeces = np.where(all_high_points == highest_value)[0]+N_STEPS
        #low_indeces = np.where(all_low_points == lowest_value)[0]+N_STEPS
        high_indeces = np.where(all_close_points == highest_value)[0]+N_STEPS
        low_indeces = np.where(all_close_points == lowest_value)[0]+N_STEPS
    else:
        all_high_points  = dataset.loc[:,"Top"].to_numpy()
        all_low_points  = dataset.loc[:,"Bottom"].to_numpy()
        high_indeces = np.where(all_high_points == 1)[0]
        low_indeces = np.where(all_low_points == 1)[0]
        IGNORE_SAMPLE_FINISH_WITH_MINMAX = False

    #如果几个最高点或最低点紧挨着，只保留连续的第一个位置
    result = []
    for i in range(len(high_indeces)):
        if (IGNORE_SAMPLE_FINISH_WITH_MINMAX == True) and ((high_indeces[i] >= len(dataset)-FEATURE_OFFSET-N_STEPS*2)):
            continue

        if i == 0 or high_indeces[i] != high_indeces[i-1] + 1:
            result.append(high_indeces[i])

    high_indeces = np.array(result)

    result = []
    for i in range(len(low_indeces)):
        if (IGNORE_SAMPLE_FINISH_WITH_MINMAX == True) and ((low_indeces[i] >= len(dataset)-FEATURE_OFFSET-N_STEPS*2)):
            continue

        if i == 0 or low_indeces[i] != low_indeces[i-1] + 1:
            result.append(low_indeces[i])

    low_indeces = np.array(result)

    
    #根据找出的所有高低点位出现的时间点，采样在此时间点之前N_STEPS个时段的开盘和收盘价，存入列表
    scaled_high_samples = list()
    scaled_low_samples = list()
    unscaled_high_samples = list()
    unscaled_low_samples = list()

    for i in range(len(high_indeces)):
        h = dataset.loc[high_indeces[i]-N_STEPS+FEATURE_OFFSET : high_indeces[i]-1+FEATURE_OFFSET, settings["FEATURES_SET"]].to_numpy().flatten().reshape(-1, 1)
        #设定缩放器
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(np.array([np.min(h), np.max(h)]).reshape(-1, 1)) 
        unscaled_high_samples.append(np.array(h).reshape(-1, FEATURES))
        h = scaler.transform(h) #缩放
        scaled_high_samples.append(np.array(h).reshape(-1, FEATURES))

    for i in range(len(low_indeces)):
        l = dataset.loc[low_indeces[i]-N_STEPS+FEATURE_OFFSET : low_indeces[i]-1+FEATURE_OFFSET, settings["FEATURES_SET"]].to_numpy().flatten().reshape(-1, 1)
        #设定缩放器
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(np.array([np.min(l), np.max(l)]).reshape(-1, 1))    
        unscaled_low_samples.append(np.array(l).reshape(-1, FEATURES))
        l = scaler.transform(l) #缩放
        scaled_low_samples.append(np.array(l).reshape(-1, FEATURES))
        
    #将该图片里的所有高点采样数据加入训练集
    add_train_set(scaled_high_samples, 1.00)
    #将该图片里的所有低点采样数据加入训练集
    add_train_set(scaled_low_samples, 0.00)

def prepare_data():    
    file_pattern = settings["SAMPLE_FILE_PATTERN"]

    #训练集的数据文件总数
    num_files = settings["TOTAL_SAMPLE_FILES"]

    if settings["AUTO_MARK"] == True:
        print("Using AUTO MARK, find out the top/bottom points automatically")
    else:
        print("NOT using AUTO MARK, the top/bottom points read from data files if the columns exist")

    #逐个处理训练集数据文件，将训练集数据加入train_set_X_list和train_set_y_list列表
    for i in range(1, num_files + 1):
        file_name = file_pattern.format(i)
        file_name = os.path.join(cfg.TRAIN_DATA_SUBDIR, file_name)
        process_data_file(data_file_name = file_name)
        if i % 100 == 0:
            print(f"{i}/{num_files} files processed, total {len(train_set_y_list)} samples, {n_top_samples} Top samples, {n_bottom_samples} Bottom samples")

    print(f"All training data files have been processed, total {len(train_set_y_list)} samples, {n_top_samples} Top samples, {n_bottom_samples} Bottom samples")

def create_LSTM_model():
    # The LSTM architecture
    model = Sequential()
    model.add(LSTM(units=125, activation="tanh", input_shape=(settings["N_STEPS"], len(settings["FEATURES_SET"])), return_sequences=True))
    model.add(LSTM(units=125, activation="tanh"))
    model.add(Dense(units=1, activation="sigmoid"))
    # Compiling the model
    model.compile(optimizer="RMSprop", loss="mse", metrics = ['accuracy'] )
    model.summary()

    return model

def create_SimpleRNN_model():
    model = Sequential()
    model.add(SimpleRNN(units=256, activation="tanh", input_shape=(settings["N_STEPS"], len(settings["FEATURES_SET"])), unroll=True, return_sequences=True))
    model.add(SimpleRNN(units=256, activation="tanh", unroll=True, return_sequences=True))
    model.add(SimpleRNN(units=256, activation="tanh", unroll=True)) 
    model.add(Dense(units=1, activation="sigmoid"))
    # Compiling the model
    model.compile(optimizer="RMSprop", loss="mse", metrics = ['accuracy'])
    model.summary()

    return model

def create_GRU_model():
    model = Sequential()
    model.add(GRU(units=125, activation="tanh", input_shape=(settings["N_STEPS"], len(settings["FEATURES_SET"])), return_sequences=True))
    model.add(GRU(units=125, activation="tanh"))
    model.add(Dense(units=1, activation="sigmoid"))
    # Compiling the model
    model.compile(optimizer="RMSprop", loss="mse", metrics = ['accuracy'])
    model.summary()

    return model

#MODEL_TYPE = SimpleRNN/GRU/LSTM
def start_train(MODEL_TYPE):

    if MODEL_TYPE == "SimpleRNN":
        model = create_SimpleRNN_model()
    elif MODEL_TYPE == "GRU":
        model = create_GRU_model()
    elif MODEL_TYPE == "LSTM":
        model = create_LSTM_model()
    else:
        print(f"Unknown MODEL_TYPE {MODEL_TYPE}")
        quit()

    #训练
    X_train = np.array(train_set_X_list)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], len(settings["FEATURES_SET"]))
    y_train = np.array(train_set_y_list)
    y_train = y_train.reshape(-1, 1)

    history = model.fit(X_train, y_train, epochs=settings["EPOCHS"], batch_size=settings["BATCH_SIZE"], validation_split=0.1)

    #保存模型
    model.save(settings["MODEL_FILE_NAME"])
    return model, history



#测试效果
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
        seq_x = np.array(seq_x).reshape(-1, len(settings["FEATURES_SET"]))
        X.append(seq_x)
    return np.array(X)

#用训练好的模型对测试数据集进行预测
def process_test_file(data_file_name):
    global dataset
    dataset = pd.read_csv(
        data_file_name, index_col="Date", parse_dates=["Date"]
    )

    testset_total = dataset.loc[:,settings["FEATURES_SET"]].to_numpy()    
    return test_X_sequence(testset_total, settings["N_STEPS"])

def start_predict():
    settings = cfg.load_settings()
    df, signals_buy, signals_sell = predict.predict(settings)
    fig = go.Figure(data=[go.Candlestick(x=pd.to_datetime(df['Date']),
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
    fig.add_trace(go.Scatter(x=pd.to_datetime(df['Date']), y=signals_buy,mode="markers+text",marker=dict(symbol='triangle-down-open', size = 12, color="blue")))
    fig.add_trace(go.Scatter(x=pd.to_datetime(df['Date']),y=signals_sell,mode="markers+text",marker=dict(symbol='triangle-down-open', size = 12, color="darkred")))
    return fig

def make_top_signals(signals, price, n_steps):
    signal   = []
    i = 0
    for value in price:
        if i >= n_steps and signals[i-n_steps] > 0.99:
            signal.append(value)
        else:
            signal.append(np.nan)
        i = i + 1
    return signal

def make_bottom_signals(signals, price, n_steps):
    signal   = []
    i = 0
    for value in price:
        if i >= n_steps and signals[i-n_steps] < 0.01:
            signal.append(value)
        else:
            signal.append(np.nan)
        
        i = i + 1
    return signal

def initialize():
    global settings
    settings = cfg.load_settings()
    
    global train_set_X_list, train_set_y_list, test_set_X_list, test_set_y_list
    train_set_X_list = list()
    train_set_y_list = list()
    test_set_X_list = list()
    test_set_y_list = list()

    global n_top_samples, n_bottom_samples
    n_top_samples = 0
    n_bottom_samples = 0
    
def get_training_history_loss_plot(ax, history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, settings["EPOCHS"]+1)
    ax[0].plot(epochs, loss, "bo", label="Training loss")
    ax[0].plot(epochs, val_loss, "b", label="Validation loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

def get_training_history_acc_plot(ax, history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    epochs = range(1, settings["EPOCHS"]+1)
    ax[1].plot(epochs, acc, "bo", label="Training accuracy")
    ax[1].plot(epochs, val_acc, "b", label="Validation accuracy")
    ax[1].set_title("Training and validation accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()


if __name__ == "__main__":
    initialize()
    prepare_data()
    model, history = start_train(settings["MODEL_TYPE"])
    fig,ax=plt.subplots(1,2)
    get_training_history_loss_plot(ax, history)
    get_training_history_acc_plot(ax, history)
    plt.show()
    
    start_predict().show()
    print("All done!")