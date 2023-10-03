import gradio as gr
import pandas as pd
import config as cfg
import plotly.graph_objects as go
import predict as predict
import selecttest as maketest
import os

# 从 CSV 文件中读取数据，并将其转换为 DataFrame
def read_csv_file(file_path):
    if os.path.exists(file_path):
        global models_df
        models_df = pd.read_csv(file_path)
        return models_df
    else:
        return None

def start_predict(model_name, pred_setting):
    df, signals_buy, signals_sell = predict.predict(pred_setting, False , model_name)
    fig = go.Figure(data=[go.Candlestick(name="Price", x=pd.to_datetime(df['Date']),
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
    fig.add_trace(go.Scatter(name="Low", x=pd.to_datetime(df['Date']), y=signals_buy,mode="markers+text",marker=dict(symbol='triangle-down-open', size = 12, color="blue")))
    fig.add_trace(go.Scatter(name="High", x=pd.to_datetime(df['Date']),y=signals_sell,mode="markers+text",marker=dict(symbol='triangle-down-open', size = 12, color="darkred")))
    return fig

def on_select(evt: gr.SelectData):
    index = evt.index[0]
    name = models_df["Name"][index]
    name = f"{cfg.SNAPSHOT_SUBDIR}/{name}"
    temp_setting = settings.copy()
    temp_setting["N_STEPS"] = int(models_df["Steps"][index])
    temp_setting["FEATURE_OFFSET"] = int(models_df["Feature Offset"][index])
    temp_setting["MODEL_TYPE"] = models_df["Model Type"][index]
    temp_setting["EPOCHS"] = int(models_df["EPOCH"][index])
    temp_setting["BATCH_SIZE"] = int(models_df["Batch Size"][index])
    temp_setting["FLOATING_POINT_ADJUSTMENT"] = False
    features_set = eval(models_df["Feature Set"][index])
    temp_setting["FEATURES_SET"] = features_set
    return start_predict(name, temp_setting)

def on_refresh():
    return gr.DataFrame(label="输出结果", value=read_csv_file(f"{cfg.SNAPSHOT_SUBDIR}/{cfg.MODELINFO_FILE}"))

def make_test(year, month, date, predict_source_file):
    settings["PREDICT_SOURCE_FILE"] = predict_source_file
    settings["TEST_YEAR"] = year
    settings["TEST_MONTH"] = month
    settings["TEST_DATE"] = date
    cfg.save_settings(settings)
    df = maketest.make(year, month, date)
    return go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

if not os.path.exists(cfg.SNAPSHOT_SUBDIR):
    os.makedirs(cfg.SNAPSHOT_SUBDIR)
    
settings = cfg.load_settings()

# 创建界面
with gr.Blocks() as demo:
    with gr.Row():
        predict_plt = gr.Plot(visible=True)
        
    with gr.Row():
        csv_output = gr.DataFrame(label="输出结果", value=read_csv_file(f"{cfg.SNAPSHOT_SUBDIR}/{cfg.MODELINFO_FILE}"))
        csv_output.select(on_select, inputs=None, outputs=predict_plt)
    with gr.Row():
        csv_refresh = gr.Button(value="Refresh")
        csv_refresh.click(on_refresh, inputs=None, outputs=csv_output)
    with gr.Row():
        predict_source_file = gr.Textbox(label="PREDICT_SOURCE_FILE", value=lambda: settings["PREDICT_SOURCE_FILE"])
        test_file_name = gr.Textbox(label="TEST_FILE_NAME", value=lambda: settings["TEST_FILE_NAME"])
        year = gr.Number(label="TEST_YEAR", precision=0, minimum=1980, value=lambda: settings["TEST_YEAR"])
        month = gr.Number(label="TEST_MONTH", precision=0, minimum=1, maximum=12, value=lambda: settings["TEST_MONTH"])
        date = gr.Number(label="TEST_DATE", precision=0, minimum=1, maximum=31, value=lambda: settings["TEST_DATE"])
        maketest_btn = gr.Button(value="Make Test File")
        maketest_btn.click(make_test, inputs=[year, month, date, predict_source_file], outputs=predict_plt)
    
demo.launch()