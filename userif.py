import gradio as gr
import config as cfg
import highlow as trainer
import matplotlib.pyplot as plt
import predict as predict
import makedata
import selecttest as maketest
import plotly.graph_objects as go
import pandas as pd
import os
import shutil

# 更新设置并保存到JSON文件
def update_settings(total_sample_files, features_set, n_steps, feature_offset, sample_file_pattern, raw_data_file, test_file_name, model_file_name, model_type, epochs, batch_size, auto_mark, floating_point_adj, predict_source_file):
    settings["TOTAL_SAMPLE_FILES"] = total_sample_files
    settings["FEATURES_SET"] = features_set
    settings["N_STEPS"] = n_steps
    settings["FEATURE_OFFSET"] = feature_offset
    settings["SAMPLE_FILE_PATTERN"] = sample_file_pattern
    settings["RAW_DATA_FILE"] = raw_data_file
    settings["TEST_FILE_NAME"] = test_file_name
    settings["MODEL_FILE_NAME"] = model_file_name
    settings["MODEL_TYPE"] = model_type
    settings["EPOCHS"] = epochs
    settings["BATCH_SIZE"] = batch_size
    settings["AUTO_MARK"] = auto_mark
    settings["FLOATING_POINT_ADJUSTMENT"] = floating_point_adj
    settings["PREDICT_SOURCE_FILE"] = predict_source_file
    cfg.save_settings(settings)

def start_training(total_sample_files, features_set, n_steps, feature_offset, sample_file_pattern, raw_data_file, test_file_name, model_file_name, model_type, epochs, batch_size, auto_mark, floating_point_adj, predict_source_file):
    update_settings(total_sample_files, features_set, n_steps, feature_offset, sample_file_pattern, raw_data_file, test_file_name, model_file_name, model_type, epochs, batch_size, auto_mark, floating_point_adj, predict_source_file)
    trainer.initialize()
    trainer.prepare_data()
    model, history = trainer.start_train(settings["MODEL_TYPE"])
    fig,ax=plt.subplots(1,2)
    trainer.get_training_history_loss_plot(ax, history)
    trainer.get_training_history_acc_plot(ax, history)
    
    return gr.Plot(value=plt, visible=True)

def make_data(total_sample_files, features_set, n_steps, feature_offset, sample_file_pattern, raw_data_file, test_file_name, model_file_name, model_type, epochs, batch_size, auto_mark, floating_point_adj, predict_source_file):
    update_settings(total_sample_files, features_set, n_steps, feature_offset, sample_file_pattern, raw_data_file, test_file_name, model_file_name, model_type, epochs, batch_size, auto_mark, floating_point_adj, predict_source_file)
    str = makedata.make()
    return gr.Textbox(label="display_box", value=str)

def make_test(year, month, date, predict_source_file):
    settings["PREDICT_SOURCE_FILE"] = predict_source_file
    settings["TEST_YEAR"] = year
    settings["TEST_MONTH"] = month
    settings["TEST_DATE"] = date
    cfg.save_settings(settings)
    df = maketest.make(year, month, date, 3)
    return go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

def event_automark_changed(automark):
    if automark:
        return gr.Checkbox.update(interactive=True), gr.Checkbox.update(interactive=True)
    else:
        return gr.Checkbox.update(value=False, interactive=False), gr.Number.update(value=1, interactive=False)
        
def generate_figure(dataframe, signals_buy, signals_sell):
    fig = go.Figure(data=[go.Candlestick(name="Price", x=pd.to_datetime(dataframe['Date']),
                open=dataframe['Open'],
                high=dataframe['High'],
                low=dataframe['Low'],
                close=dataframe['Close'])])
    fig.add_trace(go.Scatter(name="Low", x=pd.to_datetime(dataframe['Date']), y=signals_buy,mode="markers+text",marker=dict(symbol='triangle-down-open', size = 12, color="blue")))
    fig.add_trace(go.Scatter(name="High", x=pd.to_datetime(dataframe['Date']),y=signals_sell,mode="markers+text",marker=dict(symbol='star-open', size = 12, color="darkred")))
    return fig
    
def start_predict(floating_point_adj, exec_buy_adj, exec_sell_adj, exec_buy_onlyonce, exec_must_see_confirm, exec_ma_adj_value):
    settings["FLOATING_POINT_ADJUSTMENT"] = floating_point_adj
    settings["EXECUTION_BUYING_ADJ_MA"] = exec_buy_adj
    settings["EXECUTION_SELLING_ADJ_MA"] = exec_sell_adj
    settings["EXECUTION_BUYING_ADJ_ONLYONCE"] = exec_buy_onlyonce
    settings["EXECUTION_MUST_SEE_CONFIRM"] = exec_must_see_confirm
    settings["MA_ADJ_VALUE"] = exec_ma_adj_value
    cfg.save_settings(settings)
    df, signals_buy, signals_sell = predict.predict(settings)
    return generate_figure(df, signals_buy, signals_sell)

def start_predict_by_model_name(model_name, pred_setting):
    df, signals_buy, signals_sell = predict.predict(pred_setting, False , model_name)
    return generate_figure(df, signals_buy, signals_sell)

def snapshot_model(name):
    if len(name) == 0:
        return
    
    settings = cfg.load_settings()
    
    if not os.path.exists(settings["MODEL_FILE_NAME"]):
        return
     
    if not os.path.exists(cfg.SNAPSHOT_SUBDIR):
        os.makedirs(cfg.SNAPSHOT_SUBDIR)
    
    df = pd.DataFrame(columns=['Name', 'EPOCH', 'Batch Size', 'Steps', 'Feature Offset', 'Feature Set', 'Model Type'])
    modelinfo_csv = f"{cfg.SNAPSHOT_SUBDIR}/{cfg.MODELINFO_FILE}"
    if os.path.exists(modelinfo_csv):
        df = pd.read_csv(modelinfo_csv)
    
    snapshot_name = f"{name}.keras"
    destination_file = f"{cfg.SNAPSHOT_SUBDIR}/{snapshot_name}"
    if os.path.exists(destination_file):
        #如果同名Snapshot存在则不保存
        return
    
    df.loc[len(df)] = [snapshot_name, settings['EPOCHS'], settings['BATCH_SIZE'], settings['N_STEPS'], settings['FEATURE_OFFSET'], settings['FEATURES_SET'], settings['MODEL_TYPE']]
    df.to_csv(modelinfo_csv, index=False)
    
    shutil.copy(settings["MODEL_FILE_NAME"], destination_file)
    on_refresh()
    
# 从 CSV 文件中读取数据，并将其转换为 DataFrame
def read_snapshot_table(file_path):
    if os.path.exists(file_path):
        global models_df
        models_df = pd.read_csv(file_path)
        return models_df
    else:
        return None

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
    features_set = eval(models_df["Feature Set"][index])
    temp_setting["FEATURES_SET"] = features_set
    return start_predict_by_model_name(name, temp_setting)

def on_refresh():
    return gr.DataFrame(label="输出结果", value=read_snapshot_table(f"{cfg.SNAPSHOT_SUBDIR}/{cfg.MODELINFO_FILE}"))

    
settings = cfg.load_settings()
with gr.Blocks() as interface:
    with gr.Row():
        with gr.Column():
            sample_file_pattern = gr.Textbox(label="SAMPLE_FILE_PATTERN", value=lambda: settings["SAMPLE_FILE_PATTERN"])
            raw_data_file = gr.Textbox(label="RAW_DATA_FILE", value=lambda: settings["RAW_DATA_FILE"])
        with gr.Column():
            auto_mark = gr.Checkbox(label="AUTO_MARK", value=lambda: settings["AUTO_MARK"])
            total_sample_files = gr.Number(label="TOTAL_SAMPLE_FILES", precision=0, value=lambda: settings["TOTAL_SAMPLE_FILES"])
        with gr.Column():
            make = gr.Button(value="Make Data")
            display_box = gr.Label(label="Make Data Output", value="")
    with gr.Row():        
        save = gr.Button(value="Save")
    with gr.Row():
        with gr.Column():
            epochs = gr.Number(label="EPOCHS", precision=0, value=lambda: settings["EPOCHS"])
            batch_size = gr.Number(label="BATCH_SIZE", precision=0, value=lambda: settings["BATCH_SIZE"])
            n_steps = gr.Number(label="N_STEPS", precision=0, value=lambda: settings["N_STEPS"])
            feature_offset = gr.Number(label="FEATURE_OFFSET", precision=0, value=lambda: settings["FEATURE_OFFSET"])
        with gr.Column():
            model_file_name = gr.Textbox(label="MODEL_FILE_NAME", value=lambda: settings["MODEL_FILE_NAME"])
            model_type = gr.Dropdown(label="MODEL_TYPE", choices=["SimpleRNN", "GRU", "LSTM"], value=lambda: settings["MODEL_TYPE"])
            features_set = gr.CheckboxGroup(label="FEATURES_SET", choices=cfg.DEFAULT_SETTINGS["FEATURES_SET"], value=lambda: settings["FEATURES_SET"])
            train = gr.Button(value="Train")
            
    with gr.Row():
        train_res_plot = gr.Plot(visible=True)
    with gr.Row():
        predict_source_file = gr.Textbox(label="PREDICT_SOURCE_FILE", value=lambda: settings["PREDICT_SOURCE_FILE"])
        test_file_name = gr.Textbox(label="TEST_FILE_NAME", value=lambda: settings["TEST_FILE_NAME"])
        year = gr.Number(label="TEST_YEAR", precision=0, minimum=1980, value=lambda: settings["TEST_YEAR"])
        month = gr.Number(label="TEST_MONTH", precision=0, minimum=1, maximum=12, value=lambda: settings["TEST_MONTH"])
        date = gr.Number(label="TEST_DATE", precision=0, minimum=1, maximum=31, value=lambda: settings["TEST_DATE"])
        maketest_btn = gr.Button(value="Make Test File")
    with gr.Row():
        floating_point_adj = gr.Checkbox(label="SLIDING_POINT_ADJUSTMENT", value=lambda: settings["FLOATING_POINT_ADJUSTMENT"])
        exec_buy_adj = gr.Checkbox(label="BUYING MA_ADJ", value=lambda: settings["EXECUTION_BUYING_ADJ_MA"])
        exec_sell_adj = gr.Checkbox(label="SELLING MA_ADJ", value=lambda: settings["EXECUTION_SELLING_ADJ_MA"])
        exec_buy_onlyonce = gr.Checkbox(label="BUYING ONLYONCE", value=lambda: settings["EXECUTION_BUYING_ADJ_ONLYONCE"])
        exec_must_see_confirm = gr.Checkbox(label="WAIT CONFIRM", value=lambda: settings["EXECUTION_MUST_SEE_CONFIRM"])
        exec_ma_adj_value = gr.Number(label="MA_ADJ VALUE", precision=0, minimum=1, value=lambda: settings["MA_ADJ_VALUE"])
        
        predict_btn = gr.Button(value="Predict")
    
    with gr.Row():
        predict_plt = gr.Plot(visible=True)
    with gr.Row():
        snapshot_name = gr.Textbox(label="SNAPSHOT_NAME", value="")
        snapshot = gr.Button(value="Snapshot")
    with gr.Row():
        csv_output = gr.DataFrame(label="Model Snapshots", value=read_snapshot_table(f"{cfg.SNAPSHOT_SUBDIR}/{cfg.MODELINFO_FILE}"))
        csv_output.select(on_select, inputs=None, outputs=predict_plt)
    with gr.Row():
        csv_refresh = gr.Button(value="Refresh")
        csv_refresh.click(on_refresh, inputs=None, outputs=csv_output)

    auto_mark.change(event_automark_changed, inputs=[auto_mark], outputs=[floating_point_adj, total_sample_files])  
    save.click(update_settings, inputs=[total_sample_files, features_set, n_steps, feature_offset, sample_file_pattern, raw_data_file, test_file_name, model_file_name, model_type, epochs, batch_size, auto_mark, floating_point_adj, predict_source_file],
        outputs=None)
    train.click(start_training, inputs=[total_sample_files, features_set, n_steps, feature_offset, sample_file_pattern, raw_data_file, test_file_name, model_file_name, model_type, epochs, batch_size, auto_mark, floating_point_adj, predict_source_file], outputs=train_res_plot)
    make.click(make_data, inputs=[total_sample_files, features_set, n_steps, feature_offset, sample_file_pattern, raw_data_file, test_file_name, model_file_name, model_type, epochs, batch_size, auto_mark, floating_point_adj, predict_source_file], outputs=display_box)
    maketest_btn.click(make_test, inputs=[year, month, date, predict_source_file], outputs=predict_plt)
    predict_btn.click(start_predict, inputs=[floating_point_adj, exec_buy_adj, exec_sell_adj, exec_buy_onlyonce, exec_must_see_confirm, exec_ma_adj_value], outputs=predict_plt)
    snapshot.click(snapshot_model, inputs=snapshot_name, outputs=None)
 
# 运行界面
if __name__ == "__main__":
    interface.launch()