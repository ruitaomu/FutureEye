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
    df = maketest.make(year, month, date)
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
        

def start_predict(floating_point_adj):
    settings["FLOATING_POINT_ADJUSTMENT"] = floating_point_adj
    cfg.save_settings(settings)
    df, signals_buy, signals_sell = predict.predict(settings)
    fig = go.Figure(data=[go.Candlestick(name="Price", x=pd.to_datetime(df['Date']),
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
    fig.add_trace(go.Scatter(name="Low", x=pd.to_datetime(df['Date']), y=signals_buy,mode="markers+text",marker=dict(symbol='triangle-down-open', size = 12, color="blue")))
    fig.add_trace(go.Scatter(name="High", x=pd.to_datetime(df['Date']),y=signals_sell,mode="markers+text",marker=dict(symbol='triangle-down-open', size = 12, color="darkred")))
    return fig

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
    df.loc[len(df)] = [snapshot_name, settings['EPOCHS'], settings['BATCH_SIZE'], settings['N_STEPS'], settings['FEATURE_OFFSET'], settings['FEATURES_SET'], settings['MODEL_TYPE']]
    df.to_csv(modelinfo_csv, index=False)
    
    destination_file = f"{cfg.SNAPSHOT_SUBDIR}/{snapshot_name}"
    shutil.copy(settings["MODEL_FILE_NAME"], destination_file)
    
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
            snapshot_name = gr.Textbox(label="SNAPSHOT_NAME", value="")
            snapshot = gr.Button(value="Snapshot")
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
        floating_point_adj = gr.Checkbox(label="FLOATING_POINT_ADJUSTMENT", value=lambda: settings["FLOATING_POINT_ADJUSTMENT"])
        predict_btn = gr.Button(value="Predict")
    with gr.Row():
        predict_plt = gr.Plot(visible=True)

    auto_mark.change(event_automark_changed, inputs=[auto_mark], outputs=[floating_point_adj, total_sample_files])  
    save.click(update_settings, inputs=[total_sample_files, features_set, n_steps, feature_offset, sample_file_pattern, raw_data_file, test_file_name, model_file_name, model_type, epochs, batch_size, auto_mark, floating_point_adj, predict_source_file],
        outputs=None)
    train.click(start_training, inputs=[total_sample_files, features_set, n_steps, feature_offset, sample_file_pattern, raw_data_file, test_file_name, model_file_name, model_type, epochs, batch_size, auto_mark, floating_point_adj, predict_source_file], outputs=train_res_plot)
    make.click(make_data, inputs=[total_sample_files, features_set, n_steps, feature_offset, sample_file_pattern, raw_data_file, test_file_name, model_file_name, model_type, epochs, batch_size, auto_mark, floating_point_adj, predict_source_file], outputs=display_box)
    maketest_btn.click(make_test, inputs=[year, month, date, predict_source_file], outputs=predict_plt)
    predict_btn.click(start_predict, inputs=floating_point_adj, outputs=predict_plt)
    snapshot.click(snapshot_model, inputs=snapshot_name, outputs=None)
 
# 运行界面
if __name__ == "__main__":
    interface.launch()