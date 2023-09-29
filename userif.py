import gradio as gr
import config as cfg
import highlow as trainer
import matplotlib.pyplot as plt
import predict_step as predict
import makedata

# 更新设置并保存到JSON文件
def update_settings(total_sample_files, features_set, n_steps, feature_offset, sample_file_pattern, raw_data_file, test_file_name, model_file_name, model_type, epochs, batch_size, auto_mark):
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
    cfg.save_settings(settings)

def start_training():
    trainer.initialize()
    trainer.prepare_data()
    model, history = trainer.start_train(settings["MODEL_TYPE"])
    fig,ax=plt.subplots(1,2)
    trainer.get_training_history_loss_plot(ax, history)
    trainer.get_training_history_acc_plot(ax, history)
    
    return gr.Plot(value=plt, visible=True)

settings = cfg.load_settings()
with gr.Blocks() as interface:
    with gr.Row():
        with gr.Column():
            sample_file_pattern = gr.Textbox(label="SAMPLE_FILE_PATTERN", value=settings["SAMPLE_FILE_PATTERN"])
            raw_data_file = gr.Textbox(label="RAW_DATA_FILE", value=settings["RAW_DATA_FILE"])
            test_file_name = gr.Textbox(label="TEST_FILE_NAME", value=settings["TEST_FILE_NAME"])
            model_file_name = gr.Textbox(label="MODEL_FILE_NAME", value=settings["MODEL_FILE_NAME"])
        with gr.Column():
            model_type = gr.Dropdown(label="MODEL_TYPE", choices=["SimpleRNN", "GRU", "LSTM"], value=settings["MODEL_TYPE"])
            epochs = gr.Number(label="EPOCHS", precision=0, value=settings["EPOCHS"])
            batch_size = gr.Number(label="BATCH_SIZE", precision=0, value=settings["BATCH_SIZE"])
            auto_mark = gr.Checkbox(label="AUTO_MARK", value=settings["AUTO_MARK"])
        with gr.Column():
            total_sample_files = gr.Number(label="TOTAL_SAMPLE_FILES", precision=0, value=settings["TOTAL_SAMPLE_FILES"])
            features_set = gr.CheckboxGroup(label="FEATURES_SET", choices=cfg.DEFAULT_SETTINGS["FEATURES_SET"], value=settings["FEATURES_SET"])
            n_steps = gr.Number(label="N_STEPS", precision=0, value=settings["N_STEPS"])
            feature_offset = gr.Number(label="FEATURE_OFFSET", precision=0, value=settings["FEATURE_OFFSET"])
    with gr.Row():
        save = gr.Button(value="Save")
        make = gr.Button(value="Make Data")
    with gr.Row():
        train = gr.Button(value="Train")
    with gr.Row():
        train_res_plot = gr.Plot(visible=False)

    save.click(update_settings, inputs=[total_sample_files, features_set, n_steps, feature_offset, sample_file_pattern, raw_data_file, test_file_name, model_file_name, model_type, epochs, batch_size, auto_mark],
        outputs=None)
    train.click(start_training, inputs=None, outputs=train_res_plot)
    make.click(makedata.make, inputs=None, outputs=None)

 
# 运行界面
if __name__ == "__main__":
    interface.launch()