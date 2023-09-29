import gradio as gr
import config as cfg
import highlow as trainer

# 创建界面
def create_interface():
    settings = cfg.load_settings()

    # 创建输入组件
    total_sample_files = gr.Number(label="TOTAL_SAMPLE_FILES", precision=0, value=settings["TOTAL_SAMPLE_FILES"])
    features_set = gr.CheckboxGroup(label="FEATURES_SET", choices=cfg.DEFAULT_SETTINGS["FEATURES_SET"], value=settings["FEATURES_SET"])
    n_steps = gr.Number(label="N_STEPS", precision=0, value=settings["N_STEPS"])
    feature_offset = gr.Number(label="FEATURE_OFFSET", precision=0, value=settings["FEATURE_OFFSET"])
    sample_file_pattern = gr.Textbox(label="SAMPLE_FILE_PATTERN", value=settings["SAMPLE_FILE_PATTERN"])
    raw_data_file = gr.Textbox(label="RAW_DATA_FILE", value=settings["RAW_DATA_FILE"])
    test_file_name = gr.Textbox(label="TEST_FILE_NAME", value=settings["TEST_FILE_NAME"])
    model_file_name = gr.Textbox(label="MODEL_FILE_NAME", value=settings["MODEL_FILE_NAME"])
    model_type = gr.Dropdown(label="MODEL_TYPE", choices=["SimpleRNN", "GRU", "LSTM"], value=settings["MODEL_TYPE"])
    epochs = gr.Number(label="EPOCHS", precision=0, value=settings["EPOCHS"])
    batch_size = gr.Number(label="BATCH_SIZE", precision=0, value=settings["BATCH_SIZE"])
    auto_mark = gr.Checkbox(label="AUTO_MARK", value=settings["AUTO_MARK"])

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
        
        #trainer.train_highlow()

    # 创建界面
    interface = gr.Interface(
        fn=update_settings,
        inputs=[total_sample_files, features_set, n_steps, feature_offset, sample_file_pattern, raw_data_file, test_file_name, model_file_name, model_type, epochs, batch_size, auto_mark],
        outputs=None,
        title="参数设置界面",
        description="使用Gradio生成的用户界面，用于设置参数并保存到JSON文件中。",
        theme="light"
    )
    return interface

# 运行界面
if __name__ == "__main__":
    interface = create_interface()
    interface.launch()