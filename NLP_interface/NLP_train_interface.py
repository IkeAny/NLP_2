import os
import sys
import shutil
import zipfile
import gradio as gr
from NLP_program.NLP_train import train_nlp_model, load_data

sys.path.append('NLP_interface')
sys.path.append('NLP_program')
sys.path.append('NLP_MachineLearning')
sys.path.append('./')

shared_theme = gr.themes.Base()

def retrain_and_download(user_text, user_label):
    train_data, valid_data = load_data('data/sentiment1_data.csv')
    new_data = (user_text, {'positive': user_label == "positive", 'negative': user_label != "positive"})
    train_data.append(new_data)
    
    # Train and save the updated model
    train_nlp_model(train_data, valid_data, 'model_sentiment_updated')
    
    # Zip the updated model directory for download
    shutil.make_archive('model_sentiment_updated', 'zip', 'model_sentiment_updated')
    return 'model_sentiment_updated.zip'

def build_train_interface():
    with gr.Blocks(theme=shared_theme) as app:
        gr.Markdown("# Sentiment Analysis Model Trainer")
        gr.Markdown("""
        Sentiment analysis is a text analytics technique that uses machine learning and natural language processing (NLP) to determine if a text has a positive, negative, or neutral emotional tone.
        This interface allows you to input a text example and its sentiment (positive or negative). 
        Upon submission, the example is added to the training data, and the sentiment analysis model is retrained. 
        You can then download the updated model directly from this interface.
        """)
        with gr.Row():
            gr.Markdown("""Add a new training example and retrain the model.""")
        with gr.Row():
            text_input = gr.Textbox(label="Input Text", placeholder="Type your sentiment text here...")
            label_input = gr.Radio(choices=["positive", "negative"], label="Sentiment")
            submit_btn = gr.Button("Submit and Retrain")
        output = gr.File(label="Download Updated Model")

        clear_comp_list = [text_input, label_input, output]


        submit_btn.click(fn=retrain_and_download, inputs=[text_input, label_input], outputs=output)
        clear_but = gr.ClearButton(value='Clear All',components=clear_comp_list,
                    interactive=True,visible=True)
    return app

if __name__== "__main__" :
    app = build_train_interface()