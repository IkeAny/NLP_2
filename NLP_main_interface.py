import gradio as gr
import sys
import pandas as pd
import spacy
from spacy.training import Example
from spacy.util import minibatch
import random
from pathlib import Path

sys.path.append('NLP_interface')
sys.path.append('NLP_program')
sys.path.append('NLP_MachineLearning')
sys.path.append('./')  # to run '$ python *.py' files in subdirectories


shared_theme = gr.themes.Base()

#from train import data
from NLP_interface.NLP_detect_interface import build_detect_interface
from NLP_interface.NLP_train_interface import build_train_interface


def build_main_interface():
    detect = build_detect_interface()
    train = build_train_interface()
    
    with gr.Blocks(title="NLP Interface",theme=shared_theme) as iface:
        gr.Markdown("# NLP interface made with SPACY and OPENAI")
        gr.Markdown(
        """
        Natural Language Processing (NLP) bridges computers and human language, aiming to understand and interpret 
        human language in a valuable way. It leverages computational linguistics and machine learning to enable 
        applications like speech recognition, language translation, and chatbots. Despite human language's complexity, 
        advances in deep learning have greatly improved NLP's capabilities. Today, NLP is essential in how we interact 
        with technology, making it more intuitive and responsive to our needs.

        Choose between the Detect and Train interfaces:
        """)
        gr.TabbedInterface(interface_list=[detect, train], 
                            tab_names=["Detect", "Train"],
                            theme=shared_theme,
                            analytics_enabled=True)
    
            
    return iface

if __name__== "__main__" :
    # run_main_interface()
    iface = build_main_interface()
    iface.queue().launch()