import sys
import gradio as gr
import openai
import pandas as pd
from NLP_program.NLP_detect import analyze_text, generate_description

sys.path.append('NLP_interface')
sys.path.append('NLP_program')
sys.path.append('NLP_MachineLearning')
sys.path.append('./')

shared_theme = gr.themes.Base()

# def openai_key(openai_api_key):
#     openai.api_key = openai_api_key

# def complete_analysis(text, question, num_words, quality, temperature):
#     #openai.api_key = openai_api_key
#     if text.strip():
#         entities_df, keywords_df, sentiment_score = analyze_text(text, num_words)
#         keywords = keywords_df['Keyword'].tolist()
#         description = generate_description(question, text, keywords, quality, temperature)
#         return entities_df, keywords_df, description
#     else:
#         return pd.DataFrame(), pd.DataFrame(), "No text provided."
    

def complete_analysis(text, question, num_words, quality, temperature, api_key):
    entities_df, keywords, sentiment_score = analyze_text(text, num_words)
    description = generate_description(question, text, keywords, quality, temperature, api_key)
    return entities_df, keywords, description
    

def build_detect_interface():
    with gr.Blocks(theme=shared_theme) as iface:
        gr.Markdown("# NLP Text Analysis with AI-Generated Description")
        gr.Markdown("""
        There are 4 different tabs, 3 for anaalysis which can simultaneously run at the same time and a description tab which talks about what this program does. Enjoy!
        """)
        api_key_input = gr.Textbox(label="OpenAI API Key", placeholder="Enter your OpenAI API key here", type="password")
        
        with gr.Tab("Description"):
            gr.Markdown("""
            Created by Chinedu Ike-Anyanwu, this NLP tool is designed to analyze text, extract and rank keywords, and generate descriptive summaries or answers to specific questions. It leverages advanced AI models to provide insights into the provided text, making it a valuable resource for faculty and students of Rowan University.

            - **Question (optional):** You can ask a specific question related to the text, or leave this blank to get a summary based on the input text and the most common words.
            - **Input Text:** Place the text you want analyzed here.
            - **Number of Keywords:** Select how many keywords to focus on. This influences the focus of the AI's analysis and generated description.
            - **Choose Quality:** Choose 'Speed' for faster, less detailed responses using 'gpt-3.5-turbo-instruct-0914'. Choose 'Accuracy' for more detailed responses using 'gpt-3.5-turbo-16k-0613'.
            - **Creativity Level (Temperature):** Adjust the creativity of the AI's response. Lower values produce more predictable, conservative outputs; higher values encourage creativity and novelty.

            This tool is intended for use by the faculty or students of Rowan University as part of educational and research activities.            
        """)
        with gr.Tab("Analysis 1"):
            gr.Markdown("""
            This tab is for analysis 1.
            """)
            with gr.Row():
                question1 = gr.Textbox(label="Question 1 (optional)", placeholder="Type your question here...")
                text1 = gr.TextArea(label="Input Text 1", placeholder="Type the text here...", lines=7)
            with gr.Row():
                num_words1 = gr.Slider(minimum=1, maximum=20, step=1, label="Number of Keywords for Analysis 1")
                temperature1 = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Creativity Level (Temperature) for Analysis 1")
                quality1 = gr.Radio(choices=["Speed", "Accuracy"], label="Choose Quality")
            analyze_btn1 = gr.Button("Analyze Text 1")

            entities1 = gr.Dataframe(label="Named Entities 1")
            keywords1 = gr.Dataframe(label="Keywords Ranked by Occurrence 1")
            description1 = gr.Textbox(label="AI-Generated Description 1")
            
            clear_comp_list1 = [text1, question1, num_words1, quality1, temperature1, entities1, keywords1, description1]

            analyze_btn1.click(fn=complete_analysis, inputs=[text1, question1, num_words1, quality1, temperature1, api_key_input], outputs=[entities1, keywords1, description1])
            clear_but1 = gr.ClearButton(value='Clear All',components=clear_comp_list1,
                    interactive=True,visible=True)
            
        with gr.Tab("Analysis 2"):
            gr.Markdown("""
            This tab is for analysis 2.
            """)
            with gr.Row():
                question2 = gr.Textbox(label="Question 2 (optional)", placeholder="Type your question here...")
                text2 = gr.TextArea(label="Input Text 2", placeholder="Type the text here...", lines=7)
            with gr.Row():
                num_words2 = gr.Slider(minimum=1, maximum=20, step=1, label="Number of Keywords for Analysis 2")
                temperature2 = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Creativity Level (Temperature) for Analysis 2")
                quality2 = gr.Radio(choices=["Speed", "Accuracy"], label="Choose Quality")
            analyze_btn2 = gr.Button("Analyze Text 2")
            
            entities2 = gr.Dataframe(label="Named Entities 2")
            keywords2 = gr.Dataframe(label="Keywords Ranked by Occurrence 2")
            description2 = gr.Textbox(label="AI-Generated Description 2")
            
            clear_comp_list2 = [text2, question2, num_words2, quality2, temperature2, entities2, keywords2, description2]

            analyze_btn2.click(fn=complete_analysis, inputs=[text2, question2, num_words2, quality2, temperature2, api_key_input], outputs=[entities2, keywords2, description2])
            clear_but2 = gr.ClearButton(value='Clear All',components=clear_comp_list2,
                    interactive=True,visible=True)
            
        with gr.Tab("Analysis 3"):
            gr.Markdown("""
            This tab is for analysis 3.
            """)
            with gr.Row():
                question3 = gr.Textbox(label="Question 3 (optional)", placeholder="Type your question here...")
                text3 = gr.TextArea(label="Input Text 3", placeholder="Type the text here...", lines=7)
            with gr.Row():
                num_words3 = gr.Slider(minimum=1, maximum=20, step=1, label="Number of Keywords for Analysis 3")
                temperature3 = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Creativity Level (Temperature) for Analysis 3")
                quality3 = gr.Radio(choices=["Speed", "Accuracy"], label="Choose Quality")
            analyze_btn3 = gr.Button("Analyze Text 3")
            
            entities3 = gr.Dataframe(label="Named Entities 3")
            keywords3 = gr.Dataframe(label="Keywords Ranked by Occurrence 3")
            description3 = gr.Textbox(label="AI-Generated Description 3")
            
            clear_comp_list3 = [text3, question3, num_words3, quality3, temperature3, entities3, keywords3, description3]

            analyze_btn3.click(fn=complete_analysis, inputs=[ text3, question3, num_words3, quality3, temperature3, api_key_input], outputs=[entities3, keywords3, description3])
            clear_but3 = gr.ClearButton(value='Clear All',components=clear_comp_list3,
                    interactive=True,visible=True)
    return iface
if __name__== "__main__" :
    iface = build_detect_interface()