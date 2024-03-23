# run python -m spacy download en_core_web_trf and pip install -r requirements.txt
import spacy
import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import openai

# Directly set your OpenAI API key here
openai.api_key = "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"

# Load the spaCy model
nlp = spacy.load("en_core_web_trf")

def analyze_text(text, num_words, quality):
    """
    Analyzes text using spaCy for NER and ranks keywords based on occurrence in the text.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents][:num_words]
    entities_df = pd.DataFrame(entities, columns=["Entity", "Type"])
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tf_idf_matrix = vectorizer.fit_transform([text])
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = tf_idf_matrix.toarray()[0].argsort()[::-1]
    
    top_keywords = feature_array[tfidf_sorting][:num_words]
    word_counts = Counter([word for word in text.lower().split() if word in top_keywords])
    keywords_counts = [(word, count) for word, count in word_counts.items()]
    keywords_df = pd.DataFrame(keywords_counts, columns=["Keyword", "Count"]).sort_values(by="Count", ascending=False)
    
    return keywords_df, entities_df

def generate_description(question, text, keywords, quality, temperature):
    """
    Generates a description using OpenAI's GPT-3.5 Turbo model based on the input text,
    incorporating the provided question (if any), focusing on the important keywords,
    and adjusting for quality and temperature.
    """
    keywords_str = ", ".join(keywords['Keyword'])
    
    if question.strip():
        prompt = f"Based on the text: '{text}' and considering these keywords: {keywords_str}, answer the question: '{question}'."
    else:
        prompt = f"Summarize the following text, making sure to emphasize these keywords: {keywords_str}. Text: '{text}'"
    
    model = "gpt-3.5-turbo-instruct-0914"
    max_tokens = 100 if quality == "Speed" else 150

    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return response['choices'][0]['text'].strip()

def NLP_analysis(question, text, num_words, quality, temperature):
    keywords_df, entities_df = analyze_text(text, num_words, quality)
    description = generate_description(question, text, keywords_df, quality, temperature)
    return entities_df, keywords_df, description

# Define the Gradio interface
iface = gr.Interface(
    fn=NLP_analysis,
    inputs=[
        gr.Textbox(label="Question (optional)", placeholder="Type your question here, or leave blank for a summary."),
        gr.Textbox(lines=10, label="Input Text", placeholder="Type the text here..."),
        gr.Slider(minimum=1, maximum=20, step=1, label="Number of Keywords"),
        gr.Radio(choices=["Speed", "Accuracy"], label="Choose Quality"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Creativity Level (Temperature)")
    ],
    outputs=[
        gr.Dataframe(label="Named Entities"),
        gr.Dataframe(label="Keywords Ranked by Occurrence"),
        gr.Textbox(label="AI-Generated Description"),
    ],
    title="NLP Text Analysis with AI-Generated Description",
    description="""Created by Chinedu Ike-Anyanwu, this NLP tool is designed to analyze text, extract and rank keywords, and generate descriptive summaries or answers to specific questions. It leverages advanced AI models to provide insights into the provided text, making it a valuable resource for faculty and students of Rowan University.

- **Question (optional):** You can ask a specific question related to the text, or leave this blank to get a summary based on the input text and the most common words.
- **Input Text:** Place the text you want analyzed here.
- **Number of Keywords:** Select how many keywords to focus on. This influences the focus of the AI's analysis and generated description.
- **Choose Quality:** Choose 'Speed' for faster, less detailed responses using 'gpt-3.5-turbo-instruct-0914'. Choose 'Accuracy' for more detailed responses using 'gpt-3.5-turbo-16k-0613'.
- **Creativity Level (Temperature):** Adjust the creativity of the AI's response. Lower values produce more predictable, conservative outputs; higher values encourage creativity and novelty.

This tool is intended for use by the faculty or students of Rowan University as part of educational and research activities."""
)

# Launch the interface
iface.launch()
#interface.launch()
