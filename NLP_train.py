import gradio as gr
import spacy
from collections import Counter
from transformers import pipeline
import pandas as pd  # Import pandas
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure you have the necessary models and libraries installed:
# pip install gradio spacy transformers scikit-learn pandas
# python -m spacy download en_core_web_trf

# Load spaCy model with text categorization
nlp = spacy.load("en_core_web_trf")

# Initialize summarization pipeline
summarization = pipeline("summarization", model="facebook/bart-large-cnn")

def analyze_text(text):
    """
    Analyzes text using spaCy and TF-IDF, returning relevant information.
    """
    doc = nlp(text)

    # Named entity recognition (NER)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Keyword extraction using TF-IDF
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()[tf_idf_matrix.toarray()[0].argsort()[-10:][::-1]]

    return {
        "Named Entities": entities,
        "Keywords": keywords,
    }

def generate_description(text, keywords):
    """
    Uses summarization pipeline to create a description based on text and keywords.
    """
    prompt = f"{text}\nKeywords: {', '.join(keywords)}"
    response = summarization(prompt, max_length=150, min_length=50, truncation=True)
    return response[0]["summary_text"]

def NLP_analysis(text):
    """
    Main function for NLP analysis and description generation.
    """
    analysis_result = analyze_text(text)
    description = generate_description(text, analysis_result["Keywords"])
    analysis_df = pd.DataFrame(
        list(analysis_result.items()), columns=["Category", "Value"]
    )
    return analysis_df, description

# Define Gradio interface
interface = gr.Interface(
    fn=NLP_analysis,
    inputs="text",
    outputs=[gr.Dataframe(label="Analysis Results", type = "pandas"), gr.Textbox(label="AI-Generated Description")],
    title="Advanced NLP Text Analysis",
    description="This tool analyzes text for named entities and keywords using spaCy and TF-IDF, then generates a descriptive summary using a summarization model.",
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()
