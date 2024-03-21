import gradio as gr
import spacy
from collections import Counter
from transformers import pipeline
import pandas as pd  # Import pandas
from sklearn.feature_extraction.text import TfidfVectorizer


# Download spaCy model with text categorization (first time only)
nlp = spacy.load("en_core_web_trf")

summarization = pipeline("summarization", model="facebook/bart-base")


def analyze_text(text):
  """
  Analyzes text using spaCy and TF-IDF, returning relevant information.
  """
  if isinstance(text, list):
    text = text[0]  # Assuming the first element is the text (adjust if needed)
  doc = nlp(text)

  # Sentiment analysis with spaCy v3 textcat component (if applicable)
  try:
      sentiment = doc.cats["sentiment"]  # Access sentiment score directly
  except KeyError:
      sentiment = None  # Set sentiment to None if textcat is not available

  # Named entity recognition (NER)
  entities = [(ent.text, ent.label_) for ent in doc.ents]

  # Keyword extraction using TF-IDF (external library)
  vectorizer = TfidfVectorizer()
  tf_idf_matrix = vectorizer.fit_transform([text])
  keywords = vectorizer.get_feature_names_out()[tf_idf_matrix.toarray()[0].argsort()[-10:][::-1]]

  return {
      "Sentiment": sentiment,
      "Named Entities": entities,
      "Keywords": keywords,
  }


def generate_description(text, keywords):
  """
  Uses Bard large language model to create a description based on text and keywords.
  """
  prompt = f"Provide a creative and informative description of the following text, highlighting the key themes, insights, and named entities:\n{text}\nHere are some of the most relevant keywords: {', '.join(keywords)}"
  try:
      response = summarization(prompt, max_length=150, min_length=50)[0]
      return response["generated_text"]
  except KeyError:
      # Handle KeyError (Bard model unavailable or different response structure)
      return "Error: Unable to generate description at this time."


def NLP_analysis(text):  # Replace 'textbox' with the chosen input component
  """
  Main function for NLP analysis and description generation using Bard.
  """
  analysis_result = analyze_text(text)
  description = generate_description(text, analysis_result["Keywords"])

  # Option 1: Create a list of dictionaries for DataFrame (Gradio v0.8.9 or lower)
  # analysis_data = [
  #     {"Category": "Sentiment", "Value": analysis_result["Sentiment"]},
  #     {"Category": "Named Entities", "Value": analysis_result["Named Entities"]},
  #     {"Category": "Keywords", "Value": analysis_result["Keywords"]},
  # ]
  # analysis_df = pd.DataFrame(analysis_data)

  # Option 2: Use Pandas DataFrame directly (Gradio v0.8.10 or higher)
  analysis_df = pd.DataFrame(
      analysis_result.items(), columns=["Category", "Value"]
  )

  return analysis_df, analysis_result, description  # Provide three outputs


# Define Gradio interface outputs
outputs = [
    gr.Textbox(label="Enter Text"),
    gr.Dataframe(label="Analysis", type="pandas"),  # No value needed here
    gr.Textbox(label="AI-Generated Description"),
]

# Create Gradio interface
interface = gr.Interface(
    NLP_analysis,
    "textbox",
    title="NLP Text Analysis with AI Description (including NER)",
    description="Analyze text using spaCy's text categorization for sentiment (en_core_web_trf model), named entity recognition (NER), TF-IDF for keywords (using sklearn), then generate a description with Bard large language model.",
    outputs=outputs,
)

# Launch the interface
interface.launch()
