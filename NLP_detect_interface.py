import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import gradio as gr

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def get_word_frequencies(text):
    """Process the input text, remove stopwords, and return sorted word frequencies."""
    text = re.sub(r'\W+', ' ', text.lower())  # Lowercase and remove non-alphabetic characters
    tokens = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word not in stop_words]  # Remove stopwords
    word_frequencies = Counter(words)  # Count word frequencies
    sorted_words = word_frequencies.most_common()  # Sort by frequency
    return sorted_words

def describe_words(words):
    """Mock function to return descriptions for the top 5 words."""
    descriptions = {word: f"Description of {word}." for word, _ in words[:5]}
    return descriptions

def summarize_text(text):
    """Mock function to return a summary of the text."""
    return "Summary of the input text."

def process_text(text):
    """Process the text and return word frequencies, descriptions of the top 5 words, and a summary."""
    sorted_words = get_word_frequencies(text)
    word_descriptions = describe_words(sorted_words)
    summary = summarize_text(text)
    sorted_word_list = [f"{word}: {freq}" for word, freq in sorted_words]
    return "\n".join(sorted_word_list), "\n".join([f"{word}: {desc}" for word, desc in word_descriptions.items()]), summary

# Setting up the Gradio interface
NLP_detect = gr.Blocks()

with NLP_detect:
    gr.Markdown("### NLP Common Word Detector and Analyzer")
    with gr.Row():
        text_input = gr.Textbox(label="Enter text", placeholder="Type your text here...", lines=6)
        word_freq_output = gr.Textbox(label="Word Frequencies", placeholder="Word frequencies will be displayed here...", lines=10)
        word_desc_output = gr.Textbox(label="Top 5 Word Descriptions", placeholder="Descriptions of the top 5 words will be displayed here...", lines=6)
        summary_output = gr.Textbox(label="Text Summary", placeholder="A summary of the text will be displayed here...", lines=3)
        
    text_input.change(process_text, inputs=[text_input], outputs=[word_freq_output, word_desc_output, summary_output])

demo = NLP_detect.launch()
