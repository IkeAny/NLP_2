import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from collections import Counter
import openai
import sys

# Assuming the necessary paths for your project
sys.path.append('NLP_interface')
sys.path.append('NLP_program')
sys.path.append('NLP_MachineLearning')
sys.path.append('./')

# Load the custom-trained spaCy model for sentiment analysis
try:
    nlp_sentiment = spacy.load("model_sentiment")  # Adjust this path to your model
except IOError:
    print("Custom sentiment model not found.")
    nlp_sentiment = None

# Load a general-purpose spaCy model for other NLP tasks
try:
    nlp_general = spacy.load("en_core_web_trf")
except IOError:
    spacy.cli.download("en_core_web_trf")
    nlp_general = spacy.load("en_core_web_trf")

def analyze_text(text, num_words=10):
    sentiment_score = {}
    if nlp_sentiment:
        doc_sentiment = nlp_sentiment(text)
        sentiment_score = doc_sentiment.cats  # Assuming .cats for sentiment scores
    
    # Perform Named Entity Recognition using a general-purpose model
    doc_general = nlp_general(text)
    entities = [(ent.text, ent.label_) for ent in doc_general.ents][:num_words]
    entities_df = pd.DataFrame(entities, columns=["Entity", "Type"])
    
    # TF-IDF and Keyword Extraction
    custom_stop_words = set(ENGLISH_STOP_WORDS).union({"you", "for", "like", "got", "going", "said", "thing", 
                                                       "i'm","me","say","hey","bit","say,","hey,","little","you'd"})
    vectorizer = TfidfVectorizer(stop_words=list(custom_stop_words))
    tf_idf_matrix = vectorizer.fit_transform([text])
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = tf_idf_matrix.toarray()[0].argsort()[::-1]
    
    top_keywords = feature_array[tfidf_sorting][:num_words]
    filtered_text = ' '.join([word for word in text.lower().split() if word not in custom_stop_words])
    word_counts = Counter(filtered_text.split()).most_common(num_words)
    keywords_df = pd.DataFrame(word_counts, columns=["Keyword", "Count"])
    
    return entities_df, keywords_df, sentiment_score

def generate_description(question, text, keywords, quality, temperature, api_key):
    """
    Generates a description using OpenAI's GPT-3.5 Turbo model based on the input text,
    incorporating the provided question (if any), focusing on the important keywords,
    and adjusting for quality and temperature. The response is structured to emphasize a third-person narrative, starting with 'students'.
    """
    openai.api_key = api_key

    keywords_str = ", ".join(keywords)
    
    # Update the introduction to explicitly request a third-person perspective
    introduction = "Write a response in the third person, focusing on what students think or observe."
    
    if question.strip():
        prompt = f"{introduction} They consider the question: '{question}' based on the text: '{text}' and considering these keywords: {keywords_str}."
    else:
        prompt = f"{introduction} They summarize the following text, making sure to emphasize these keywords: {keywords_str}. Text: '{text}'"
    
    model = "gpt-3.5-turbo-instruct" if quality == "Speed" else "gpt-3.5-turbo-instruct-0914"
    max_tokens = 350 if quality == "Speed" else 500

    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return response['choices'][0]['text'].strip()
