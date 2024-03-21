import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
import spacy
from spacy.lang.en import English
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download NLTK resources (uncomment if not downloaded yet)
nltk.download('punkt')
nltk.download('stopwords')

# Download spaCy English model
spacy.cli.download("en_core_web_sm")

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    # Tokenization and lemmatization using spaCy
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return tokens

def analyze_trends_and_themes(text_data):
    # Preprocess each interview text
    preprocessed_texts = [preprocess_text(text) for text in text_data]

    # Flatten the list of lists
    all_tokens = [token for sublist in preprocessed_texts for token in sublist]

    # Create frequency distribution
    freq_dist = FreqDist(all_tokens)

    # Display the most common words
    print("Most common words:")
    print(freq_dist.most_common(10))

    # Generate and display a Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dist)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def perform_sentiment_analysis(text_data):
    sid = SentimentIntensityAnalyzer()
    sentiments = [sid.polarity_scores(text) for text in text_data]
    print("Sentiment analysis results:")
    for i, sentiment in enumerate(sentiments):
        print(f"Interview {i + 1}: {sentiment}")

def perform_topic_modeling(text_data):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text_data)
    
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)

    feature_names = vectorizer.get_feature_names_out()

    print("Top words per topic:")
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-10 - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")

def perform_ner_analysis(text_data):
    for i, text in enumerate(text_data):
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"Named entities in Interview {i + 1}: {entities}")


if __name__ == "__main__":
    # Get user input for interview texts
    interview_texts = []
    while True:
        text = input("Enter interview text (or type 'done' when finished): ")
        if text.lower() == 'done':
            break
        interview_texts.append(text)

    # Call the analysis functions
    analyze_trends_and_themes(interview_texts)
    perform_sentiment_analysis(interview_texts)
    perform_topic_modeling(interview_texts)
    perform_ner_analysis(interview_texts)
