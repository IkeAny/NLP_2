import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
import spacy
from spacy.lang.en import English
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK resources (uncomment if not downloaded yet)
# nltk.download('punkt')
# nltk.download('stopwords')
nltk.download('vader_lexicon')

# Download spaCy English model
spacy.cli.download("en_core_web_sm")

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    # Tokenization and lemmatization using spaCy
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return tokens

def analyze_survey_responses(survey_responses):
    # Preprocess each survey response
    preprocessed_responses = [preprocess_text(response) for response in survey_responses]

    # Flatten the list of lists
    all_tokens = [token for sublist in preprocessed_responses for token in sublist]

    # Create frequency distribution
    freq_dist = FreqDist(all_tokens)

    # Display the most common words
    print("Most common words in survey responses:")
    print(freq_dist.most_common(10))

    # Generate and display a Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dist)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # Perform sentiment analysis on the survey responses
    perform_sentiment_analysis(survey_responses)

def perform_sentiment_analysis(text_data):
    sid = SentimentIntensityAnalyzer()
    sentiments = [sid.polarity_scores(text) for text in text_data]
    print("\nSentiment analysis results:")
    for i, sentiment in enumerate(sentiments):
        print(f"Survey Response {i + 1}: {sentiment}")

if __name__ == "__main__":
    # Get survey responses from users
    survey_responses = []
    while True:
        response = input("Enter survey response (or type 'done' when finished): ")
        if response.lower() == 'done':
            break
        survey_responses.append(response)

    # Call the analysis functions
    analyze_survey_responses(survey_responses)
