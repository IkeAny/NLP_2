import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import gradio as gr
from spacy.lang.en import English
from PyPDF2 import PdfFileReader  # You may need to install this package

# Download spaCy English model
# spacy.cli.download("en_core_web_sm")

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return tokens

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PdfFileReader(file)
        text = ''
        for page_num in range(pdf_reader.numPages):
            text += pdf_reader.getPage(page_num).extractText()
        return text

def analyze_trends_and_themes(text_data):
    preprocessed_texts = [preprocess_text(text) for text in text_data]
    all_tokens = [token for sublist in preprocessed_texts for token in sublist]
    freq_dist = FreqDist(all_tokens)

    print("Most common words:")
    print(freq_dist.most_common(10))

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dist)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig("wordcloud.png")
    plt.close()

def perform_topic_modeling(text_data):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text_data)

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)

    feature_names = vectorizer.get_feature_names_out()

    top_words_per_topic = []
    print("Top words per topic:")
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-10 - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_words_per_topic.extend(top_words)
        print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")

    return top_words_per_topic


def display_merged_topics(top_words_per_topic):
    if isinstance(top_words_per_topic, str):
        return top_words_per_topic

    word_counts = FreqDist(top_words_per_topic)
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    print("\nMerged topics with ranks:")
    for rank, (word, count) in enumerate(sorted_words, start=1):
        print(f"{rank}. {word} (Count: {count})")

    html_list = "<ul>"
    for rank, (word, count) in enumerate(sorted_words, start=1):
        html_list += f"<li>{rank}. {word} (Count: {count})</li>"
    html_list += "</ul>"

    return html_list


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# ...

def analyze_and_display(text_input, file_input):
    # Your existing code for processing and analysis
    text_data = text_input if text_input else read_file(file_input.name)

    analyze_trends_and_themes([text_data])
    top_words_per_topic = perform_topic_modeling([text_data])
    merged_topics = display_merged_topics(top_words_per_topic)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
        FreqDist(preprocess_text(text_data)))
    wordcloud_pil = wordcloud.to_image()

    # Save the processed text to a file
    with open("processed_text.txt", "w", encoding="utf-8") as output_file:
        output_file.write(text_data)

    processed_text_file = gr.File("processed_text.txt", type='binary')

    # Return the Gradio components
    return gr.Image(wordcloud_pil), merged_topics, processed_text_file

if __name__ == "__main__":
    # Example usage for testing
    sample_input_text = "This is a sample input text for testing."
    analyze_and_display(sample_input_text, None)  # Use None for file_input when testing with direct text input
