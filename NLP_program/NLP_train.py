import sys
import spacy
from spacy.training import Example
from spacy.util import minibatch
import random
from sklearn.model_selection import train_test_split
import pandas as pd
import os

sys.path.append('NLP_interface')
sys.path.append('NLP_program')
sys.path.append('NLP_MachineLearning')
sys.path.append('./')

def load_data(filepath='data/sentiment2_data.csv'):
    df = pd.read_csv(filepath)
    # Correctly format 'cats' as {'positive': True, 'negative': False} directly
    df['cats'] = df.label.apply(lambda x: {'positive': x == 'positive', 'negative': x == 'negative'})
    # Splitting data into training and validation sets
    data = [(row['text'], row['cats']) for index, row in df.iterrows()]
    return train_test_split(data, test_size=0.2, random_state=42)

def train_nlp_model(train_data, valid_data, model_path='model_sentiment'):
    nlp = spacy.blank('en')
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.add_pipe('textcat', last=True)
    textcat.add_label('positive')
    textcat.add_label('negative')
    
    optimizer = nlp.begin_training()
    for i in range(5):  # Number of iterations
        random.shuffle(train_data)
        losses = {}
        for batch in minibatch(train_data, size=8):
            examples = []
            for text, cats in batch:
                # Ensure the 'cats' dictionary directly contains the labels and boolean values
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, {"cats": cats})
                examples.append(example)
            nlp.update(examples, drop=0.2, losses=losses)
        print(f'Iteration {i}, Losses: {losses}')
    
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    nlp.to_disk(model_path)
    print(f'Model saved to {model_path}')


def evaluate_model(nlp, valid_data):
    correct_predictions = 0
    for text, cats in valid_data:
        doc = nlp(text)
        # Calculate accuracy
        if (doc.cats['positive'] > 0.5 and cats['positive']) or (doc.cats['negative'] > 0.5 and cats['negative']):
            correct_predictions += 1
    accuracy = correct_predictions / len(valid_data)
    print(f'Validation Accuracy: {correct_predictions / len(valid_data)}')

