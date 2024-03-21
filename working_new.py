import gradio as gr
import nltk
from collections import Counter
import openai

# Replace with your OpenAI API key
openai.api_key = "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"


def analyze_text(text):
    """
    Analyzes text, separates speaker lines, and returns word ranking and description for each speaker.

    Args:
        text: The text to analyze.

    Returns:
        A tuple containing:
            - speaker1_text: Text spoken by speaker 1.
            - speaker2_text: Text spoken by speaker 2 (if applicable).
            - speaker1_counts: Counter object of word counts for speaker 1.
            - speaker1_ranks: List of tuples (word, count) for speaker 1, ranked by count.
            - speaker1_description: Description for speaker 1's text.
            - speaker2_counts: Counter object of word counts for speaker 2 (if applicable).
            - speaker2_ranks: List of tuples (word, count) for speaker 2 (if applicable).
            - speaker2_description: Description for speaker 2's text (if applicable).
            - speaker1_name: Name of speaker 1 (based on user input or logic).
            - speaker2_name: Name of speaker 2 (based on user input or logic, optional).
    """

    # Preprocess text (lowercase, tokenize, remove stopwords)
    tokens = [word.lower() for word in nltk.word_tokenize(text) if word not in nltk.corpus.stopwords.words("english")]

    # Implement speaker diarization logic here (replace with your chosen method)
    # This example uses a basic approach, consider exploring more robust techniques.
    speaker_labels = separate_speakers(tokens)

    # Separate speaker lines based on identified labels
    speaker_texts = {}
    for i, token in enumerate(tokens):
        speaker = speaker_labels[i]
        if speaker not in speaker_texts:
            speaker_texts[speaker] = []
        speaker_texts[speaker].append(token)

    # Analyze each speaker's text
    speaker_results = {}
    for speaker, words in speaker_texts.items():
        speaker_counts = Counter(words)
        speaker_ranks = sorted(speaker_counts.items(), key=lambda item: item[1], reverse=True)
        speaker_prompt = f"Write a short description of the following text:\n{' '.join(words)}"
        speaker_response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=speaker_prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        speaker_description = speaker_response.choices[0].text.strip()
        speaker_results[speaker] = (speaker_counts, speaker_ranks, speaker_description)

    # Extract and return results
    speaker1_text, speaker2_text = None, None
    speaker1_counts, speaker1_ranks, speaker1_description = None, None, None
    speaker2_counts, speaker2_ranks, speaker2_description = None, None, None
    speaker1_name, speaker2_name = None, None

    if "speaker1" in speaker_results:
        speaker1_counts, speaker1_ranks, speaker1_description = speaker_results["speaker1"]
        speaker1_text = " ".join(speaker_texts["speaker1"])
        speaker1_name = "Speaker 1"  # Replace with actual name retrieval based on user input or logic

    if "speaker2" in speaker_results:
        speaker2_counts, speaker2_ranks, speaker2_description = speaker_results["speaker2"]
        speaker2_text = " ".join(speaker_texts["speaker2"])
        speaker2_name = "Speaker 2" if speaker2_text else None  # Handle optional speaker 2

    return (
        speaker1_text,
        speaker2_text,
        speaker1_counts,
        speaker1_ranks,
        speaker1_description,
        speaker2_counts,
        speaker2_ranks,
        speaker2_description,
        speaker1_name,
        speaker2_name,
    )

def separate_speakers(tokens):
    # Implement your speaker diarization logic here
    # This example simply assigns labels based on even/odd indices (not accurate)
    speaker_labels = [("speaker1" if i % 2 == 0 else "speaker2") for i in range(len(tokens))]
    return speaker_labels


interface = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(lines=10, label="Enter your text (separate speaker turns with dashes '-'):"),
    outputs=[
        gr.Textbox(label="Speaker 1 Text"),
        gr.Textbox(label="Speaker 2 Text (if applicable)"),
        gr.JSON(label="Speaker 1 Word Counts"),
        gr.List(label="Speaker 1 Word Ranks"),
        gr.Textbox(label="Speaker 1 Description"),
        gr.JSON(label="Speaker 2 Word Counts (if applicable)"),
        gr.List(label="Speaker 2 Word Ranks (if applicable)"),
        gr.Textbox(label="Speaker 2 Description (if applicable)"),
        gr.Textbox(label="Enter Speaker 1 Name"),
        gr.Textbox(label="Enter Speaker 2 Name (if applicable)"),
    ],
    title="NLP Analysis Tool with Speaker Differentiation",
    description="Analyze text, differentiate speaker lines, and get word ranking and description for each speaker. Assign speaker names for identification.",
    layout="tabs",
    theme="default",
)


@interface.launch(debug=True)
def launch(outputs):
    speaker1_text, speaker2_text, *other_outputs = outputs

    # Access speaker names after user input
    speaker_names = [outputs[-2], outputs[-1]]

    with gr.Tabs() as tabs:
        with gr.TabItem("Speaker 1"):
            gr.Markdown("**Text:**")
            gr.Textbox(label="", value=speaker1_text)
            gr.Markdown("**Word Counts:**")
            gr.JSON(label="", value=other_outputs[0])  # Speaker 1 word counts
            gr.Markdown("**Word Ranks:**")
            gr.List(label="", value=list(other_outputs[1]))  # Convert ranks to list
            gr.Markdown("**Description:**")
            gr.Textbox(label="", value=other_outputs[2])  # Speaker 1 description
        with gr.TabItem("Speaker 2 (if applicable)"):
            gr.Markdown("**Text:**")
            gr.Textbox(label="", value=speaker2_text) if speaker2_text else gr.Markdown("No speaker 2 identified.")
            gr.Markdown("**Word Counts:**")
            gr.JSON(label="", value=other_outputs[5]) if speaker2_text else gr.Markdown("N/A")  # Handle empty speaker 2
            gr.Markdown("**Word Ranks:**")
            gr.List(label="", value=list(other_outputs[6]) if speaker2_text else [])  # Handle empty ranks
            gr.Markdown("**Description:**")
            gr.Textbox(label="", value=other_outputs[7]) if speaker2_text else gr.Markdown("N/A")  # Handle empty speaker 2

# Add speaker name input handling
        if speaker_names[0]:
                gr.Markdown(f"**Speaker Name:** {speaker_names[0]}")
        if speaker_names[1]:
                gr.Markdown(f"**Speaker Name:** {speaker_names[1]}")

interface.launch()