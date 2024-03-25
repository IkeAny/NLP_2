**NLP for Survey Response Detection**

Created by Chinedu Ike-Anyanwu

This NLP (Natural Language Processing) tool is designed to analyze text data from surveys, extract and rank keywords, and generate summaries or answers to specific questions. It leverages advanced AI models to provide insights into survey responses, making it a valuable resource for faculty and students of Rowan University.

Features:

Question (optional): Ask a specific question related to the text for a focused analysis. Leave blank for a general summary.
Input Text: Paste the text you want to analyze here (e.g., survey responses).
Number of Keywords: Select the number of keywords to focus on, influencing the analysis and description.
Choose Quality: Choose between 'Speed' for faster, less detailed responses or 'Accuracy' for more detailed responses using different AI models.
Creativity Level (Temperature): Adjust the creativity of the AI's response. Lower values provide conservative outputs, while higher values encourage novelty.
Intended Use:

This tool is intended for educational and research activities by Rowan University faculty and students.

Local Setup:

This tool requires Python and specific libraries. Here's how to set it up locally:

Prerequisites:

Anaconda: https://www.anaconda.com/

Visual Studio Code: https://code.visualstudio.com/
Installation:

Install Anaconda following their instructions.

Download and install Visual Studio Code.

Open Anaconda Prompt.

Create a new environment for this project using conda create -n yourenvname python=3.8 (replace yourenvname with your desired name).

Activate the environment: conda activate yourenvname

Install required libraries:

CPU only: pip3 install torch torchvision torchaudio

GPU: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Download the requirements.txt file (included in the repository).

In Anaconda Prompt, navigate to the folder containing the requirements.txt file using cd yourpath (replace yourpath with the actual path).

Install additional dependencies listed in requirements.txt: pip3 install -r requirements.txt

Running the code:

Once the setup is complete, you can run the code using your preferred Python IDE or from the command line within the activated environment.

Disclaimer:

This is a basic guide. Refer to the specific documentation of Anaconda, Visual Studio Code, PyTorch, and other libraries for more detailed installation instructions and troubleshooting.
