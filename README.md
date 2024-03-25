**NLP for Survey Response Detection**

***Created by Chinedu Ike-Anyanwu***

This NLP (Natural Language Processing) tool is designed to analyze text data from surveys, extract and rank keywords, and generate summaries or answers to specific questions. It leverages advanced AI models to provide insights into survey responses, making it a valuable resource for faculty and students of Rowan University.

**Features**:

1. Question (optional): Ask a specific question related to the text for a focused analysis. Leave blank for a general summary.
2. Input Text: Paste the text you want to analyze here (e.g., survey responses).
3. Number of Keywords: Select the number of keywords to focus on, influencing the analysis and description.
4. Choose Quality: Choose between 'Speed' for faster, less detailed responses or 'Accuracy' for more detailed responses using different AI models.
5. Creativity Level (Temperature): Adjust the creativity of the AI's response. Lower values provide conservative outputs, while higher values encourage novelty.

**Intended Use**:

This tool is intended for educational and research activities by Rowan University faculty and students.

**Local Setup**:

This tool requires Python and specific libraries. Here's how to set it up locally:

**Prerequisites**:

``Anaconda``: https://www.anaconda.com/

``Visual Studio Code``: https://code.visualstudio.com/

**Installation**:

1. Install Anaconda following their instructions.

2. Download and install Visual Studio Code.

3. Open Anaconda Prompt.

4. Create a new environment for this project using ``conda create -n yourenvname python=3.8`` (replace yourenvname with your desired name).

5. Activate the environment: ``conda activate yourenvname``

6. Install required libraries:

    CPU only: ``pip3 install torch torchvision torchaudio``

    GPU: ``pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118``

7. Download the requirements.txt file (included in the repository).

8. In Anaconda Prompt, navigate to the folder containing the requirements.txt file using ``cd yourpath`` (replace yourpath with the actual path).

9. Install additional dependencies listed in requirements.txt: ``pip3 install -r requirements.txt``

10. Running the code:

    Once the setup is complete, you can run the code using your preferred Python IDE or from the command line within the activated environment.

***Disclaimer***:

This is a basic guide. Refer to the specific documentation of Anaconda, Visual Studio Code, PyTorch, and other libraries for more detailed installation instructions and troubleshooting.
