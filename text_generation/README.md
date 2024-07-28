# Friends TV Show Text Generation

This repository contains a text generation model based on the dialogue from the TV show "Friends". The model employs LSTM (Long Short-Term Memory) networks to simulate the speech patterns of the main characters, enabling it to generate new dialogues that resemble the style of the show.

## Project Structure

- **`main.ipynb`**: The Jupyter notebook serves as the primary interface where the entire process is orchestrated. It calls functions from other modules to perform tasks like data loading, preprocessing, model training, and text generation.
- **`data_preprocessing.py`**: This module handles all the data preprocessing tasks, including loading and cleaning the data to prepare it for model training.
- **`model_training.py`**: Contains functions related to setting up, training, and evaluating the LSTM model.
- **`text_generation.py`**: Provides functionality to generate text using the trained model.
- **`config.py`**: Configuration file holding parameters like file paths and model settings which are used across the project.
- **`requirements.txt`**: Lists all the necessary Python packages required to run the project.

## Setup

To run this project, ensure that you have Python installed, and then install the necessary dependencies:

```bash
pip install -r requirements.txt
