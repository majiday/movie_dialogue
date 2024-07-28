from data_preprocessing import load_data, preprocess_text
from model_training import train_model, prepare_sequences
from text_generation.text_generation import generate_text
import config

# Load and preprocess data
data = load_data(config.DATA_PATH)
preprocessed_data = preprocess_text(data)

# Prepare data for training
input_sequences, max_sequence_len, tokenizer = prepare_sequences(preprocessed_data)

# Train the model
model = train_model(input_sequences, max_sequence_len, tokenizer)

# Generate text
seed_text = "There is something"
generated_text = generate_text(seed_text, 10, model, tokenizer, max_sequence_len)
print(generated_text)

# Save the model
model.save(config.MODEL_PATH)
