from data_preprocessing import load_data, preprocess_text, vectorize_text, prepare_data_nn
from model_training import train_naive_bayes, evaluate_model, train_nn_model, save_model
import config

# Load and preprocess data
data = load_data(config.DATA_PATH)
preprocessed_data = preprocess_text(data)

# Train and evaluate Naive Bayes Model
X_train_vec, X_test_vec, y_train, y_test = vectorize_text(preprocessed_data)
nb_model = train_naive_bayes(X_train_vec, y_train)
evaluate_model(nb_model, X_test_vec, y_test)

# Train and evaluate Neural Network Model
X_train, X_test, y_train, y_test, tokenizer = prepare_data_nn(data)
nn_model = train_nn_model(X_train, y_train, tokenizer)
evaluate_model(nn_model, X_test, y_test)
save_model(nn_model, config.MODEL_PATH_NN)
