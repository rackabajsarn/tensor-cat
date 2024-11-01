import os
import json
import subprocess
import numpy as np
from tensorflow.keras.models import load_model
import argparse

# Directory for saving optimization results
OPTIMIZATION_DIR = 'optimization'
os.makedirs(OPTIMIZATION_DIR, exist_ok=True)

# File paths
PARAM_RESULTS_FILE = os.path.join(OPTIMIZATION_DIR, 'parameter_results.json')
BEST_MODEL_FILE = os.path.join(OPTIMIZATION_DIR, 'best_model.keras')
BEST_PARAMS_FILE = os.path.join(OPTIMIZATION_DIR, 'best_params.json')

# Load the best parameters from the JSON file
def load_best_parameters():
    if not os.path.exists(PARAM_RESULTS_FILE):
        raise FileNotFoundError("Parameter results file not found. Run find_optimal_parameters.py first.")

    with open(PARAM_RESULTS_FILE, 'r') as f:
        data = json.load(f)

    # Find the best parameters based on the highest F1 score for prey
    best_params = max(data["results"], key=lambda x: x["avg_f1_score_prey"])

    # Save best parameters to a separate file
    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump(best_params, f, indent=4)

    return best_params

# Run train_model.py with the best parameters
def run_training(epochs, fine_tune_epochs, learning_rate, fine_tune_at):
    command = [
        "python", "train_model.py",
        "--epochs", str(epochs),
        "--fine_tune_epochs", str(fine_tune_epochs),
        "--learning_rate", str(learning_rate),
        "--fine_tune_at", str(fine_tune_at)
    ]
    subprocess.run(command)

# Find the optimal model by training multiple times with the best parameters
def find_optimal_model(num_runs):
    best_params = load_best_parameters()

    # Extract parameters
    epochs = best_params["epochs"]
    fine_tune_epochs = best_params["fine_tune_epochs"]
    learning_rate = best_params["learning_rate"]
    fine_tune_at = best_params["fine_tune_at"]

    best_f1_score = 0

    # Train the model multiple times to find the best version
    for i in range(num_runs):  # Number of times to run the training
        print(f"Training iteration {i + 1}/{num_runs} with the best parameters...")
        run_training(epochs, fine_tune_epochs, learning_rate, fine_tune_at)

        # Load the model after training
        model_path = os.path.join('model', 'best_model.keras')
        model = load_model(model_path)

        # Evaluate the model on the validation set
        val_accuracy, f1_score_prey = evaluate_model(model)

        # Keep track of the best model
        if f1_score_prey > best_f1_score:
            best_f1_score = f1_score_prey
            # Save the best model to the optimization directory
            model.save(BEST_MODEL_FILE)
            print(f"New best model saved with F1 Score: {f1_score_prey}")

# Evaluate the model and return validation accuracy and F1 score for prey
def evaluate_model(model):
    # Load validation dataset (assuming validation dataset is available as NumPy arrays)
    val_images = np.load(os.path.join(OPTIMIZATION_DIR, 'val_images.npy'))
    val_labels = np.load(os.path.join(OPTIMIZATION_DIR, 'val_labels.npy'))

    val_predictions = model.predict(val_images)
    val_pred_labels = np.argmax(val_predictions, axis=1)

    # Calculate metrics
    from sklearn.metrics import classification_report
    report = classification_report(val_labels, val_pred_labels, output_dict=True, zero_division=0)
    val_accuracy = report['accuracy']
    f1_score_prey = report['prey']['f1-score']

    return val_accuracy, f1_score_prey

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find the optimal model.')
    parser.add_argument('--num_runs', type=int, default=5,
                        help='Number of times to run the training.')
    args = parser.parse_args()

    find_optimal_model(args.num_runs)
