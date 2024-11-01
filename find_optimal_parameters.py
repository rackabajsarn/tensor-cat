import os
import json
import subprocess
import itertools
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Directory for saving optimization results
OPTIMIZATION_DIR = 'optimization'
os.makedirs(OPTIMIZATION_DIR, exist_ok=True)

# File paths
PARAM_RESULTS_FILE = os.path.join(OPTIMIZATION_DIR, 'parameter_results.json')
ACCURACY_PLOT_FILE = os.path.join(OPTIMIZATION_DIR, 'accuracy_plot.png')

# Function to save results to JSON file
def save_results(results):
    if os.path.exists(PARAM_RESULTS_FILE):
        with open(PARAM_RESULTS_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = {"results": []}

    data["results"].append(results)

    with open(PARAM_RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# Run train_model.py with specified parameters and return metrics
def run_training(epochs, fine_tune_epochs, learning_rate, fine_tune_at):
    command = [
        "python", "train_model.py",
        "--epochs", str(epochs),
        "--fine_tune_epochs", str(fine_tune_epochs),
        "--learning_rate", str(learning_rate),
        "--fine_tune_at", str(fine_tune_at)
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    # print("Subprocess Output:")
    # print(result.stdout)  # Add this line to see the output

    # Parse JSON output
    
    for line in result.stdout.strip().split('\n'):
        if line.startswith("{"):
            try:
                metrics = json.loads(line)
                accuracy = metrics.get("val_accuracy")
                f1_score_prey = metrics.get("f1_score")
                print(f"Extracted accuracy: {accuracy}")
                print(f"Extracted F1 score: {f1_score_prey}")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON output: {e}")
                print("Subprocess output was:")
                print(result.stdout)
                accuracy = None
                f1_score_prey = None

    return accuracy, f1_score_prey

# Run the optimization
def find_optimal_parameters(epochs_range, fine_tune_epochs_range, learning_rates, fine_tune_at_range, num_runs):
    all_results = []

    parameter_combinations = itertools.product(epochs_range, fine_tune_epochs_range, learning_rates, fine_tune_at_range)

    for epochs, fine_tune_epochs, learning_rate, fine_tune_at in parameter_combinations:
        accuracies = []
        f1_scores = []

        for _ in range(num_runs):
            accuracy, f1_score_prey = run_training(epochs, fine_tune_epochs, learning_rate, fine_tune_at)
            if accuracy is not None and f1_score_prey is not None:
                accuracies.append(accuracy)
                f1_scores.append(f1_score_prey)

        if accuracies and f1_scores:
            avg_accuracy = np.mean(accuracies)
            avg_f1_score = np.mean(f1_scores)

            result = {
                "epochs": epochs,
                "fine_tune_epochs": fine_tune_epochs,
                "learning_rate": learning_rate,
                "fine_tune_at": fine_tune_at,
                "avg_accuracy": avg_accuracy,
                "avg_f1_score_prey": avg_f1_score
            }
            save_results(result)
            all_results.append(result)

    # Plot the results
    plot_results(all_results)

# Plot model performance per combination
def plot_results(results):
    if not results:
        print("No results to plot.")
        return

    # Extract unique epochs and corresponding F1 scores
    epochs = [res["epochs"] for res in results]
    f1_scores = [res["avg_f1_score_prey"] for res in results]

    if not epochs or not f1_scores:
        print("No data available to plot.")
        return

    plt.plot(epochs, f1_scores, marker='o', label="F1 Score vs Epochs")

    plt.xlabel("Epochs")
    plt.ylabel("F1 Score (Prey)")
    plt.title("Model Performance")
    plt.legend()
    plt.savefig(ACCURACY_PLOT_FILE)
    plt.close()
    print(f"Accuracy plot saved to {ACCURACY_PLOT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize model parameters.')

    parser.add_argument('--epochs_range', nargs='+', type=int, required=True,
                        help='List of epochs values.')
    parser.add_argument('--fine_tune_epochs_range', nargs='+', type=int, required=True,
                        help='List of fine-tune epochs values.')
    parser.add_argument('--learning_rates', nargs='+', type=float, required=True,
                        help='List of learning rate values.')
    parser.add_argument('--fine_tune_at_range', nargs='+', type=int, required=True,
                        help='List of unfreeze layers values.')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of runs per parameter combination.')

    args = parser.parse_args()

    EPOCHS_RANGE = args.epochs_range
    FINE_TUNE_EPOCHS_RANGE = args.fine_tune_epochs_range
    LEARNING_RATES = args.learning_rates
    FINE_TUNE_AT_RANGE = args.fine_tune_at_range
    NUM_RUNS = args.num_runs

    find_optimal_parameters(EPOCHS_RANGE, FINE_TUNE_EPOCHS_RANGE, LEARNING_RATES, FINE_TUNE_AT_RANGE, NUM_RUNS)
