import os
import json
import subprocess
import itertools
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re

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
        "python", "test_model.py",
        "--epochs", str(epochs),
        "--fine_tune_epochs", str(fine_tune_epochs),
        "--learning_rate", str(learning_rate),
        "--fine_tune_at", str(fine_tune_at)
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    # print("Subprocess Output:")
    # print(result.stdout)  # Add this line to see the output
    
    accuracy = None
    f1_score_prey = None

    # Parse JSON output

    for line in result.stdout.strip().split('\n'):
        accuracy_match = re.search(r'{"val_accuracy":', line)
        if accuracy_match:
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

    return accuracy, f1_score_prey

# Run the optimization
def find_optimal_parameters(epochs_range, fine_tune_epochs_range, learning_rates, fine_tune_at_range, num_runs):
    all_results = []

    parameter_combinations = itertools.product(epochs_range, fine_tune_epochs_range, learning_rates, fine_tune_at_range)
    total_combinations = (len(epochs_range) *
                          len(fine_tune_epochs_range) *
                          len(learning_rates) *
                          len(fine_tune_at_range) *
                          num_runs)
    current_iter = 0

    for epochs, fine_tune_epochs, learning_rate, fine_tune_at in parameter_combinations:
        accuracies = []
        f1_scores = []

        for _ in range(num_runs):
            current_iter += 1
            print("Starting iteration ", current_iter, "out of ", total_combinations, "...")
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

    import pandas as pd
    import seaborn as sns

    # Convert results to a DataFrame for easier manipulation
    df = pd.DataFrame(results)

    # Set up the plot style
    sns.set(style="whitegrid", palette="muted")
    plt.figure(figsize=(12, 8))

    # Check which parameters have multiple values
    varying_params = [param for param in ['epochs', 'fine_tune_epochs', 'learning_rate', 'fine_tune_at']
                      if df[param].nunique() > 1]

    if not varying_params:
        print("No varying parameters found. Plotting F1 Score vs Epochs.")
        sns.lineplot(data=df, x='epochs', y='avg_f1_score_prey', marker='o')
    else:
        # Decide which parameters to use for hue and style
        hue_param = varying_params[0]  # Use the first varying parameter for hue
        style_param = varying_params[1] if len(varying_params) > 1 else None  # Second for style if available

        # Create the line plot with hue and style
        sns.lineplot(data=df, x='epochs', y='avg_f1_score_prey',
                     hue=hue_param, style=style_param, markers=True, dashes=False)

        plt.legend(title='Parameters', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel("Epochs")
    plt.ylabel("Average F1 Score (Prey)")
    plt.title("Model Performance Across Parameter Combinations")
    plt.tight_layout()
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
