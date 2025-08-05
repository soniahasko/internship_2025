import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Open and read file
file = 'saved_results_only_analytical_9.nc'
ds = xr.open_dataset(file)

true_y = ds['true_y'].values
pred_y = ds['predicted_y'].values

def calculate_accuracy(true_labels, predicted_labels):
    correct = 0
    for true, pred in zip(true_labels, predicted_labels):
        if abs(true - pred) <= 0.005:
            correct += 1

    total_elements = len(true_labels)

    if total_elements == 0:
        return 0.0  # Handle the case of empty arrays

    accuracy = correct / total_elements
    return accuracy

all_scores = []

for i in range(true_y.shape[0]):
    y_true_i = np.array(true_y[i])
    y_pred_i = np.array(pred_y[i])

    accuracy_score = calculate_accuracy(y_true_i.flatten(), y_pred_i.flatten())

    all_scores.append(accuracy_score)

print(np.mean(all_scores))
print(np.std(all_scores))