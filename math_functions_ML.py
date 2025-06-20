from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

random_state = 42

directory_path = f'saved_data/math_functions.nc'

# Load .nc file
ds = xr.open_dataset(directory_path)

gaussians = ds["Gaussians"].values
binary = ds["BinaryArr"].values
x = ds["x"].values
peak_idx = np.where(binary == 1)

# Split data into train and test sets
gauss_train, gauss_test, binary_train, binary_test = train_test_split(gaussians, binary, test_size=0.2, random_state=random_state)

# Scale the features (the intensity values)
scaler = StandardScaler() # create the scaler object 

gauss_train = scaler.fit_transform(gauss_train)
gauss_test = scaler.transform(gauss_test)

# Initialize the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(200,100, 50), max_iter=2000, random_state=random_state, verbose=True,
                   learning_rate_init=0.01,  # Default is 0.001; can increase or decrease
                    tol=1e-5,
                   early_stopping=False) # initialize MLPClassifier

# Train the MLP classifier
mlp.fit(gauss_train, binary_train)

binary_pred = mlp.predict(gauss_test) # make predictions

probs = mlp.predict_proba(gauss_test)
print(f'probs.shape is {probs.shape}')

print(binary_test.shape)

plt.figure()

for i in range(5):
    
    idx_true = np.where(binary_test[i] == 1)
    plt.scatter(x[idx_true], gauss_test[i][idx_true], marker='o', color='red')
    plt.plot(x, gauss_test[i], color='red')

    idx = np.where(binary_pred[i] == 1)
    print(idx)
    plt.scatter(x[idx], gauss_test[i][idx], marker='x', color='purple')
  
    
   
plt.show()