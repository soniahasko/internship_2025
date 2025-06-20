from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import entropy

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
# print(f'probs.shape is {probs.shape}')
# print(binary_test.shape)
two_classprobs = np.array([[i,1-i] for i in probs]) # 1-i will give probability for the 0 class
two_classprobs = two_classprobs.reshape((probs.shape[0],probs.shape[1],2)) # reshape to (num_samples, num_features, num_classes)

confidences = []
for i in range(two_classprobs.shape[0]):
    confidence = np.max(two_classprobs[i], axis=1) # returns max value for each of the features
    confidences.append(confidence)
confidences = np.array(confidences)

uncertainties = []
for i in range(two_classprobs.shape[0]):
    uncertainties.append(entropy(two_classprobs[i].T))
uncertainties = np.array(uncertainties)

plt.figure()
plt.scatter(x, uncertainties)
plt.scatter(x, confidences)

plt.figure()

for i in range(5):
    
    idx_true = np.where(binary_test[i] == 1)
    plt.scatter(x[idx_true], gauss_test[i][idx_true], marker='o', color='red')
    plt.plot(x, gauss_test[i], color='red')

    idx = np.where(binary_pred[i] == 1)
    print(idx)
    plt.scatter(x[idx], gauss_test[i][idx], marker='x', color='purple')
  
    
   
plt.show()