from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from training_data_simulation import *
random_state = 42

def __init__(self):
    
def prepare_data(intensities, binary_peaks):
    intensities_train, intensities_test, binary_peaks_train, binary_peaks_test = train_test_split(intensities, binary_peaks, test_size=0.33,
                                                                                                  random_state=random_state)
    
    scaler = StandardScaler() #create the scaler object 

    intensities_train = scaler.fit_transform(intensities_train) # fit the scaler based on the training data
    intensities_test = scaler.transform(intensities_test) # transform test data

def initialize():
    mlp = MLPClassifier(hidden_layer_sizes=(500,100,50), max_iter=1000, random_state=42) # initialize MLPClassifier

def train(intensities_train, binary_peaks_train):
    mlp.fit(intensities_train, binary_peaks_train) # train MLPClassifier

def predict(intensities_test):
    b_peaks_pred = mlp.predict(intensities_test) # make predictions

def evaluate_acc(binary_peaks_test, b_peaks_pred):
    accuracy = accuracy_score(binary_peaks_test, b_peaks_pred)

if name == "__main__":

    # get intensities, binary_peaks

    