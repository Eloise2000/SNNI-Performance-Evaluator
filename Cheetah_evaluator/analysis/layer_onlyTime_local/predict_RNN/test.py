import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Activation, LSTM, TimeDistributed, Masking, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from kerastuner import HyperModel, RandomSearch
from joblib import dump

# Set variable here
show_server = 0 # 0: client; 1: server

if show_server:
    name = "server"
else:
    name = "client"

save_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/predict_RNN/server_result/"
y_pred_all = np.load(save_folder + "y_pred.npy")
y_test_all = np.load(save_folder + "y_test.npy")

def remove_consecutive_zeros(arr_test, arr_pred):
    filtered_segments = []
    zero_segment = []
    
    for val in arr_test:
        if val != 0:
            filtered_segments.extend(zero_segment)
            filtered_segments.append(val)
        else:
            zero_segment.append(val)
            if len(zero_segment) >= 3: break
    
    if len(zero_segment) < 3:
        filtered_segments.extend(zero_segment)

    n = len(filtered_segments)
    filtered_test = np.array(filtered_segments).reshape(-1, 1)
    filtered_pred = arr_pred[:n]
    return filtered_test, filtered_pred

for i in range(len(y_pred_all)):
    filtered_test, filtered_pred = remove_consecutive_zeros(y_test_all[i], y_pred_all[i])
    mae = mean_absolute_error(filtered_test, filtered_pred)
    print("Mean Absolute Error (MAE):", mae)