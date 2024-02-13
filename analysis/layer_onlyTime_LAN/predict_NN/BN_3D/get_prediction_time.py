# Tensorflow 2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from kerastuner import HyperModel, RandomSearch
import tensorflow as tf
from tensorflow.keras import layers
import time

show_server = 1
if show_server:
    name = "server"
    project_used ='230111-1'
else: 
    name = "client"
    project_used ='230111-1'

'''"BN1_C", "BN1_H", "BN1_W", "time_cost"'''
'''"BN2_C", "BN2_H", "BN2_W", "time_cost"'''

target_folder = "/home/eloise/eloise/script/analysis/layer_onlyTime_LAN/dataset/data_layer/"
layer_filepath_BN1 = target_folder + "BN1_onlyTime_" + name + ".csv"
layer_filepath_BN2 = target_folder + "BN2_onlyTime_3D_" + name + ".csv"
df_BN1 = pd.read_csv(layer_filepath_BN1, delimiter="\s+")
df_BN1 = df_BN1[df_BN1['time_cost'] > 0]
df_BN2 = pd.read_csv(layer_filepath_BN2, delimiter="\s+")
df_BN2 = df_BN2[df_BN2['time_cost'] > 0]

df_BN1 = df_BN1.rename(columns={'BN1_C': 'C', 'BN1_H': 'H', 'BN1_W': 'W'})
df_BN2 = df_BN2.rename(columns={'BN2_C': 'C', 'BN2_H': 'H', 'BN2_W': 'W'})
df = pd.concat([df_BN1, df_BN2], axis = 0, ignore_index=True)

x = df.loc[:, ['C', 'H', 'W']]
y = df['time_cost']

NN_folderPath = "/home/eloise/eloise/script/analysis/layer_onlyTime_LAN/predict_NN/"
best_model_BN = tf.keras.models.load_model(f"{NN_folderPath}BN_3D/tuner_results_{name}/{project_used}/best_model_BN_{name}.keras")

df_row = df.iloc[30]
x_single = df_row[["C", "H", "W"]].values.reshape(1, -1)
y_single = df_row['time_cost']

# Start prediction
start_time = time.time()
y_single_pred = best_model_BN.predict(x_single)
end_time = time.time()

# Calculate the time taken for prediction
prediction_time = end_time - start_time

print("Predicted Time:", y_single_pred)
print("Real Time:", y_single)
print("Diff Time:", y_single - y_single_pred)
print("Prediction Time:", prediction_time * 1000, "ms")