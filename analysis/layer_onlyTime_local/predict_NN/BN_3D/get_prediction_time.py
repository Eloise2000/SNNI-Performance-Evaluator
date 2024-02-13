# Tensorflow 2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from kerastuner import HyperModel, RandomSearch
import tensorflow as tf
from tensorflow.keras import layers
import time

class MyHyperModel(HyperModel):
    def __init__(self, input_shape, num_outputs):
        self.input_shape = input_shape
        self.num_outputs = num_outputs

    def build(self, hp):
        model = tf.keras.Sequential()
        
        # Tune the number of hidden layers and units
        for i in range(hp.Int('num_hidden_layers', min_value=1, max_value=2)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=128, step=32), activation='relu'))
            # model.add(layers.BatchNormalization())

        model.add(layers.Dense(self.num_outputs))
        
        # Tune the optimizer and learning rate
        # optimizer = hp.Choice('optimizer', ['adam', 'sgd'])
        optimizer = 'adam'
        # learning_rate = hp.Float('learning_rate', min_value=1e-3, max_value=1e-2)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

        return model

show_server = 1
if show_server:
    name = "server"
    project_used ='231013'
else: 
    name = "client"
    project_used ='231013'

'''"BN1_C", "BN1_H", "BN1_W", "time_cost"'''
'''"BN2_C", "BN2_H", "BN2_W", "time_cost"'''

target_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_origin/"
# target_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_without_sqnet/"
layer_filepath_BN1 = target_folder + "BN1_onlyTime_" + name + ".csv"
layer_filepath_BN2 = target_folder + "BN2_onlyTime_" + name + ".csv"
df_BN1 = pd.read_csv(layer_filepath_BN1, delimiter="\s+")
df_BN1 = df_BN1[df_BN1['time_cost'] > 0]
df_BN2 = pd.read_csv(layer_filepath_BN2, delimiter="\s+")
df_BN2 = df_BN2[df_BN2['time_cost'] > 0]

NN_folderPath = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/predict_NN/"
best_model_BN = tf.keras.models.load_model(NN_folderPath + "BN_3D/" + 'best_model_BN_' + name + '.keras')

df_row = df_BN1.iloc[3]
x_single = df_row[["BN1_C", "BN1_H", "BN1_W"]].values.reshape(1, -1)
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