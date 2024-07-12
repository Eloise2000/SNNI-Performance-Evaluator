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
    project_used ='230925'
else: 
    name = "client"
    project_used ='230925-1_used'

'''
FC columns= "FC_N", "FC_CI", "FC_CO", 'time_cost'
'''
target_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_origin/"
# target_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_without_sqnet/"
layer_filepath = target_folder + "FC_onlyTime_"+name+".csv"
df = pd.read_csv(layer_filepath, delimiter="\s+")

# print(df.head(10))
x = df.loc[:, ["FC_N", "FC_CI", "FC_CO"]]
y = df['time_cost']

NN_folderPath = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/predict_NN/"
best_model_FC = tf.keras.models.load_model(NN_folderPath + "FC/" + 'best_model_FC_' + name + '.keras')

df_row = df.iloc[30]
# print(df_row)
x_single = df_row[["FC_N", "FC_CI", "FC_CO"]].values.reshape(1, -1)
# x_single = df_row[["FC_N", "FC_CI", "FC_CO"]]
y_single = df_row['time_cost']

# Start prediction
start_time = time.time()
y_single_pred = best_model_FC.predict(x_single)
end_time = time.time()

# Calculate the time taken for prediction
prediction_time = end_time - start_time

print("Predicted Time:", y_single_pred)
print("Real Time:", y_single)
print("Diff Time:", y_single - y_single_pred)
print("Prediction Time:", prediction_time * 1000, "ms")