# Tensorflow 2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from kerastuner import HyperModel, RandomSearch
import tensorflow as tf
from tensorflow.keras import layers
import time

show_server = 0
if show_server:
    name = "server"
    project_used ='240208-1'
else: 
    name = "client"
    project_used ='240208-1'

'''
conv: conv_N, conv_H, conv_W, conv_CI, conv_FH, conv_FW, conv_CO, conv_S, conv_Padding, time_cost, CPU_avg, RAM_max, power_avg, energy
N, meta.ishape.height(), meta.ishape.width(),
meta.ishape.channels(), meta.fshape.height(), meta.fshape.width(),
meta.n_filters, meta.stride,
(meta.padding == gemini::Padding::VALID ? "VALID" : "SAME"), zPadHLeft,
zPadHRight, zPadWLeft, zPadWRight)
'''
target_folder = "/home/eloise/eloise/script/analysis/layer_onlyTime_WAN/dataset/data_layer/"
layer_filepath = target_folder + "conv_onlyTime_"+name+".csv"
df = pd.read_csv(layer_filepath, delimiter="\s+")

x = df.loc[:, ["conv_N", "conv_H", "conv_W", "conv_CI", "conv_FH", "conv_FW", "conv_CO", 
    "conv_zPadHLeft", "conv_zPadHRight", "conv_zPadWLeft", "conv_zPadWRight", 
    "conv_strideH", "conv_strideW"]]
y = df['time_cost']

NN_folderPath = "/home/eloise/eloise/script/analysis/layer_onlyTime_WAN/predict_NN/"
# best_model_conv = tf.keras.models.load_model(f"{NN_folderPath}conv/tuner_results_{name}/{project_used}/best_model_conv_{name}.keras")
best_model_conv = tf.keras.models.load_model(f"{NN_folderPath}conv/best_model_conv_{name}.keras")

df_row = df.iloc[30]
x_single = df_row[["conv_N", "conv_H", "conv_W", "conv_CI", "conv_FH", "conv_FW", "conv_CO", 
                    "conv_zPadHLeft", "conv_zPadHRight", "conv_zPadWLeft", "conv_zPadWRight", 
                    "conv_strideH", "conv_strideW"]].values.reshape(1, -1)
y_single = df_row['time_cost']

# Start prediction
start_time = time.time()
y_single_pred = best_model_conv.predict(x_single)
end_time = time.time()

# Calculate the time taken for prediction
prediction_time = end_time - start_time

print("Predicted Time:", y_single_pred)
print("Real Time:", y_single)
print("Diff Time:", y_single - y_single_pred)
print("Prediction Time:", prediction_time * 1000, "ms")