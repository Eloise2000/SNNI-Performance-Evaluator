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

'''
maxpool: "maxpool_N", "maxpool_H", "maxpool_W", "maxpool_C", "maxpool_ksizeH", "maxpool_ksizeW"
'''
target_folder = "/home/eloise/eloise/script/analysis/layer_onlyTime_LAN/dataset/data_layer/"
layer_filepath = target_folder + "maxpool_onlyTime_"+name+".csv"
df = pd.read_csv(layer_filepath, delimiter="\s+")

# print(df.head(10))
x = df.loc[:, ["maxpool_N", "maxpool_H", "maxpool_W", "maxpool_C", "maxpool_ksizeH", "maxpool_ksizeW", 
                "maxpool_zPadHLeft", "maxpool_zPadHRight", "maxpool_zPadWLeft", "maxpool_zPadWRight",
                "maxpool_strideH", "maxpool_strideW", "maxpool_N1", 
                "maxpool_imgH", "maxpool_imgW", "maxpool_C1"]]
y = df['time_cost']

NN_folderPath = "/home/eloise/eloise/script/analysis/layer_onlyTime_LAN/predict_NN/"
best_model_mp = tf.keras.models.load_model(f"{NN_folderPath}mp/tuner_results_{name}/{project_used}/best_model_mp_{name}.keras")

df_row = df.iloc[300]
x_single = df_row[["maxpool_N", "maxpool_H", "maxpool_W", "maxpool_C", "maxpool_ksizeH", "maxpool_ksizeW", 
                "maxpool_zPadHLeft", "maxpool_zPadHRight", "maxpool_zPadWLeft", "maxpool_zPadWRight",
                "maxpool_strideH", "maxpool_strideW", "maxpool_N1", 
                "maxpool_imgH", "maxpool_imgW", "maxpool_C1"]].values.reshape(1, -1)
y_single = df_row['time_cost']

# Start prediction
start_time = time.time()
y_single_pred = best_model_mp.predict(x_single)
end_time = time.time()

# Calculate the time taken for prediction
prediction_time = end_time - start_time

print("Predicted Time:", y_single_pred)
print("Real Time:", y_single)
print("Diff Time:", y_single - y_single_pred)
print("Prediction Time:", prediction_time * 1000, "ms")