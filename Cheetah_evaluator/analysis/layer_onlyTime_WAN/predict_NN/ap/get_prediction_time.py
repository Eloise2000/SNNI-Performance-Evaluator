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
avgpool: "avgpool_N", "avgpool_H", "avgpool_W", "avgpool_C", "avgpool_ksizeH", "avgpool_ksizeW"
'''
target_folder = "/home/eloise/eloise/script/analysis/layer_onlyTime_WAN/dataset/data_layer/"
layer_filepath = target_folder + "avgpool_onlyTime_"+name+".csv"
df = pd.read_csv(layer_filepath, delimiter="\s+")

x = df.loc[:, ["avgpool_N", "avgpool_H", "avgpool_W", "avgpool_C", "avgpool_ksizeH", "avgpool_ksizeW", 
                "avgpool_zPadHLeft", "avgpool_zPadHRight", "avgpool_zPadWLeft", "avgpool_zPadWRight",
                "avgpool_strideH", "avgpool_strideW", "avgpool_N1", 
                "avgpool_imgH", "avgpool_imgW", "avgpool_C1"]]
y = df['time_cost']

NN_folderPath = "/home/eloise/eloise/script/analysis/layer_onlyTime_WAN/predict_NN/"
# best_model_mp = tf.keras.models.load_model(f"{NN_folderPath}ap/tuner_results_{name}/{project_used}/best_model_ap_{name}.keras")
best_model_mp = tf.keras.models.load_model(f"{NN_folderPath}ap/best_model_ap_{name}.keras")

df_row = df.iloc[300]
x_single = df_row[["avgpool_N", "avgpool_H", "avgpool_W", "avgpool_C", "avgpool_ksizeH", "avgpool_ksizeW", 
                "avgpool_zPadHLeft", "avgpool_zPadHRight", "avgpool_zPadWLeft", "avgpool_zPadWRight",
                "avgpool_strideH", "avgpool_strideW", "avgpool_N1", 
                "avgpool_imgH", "avgpool_imgW", "avgpool_C1"]].values.reshape(1, -1)
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