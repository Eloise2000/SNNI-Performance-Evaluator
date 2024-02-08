import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import time

# Set variable here
show_server = 1 # 0: client; 1: server
if show_server:
    name = "server"
else: 
    name = "client"
log_filepath = "/home/eloise/eloise/result/sqnet/data_1/log_" + name + ".txt"

''' Initialize conv dataFrame '''
attributes_conv = ["conv_N", "conv_H", "conv_W", "conv_CI", "conv_FH", "conv_FW", "conv_CO",
                   "conv_zPadHLeft", "conv_zPadHRight", "conv_zPadWLeft", "conv_zPadWRight",
                   "conv_strideH", "conv_strideW", 'time_conv']

df_conv = pd.DataFrame(columns=attributes_conv, dtype=object)

# Initialize relu dataFrame
attributes_relu = ['relu_coeff','time_relu']

df_relu = pd.DataFrame(columns=attributes_relu, dtype=object)

# Initialize maxpool dataFrame
attributes_maxpool = ["maxpool_N", "maxpool_H", "maxpool_W", "maxpool_C", "maxpool_ksizeH", "maxpool_ksizeW",
                      "maxpool_zPadHLeft", "maxpool_zPadHRight", "maxpool_zPadWLeft", "maxpool_zPadWRight",
                      "maxpool_strideH", "maxpool_strideW", "maxpool_N1", "maxpool_imgH", "maxpool_imgW", "maxpool_C1", 'time_maxpool']

df_maxpool = pd.DataFrame(columns=attributes_maxpool, dtype=object)

# Initialize avgpool dataFrame
attributes_avgpool = ["avgpool_N", "avgpool_H", "avgpool_W", "avgpool_C", "avgpool_ksizeH", "avgpool_ksizeW",
                      "avgpool_zPadHLeft", "avgpool_zPadHRight", "avgpool_zPadWLeft", "avgpool_zPadWRight",
                      "avgpool_strideH", "avgpool_strideW", "avgpool_N1", "avgpool_imgH", "avgpool_imgW", "avgpool_C1", 'time_avgpool']

df_avgpool = pd.DataFrame(columns=attributes_avgpool, dtype=object)

''' Add sqnet layer data '''
with open(log_filepath) as f:
    for line in f:
        text = line.strip().split()
        if text[0] == "Current" and text[3] == "start":
            start = float(text[-1])
        if text[0] == "Current" and text[3] == "end":
            end = float(text[-1])
            time_cost = end - start
            if text[-3] == "conv":
                df_conv.loc[df_conv.index[-1], 'time_conv'] = time_cost
            if text[-3] == "relu":
                df_relu.loc[df_relu.index[-1], 'time_relu'] = time_cost
            if text[-3] == "maxpool":
                df_maxpool.loc[df_maxpool.index[-1], 'time_maxpool'] = time_cost
            if text[-3] == "avgpool":
                df_avgpool.loc[df_avgpool.index[-1], 'time_avgpool'] = time_cost
        if text[0] == "HomConv" and text[1][0] == "#":
            values_conv = [int(text[i].split("=")[-1][:-1]) for i in range(3, 15)]
            values_conv.append(int(text[15].split("=")[-1]))
            # Append to DataFrame without 'time_cost'
            df_conv.loc[len(df_conv)] = dict(zip(attributes_conv[:-1], values_conv))
        if text[0] == "Relu" and text[1][0] == "#":
            relu_coeff_value = int(text[3])
            df_relu.loc[len(df_relu)] = {"relu_coeff": relu_coeff_value}
        if text[0] == "Maxpool" and text[1] != "data":
            values_maxpool = [int(text[i].split("=")[-1][:-1]) for i in range(3, 18)]
            values_maxpool.append(int(text[18].split("=")[-1]))
            df_maxpool.loc[len(df_maxpool)] = dict(zip(attributes_maxpool[:-1], values_maxpool))
        if text[0] == "AvgPool" and text[1] != "data":
            values_avgpool = [int(text[i].split("=")[-1][:-1]) for i in range(3, 18)]
            values_avgpool.append(int(text[18].split("=")[-1]))
            df_avgpool.loc[len(df_avgpool)] = dict(zip(attributes_avgpool[:-1], values_avgpool))

NN_folderPath = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/predict_NN/"
''' Predict conv '''
# Load the trained model
best_model_conv = tf.keras.models.load_model(NN_folderPath + "conv/" + 'best_model_conv_' + name + '.keras')

start_time = time.time()

x_conv = df_conv.loc[:, ["conv_N", "conv_H", "conv_W", "conv_CI", "conv_FH", "conv_FW", "conv_CO", 
    "conv_zPadHLeft", "conv_zPadHRight", "conv_zPadWLeft", "conv_zPadWRight", 
    "conv_strideH", "conv_strideW"]]
y_conv = df_conv['time_conv']

# Predict on the sqnet data
y_conv_pred = best_model_conv.predict(x_conv)

print("The real time cost for conv layer: ", sum(y_conv))
print("The predict time cost for conv layer: ", sum(y_conv_pred))
print("Difference is: ", sum(y_conv_pred) - sum(y_conv))

end_time1 = time.time()
print("**** Here is ", (end_time1 - start_time) * 1000)

''' Predict relu '''
best_model_relu = tf.keras.models.load_model(NN_folderPath + "relu/" + 'best_model_relu_' + name + '.keras')

time1 = time.time()

x_relu = df_relu.loc[:, ["relu_coeff"]]
y_relu = df_relu['time_relu']

# Predict on the sqnet data
y_relu_pred = best_model_relu.predict(x_relu)

print("The real time cost for relu layer: ", sum(y_relu))
print("The predict time cost for relu layer: ", sum(y_relu_pred))
print("Difference is: ", sum(y_relu_pred) - sum(y_relu))

end_time2 = time.time()
print("**** Here is ", (end_time2 - time1) * 1000)

''' Predict maxpool '''
best_model_maxpool = tf.keras.models.load_model(NN_folderPath + "mp/" + 'best_model_mp_' + name + '.keras')

x_maxpool = df_maxpool.loc[:, ["maxpool_N", "maxpool_H", "maxpool_W", "maxpool_C", "maxpool_ksizeH", "maxpool_ksizeW", 
                "maxpool_zPadHLeft", "maxpool_zPadHRight", "maxpool_zPadWLeft", "maxpool_zPadWRight",
                "maxpool_strideH", "maxpool_strideW", "maxpool_N1", 
                "maxpool_imgH", "maxpool_imgW", "maxpool_C1"]]
y_maxpool = df_maxpool['time_maxpool']

# Predict on the sqnet data
y_maxpool_pred = best_model_maxpool.predict(x_maxpool)

print("The real time cost for maxpool layer: ", sum(y_maxpool))
print("The predict time cost for maxpool layer: ", sum(y_maxpool_pred))
print("Difference is: ", sum(y_maxpool_pred) - sum(y_maxpool))

''' Predict avgpool '''
best_model_avgpool = tf.keras.models.load_model(NN_folderPath + "ap/" + 'best_model_ap_' + name + '.keras')

x_avgpool = df_avgpool.loc[:, ["avgpool_N", "avgpool_H", "avgpool_W", "avgpool_C", "avgpool_ksizeH", "avgpool_ksizeW", 
                "avgpool_zPadHLeft", "avgpool_zPadHRight", "avgpool_zPadWLeft", "avgpool_zPadWRight",
                "avgpool_strideH", "avgpool_strideW", "avgpool_N1", 
                "avgpool_imgH", "avgpool_imgW", "avgpool_C1"]]
y_avgpool = df_avgpool['time_avgpool']

# Predict on the sqnet data
y_avgpool_pred = best_model_avgpool.predict(x_avgpool)

print("The real time cost for avgpool layer: ", sum(y_avgpool))
print("The predict time cost for avgpool layer: ", sum(y_avgpool_pred))
print("Difference is: ", sum(y_avgpool_pred) - sum(y_avgpool))

end_time = time.time()
prediction_time = end_time - start_time

''' Total time difference '''
time_all = 1691401122438-1691401079410
time_predict = sum(y_conv_pred) + sum(y_relu_pred) + sum(y_maxpool_pred) + sum(y_avgpool_pred)
print("The real all runtime is: ", time_all)
print("The predict all runtime is: ", time_predict)
print("Difference is: ", time_predict - time_all)
print("Time for prediction is: ", prediction_time * 1000, " ms")