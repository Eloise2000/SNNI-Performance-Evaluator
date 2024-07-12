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

log_filepath = "/home/eloise/eloise/result_LAN/result-"+name+"/resnet50/data_1/log_"+name+".txt"

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

# Initialize BN dataFrame
attributes_BN = ["BN_C", "BN_H", "BN_W",'time_BN']

df_BN = pd.DataFrame(columns=attributes_BN, dtype=object)

# Initialize truncation dataFrame
attributes_truncation = ['trunc_coeff','time_trunc']

df_truncation = pd.DataFrame(columns=attributes_truncation, dtype=object)

# Initialize FC dataFrame
attributes_FC = ["FC_N", "FC_CI", "FC_CO", 'time_FC']

df_FC = pd.DataFrame(columns=attributes_FC, dtype=object)

''' Add resnet50 layer data '''
with open(log_filepath) as f:
    for line in f:
        text = line.strip().split()
        if len(text) > 3 and text[3] == "after" and text[4] == "StartComputation":
            StartComputation = float(text[-1])
        if len(text) > 3 and text[3] == "end" and text[4] == "protocol":
            EndComputation = float(text[-1])
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
            if text[-3] == "BN1": # Only BN1
                df_BN.loc[df_BN.index[-1], 'time_BN'] = time_cost
            if text[-3] == "matmul":
                df_FC.loc[df_FC.index[-1], 'time_FC'] = time_cost
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
        if text[0] == "HomBN1" and text[1][0] == "#":
            BN1_C = int(text[4].split("[")[-1][:-1])
            BN1_H = int(text[5][:-1])
            BN1_W = int(text[6][:-1])
            df_BN.loc[len(df_BN)] = {"BN_C": BN1_C, "BN_H": BN1_H, "BN_W": BN1_W}
        if text[0] == "Matmul" and text[1] != "data":
            values_FC = [int(text[i].split("=")[-1][:-1]) for i in range(3, 6)]
            df_FC.loc[len(df_FC)] = dict(zip(attributes_FC[:-1], values_FC))

''' Add resnet50 BN2 layer data '''
Flag_BN2 = False
with open(log_filepath) as f:
    for line in f:
        text = line.strip().split()
        if text[0] == "Current" and text[3] == "start" and text[-3] == "BN2":
            start = float(text[-1])
        if text[0] == "Current" and text[3] == "end" and text[-3] == "BN2":
            end = float(text[-1])
            layer_time = end - start
        if text[0] == "HomBN2" and text[1][0] == "#":
            Flag_BN2 = True
            BN2_coeff = int(text[6])
        if text[0] == "HomConv" and text[1][0] == "#":
            if Flag_BN2:
                Flag_BN2 = False
                conv_H = int(text[4].split("=")[-1][:-1])
                conv_W = int(text[5].split("=")[-1][:-1])
                conv_CI = int(text[6].split("=")[-1][:-1])
                
                if conv_H * conv_W * conv_CI != BN2_coeff: 
                    raise ValueError("The multiplication doesn't fit with total neuron!")
                
                df_BN.loc[len(df_BN)] = {"BN_C": conv_CI, "BN_H": conv_H, "BN_W": conv_W, 'time_BN': layer_time}
        if text[0] == "AvgPool" and text[1] != "data":
            if Flag_BN2:
                Flag_BN2 = False
                avgpool_imgH = int(text[16].split("=")[-1][:-1])
                avgpool_imgW = int(text[17].split("=")[-1][:-1])
                avgpool_C1 = int(text[18].split("=")[-1])

                if avgpool_imgH * avgpool_imgW * avgpool_C1 != BN2_coeff: 
                    raise ValueError("The multiplication doesn't fit with total neuron!")
                
                df_BN.loc[len(df_BN)] = {"BN_C": avgpool_imgH, "BN_H": avgpool_imgW, "BN_W": avgpool_C1, 'time_BN': layer_time}

''' Add resnet50 truncation layer data '''
Flag = False
with open(log_filepath) as f:
    for line in f:
        text = line.strip().split()
        if text[0] == "Current" and text[3] == "end" and text[4] != "protocol":
            start = float(text[-1])
        # "Current time of end protocol"
        if text[0] == "Current" and text[3] == "end" and text[4] == "protocol":
            if Flag:
                end = float(text[-1])
                time_cost = end - start
                df_truncation.loc[df_truncation.index[-1], 'time_trunc'] = time_cost
                Flag = False
        if text[0] == "Current" and text[3] == "start":
            if Flag:
                end = float(text[-1])
                time_cost = end - start
                df_truncation.loc[df_truncation.index[-1], 'time_trunc'] = time_cost
                Flag = False
        if text[0] == "Truncate" and text[1][0] == "#":
            trunc_coeff_value = int(text[3])
            df_truncation.loc[len(df_truncation)] = {"trunc_coeff": trunc_coeff_value}
            Flag = True

NN_folderPath = "/home/eloise/eloise/script/analysis/layer_onlyTime_LAN/predict_NN/"
start_time = time.time()
''' Predict conv '''
# Load the trained model
best_model_conv = tf.keras.models.load_model(NN_folderPath + "conv/" + 'best_model_conv_' + name + '.keras')

x_conv = df_conv.loc[:, ["conv_N", "conv_H", "conv_W", "conv_CI", "conv_FH", "conv_FW", "conv_CO", 
    "conv_zPadHLeft", "conv_zPadHRight", "conv_zPadWLeft", "conv_zPadWRight", 
    "conv_strideH", "conv_strideW"]]
y_conv = df_conv['time_conv']

# Predict on the sqnet data
y_conv_pred = best_model_conv.predict(x_conv)

print("The real time cost for conv layer: ", sum(y_conv))
print("The predict time cost for conv layer: ", sum(y_conv_pred))
print("Difference is: ", sum(y_conv_pred) - sum(y_conv))

''' Predict relu '''
best_model_relu = tf.keras.models.load_model(NN_folderPath + "relu/" + 'best_model_relu_' + name + '.keras')

x_relu = df_relu.loc[:, ["relu_coeff"]]
y_relu = df_relu['time_relu']

# Predict on the sqnet data
y_relu_pred = best_model_relu.predict(x_relu)

print("\nThe real time cost for relu layer: ", sum(y_relu))
print("The predict time cost for relu layer: ", sum(y_relu_pred))
print("Difference is: ", sum(y_relu_pred) - sum(y_relu))

''' Predict maxpool '''
best_model_maxpool = tf.keras.models.load_model(NN_folderPath + "mp/" + 'best_model_mp_' + name + '.keras')

x_maxpool = df_maxpool.loc[:, ["maxpool_N", "maxpool_H", "maxpool_W", "maxpool_C", "maxpool_ksizeH", "maxpool_ksizeW", 
                "maxpool_zPadHLeft", "maxpool_zPadHRight", "maxpool_zPadWLeft", "maxpool_zPadWRight",
                "maxpool_strideH", "maxpool_strideW", "maxpool_N1", 
                "maxpool_imgH", "maxpool_imgW", "maxpool_C1"]]
y_maxpool = df_maxpool['time_maxpool']

# Predict on the sqnet data
y_maxpool_pred = best_model_maxpool.predict(x_maxpool)

print("\nThe real time cost for maxpool layer: ", sum(y_maxpool))
print("The predict time cost for maxpool layer: ", sum(y_maxpool_pred))
print("Difference is: ", sum(y_maxpool_pred) - sum(y_maxpool))

''' Predict avgpool '''
best_model_avgpool = tf.keras.models.load_model(NN_folderPath + "ap/" + 'best_model_ap_' + name + '.keras')

x_avgpool = df_avgpool.loc[:, ["avgpool_N", "avgpool_H", "avgpool_W", "avgpool_C", "avgpool_ksizeH", "avgpool_ksizeW", 
                "avgpool_zPadHLeft", "avgpool_zPadHRight", "avgpool_zPadWLeft", "avgpool_zPadWRight",
                "avgpool_strideH", "avgpool_strideW", "avgpool_N1", 
                "avgpool_imgH", "avgpool_imgW", "avgpool_C1"]]
y_avgpool = df_avgpool['time_avgpool']

y_avgpool_pred = best_model_avgpool.predict(x_avgpool)

print("\nThe real time cost for avgpool layer: ", sum(y_avgpool))
print("The predict time cost for avgpool layer: ", sum(y_avgpool_pred))
print("Difference is: ", sum(y_avgpool_pred) - sum(y_avgpool))

''' Predict BN '''
best_model_BN = tf.keras.models.load_model(NN_folderPath + "BN_3D/" + 'best_model_BN_' + name + '.keras')

x_BN = df_BN.loc[:,["BN_C", "BN_H", "BN_W"]]
y_BN = df_BN['time_BN']

y_BN_pred = best_model_BN.predict(x_BN)

print("\nThe real time cost for BN layer: ", sum(y_BN))
print("The predict time cost for BN layer: ", sum(y_BN_pred))
print("Difference is: ", sum(y_BN_pred) - sum(y_BN))

''' Predict truncation '''
best_model_trunc = tf.keras.models.load_model(NN_folderPath + "trunc/" + 'best_model_trunc_' + name + '.keras')

x_trunc = df_truncation['trunc_coeff']
y_trunc = df_truncation['time_trunc']

y_trunc_pred = best_model_trunc.predict(x_trunc)

print("\nThe real time cost for trunc layer: ", sum(y_trunc))
print("The predict time cost for trunc layer: ", sum(y_trunc_pred))
print("Difference is: ", sum(y_trunc_pred) - sum(y_trunc))

''' Predict FC '''
best_model_FC = tf.keras.models.load_model(NN_folderPath + "FC/" + 'best_model_FC_' + name + '.keras')

x_FC = df_FC.loc[:, ["FC_N", "FC_CI", "FC_CO"]]
y_FC = df_FC['time_FC']

y_FC_pred = best_model_FC.predict(x_FC)

print("\nThe real time cost for FC layer: ", sum(y_FC))
print("The predict time cost for FC layer: ", sum(y_FC_pred))
print("Difference is: ", sum(y_FC_pred) - sum(y_FC))

end_time = time.time()
prediction_time = end_time - start_time

''' Total time difference '''
time_all = EndComputation-StartComputation
time_predict = sum(y_conv_pred) + sum(y_relu_pred) + sum(y_maxpool_pred) + sum(y_avgpool_pred) + sum(y_BN_pred) + sum(y_trunc_pred) + sum(y_FC_pred)
print("\nThe real all runtime is: ", time_all)
print("The predict all runtime is: ", time_predict)
print("Difference is: ", time_predict - time_all)
print("Error percentage is: ", abs(time_predict - time_all)/time_all * 100, "%")
print("Time for prediction is: ", prediction_time * 1000, " ms")