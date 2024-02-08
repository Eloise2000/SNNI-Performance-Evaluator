import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from joblib import load
import time

# Set variable here
show_server = 1 # 0: client; 1: server
if show_server:
    name = "server"
else: 
    name = "client"

log_filepath = "/home/eloise/eloise/result/resnet50/data_1/log_" + name + ".txt"

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
attributes_BN = ['BN_coeff','time_BN']

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
            if text[-3][:-1] == "BN":
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
            BN_coeff_value = BN1_C * BN1_H * BN1_W
            df_BN.loc[len(df_BN)] = {"BN_coeff": BN_coeff_value}
        if text[0] == "HomBN2" and text[1][0] == "#":
            BN_coeff_value = int(text[6])
            df_BN.loc[len(df_BN)] = {"BN_coeff": BN_coeff_value}
        if text[0] == "Matmul" and text[1] != "data":
            values_FC = [int(text[i].split("=")[-1][:-1]) for i in range(3, 6)]
            df_FC.loc[len(df_FC)] = dict(zip(attributes_FC[:-1], values_FC))


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

# print(df_conv)
# print(df_relu)
# print(df_avgpool)
# print(df_maxpool)

LR_folderPath = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/predict_regression/LR_model/"
''' Predict conv '''
reg_conv = load(LR_folderPath + "conv_" + name + "_LR_model.joblib")
scaler_conv = load(LR_folderPath + "conv_" + name + "_scaler.joblib")

# Avoid the first IO, which takes most of the time
start_time = time.time()

X_conv = pd.DataFrame()
conv_H_output = (df_conv['conv_H'] + df_conv["conv_zPadHLeft"] + df_conv["conv_zPadHRight"] - df_conv["conv_FH"]) / df_conv["conv_strideH"] + 1
conv_W_output = (df_conv['conv_W'] + df_conv["conv_zPadWLeft"] + df_conv["conv_zPadWRight"] - df_conv["conv_FW"]) / df_conv["conv_strideW"] + 1
X_conv['FLOPs'] = (2 * df_conv["conv_FH"] * df_conv["conv_FW"] * df_conv["conv_CI"]) * conv_H_output * conv_W_output * df_conv["conv_CO"]
X_conv['IN_MACs'] = (df_conv['conv_H'] * df_conv['conv_W'] * df_conv["conv_CI"]) * 8
X_conv['PAR_MACs'] = (df_conv["conv_CI"] * df_conv["conv_FH"] * df_conv["conv_FW"] * df_conv["conv_CO"]) * 8
X_conv['OUT_MACs'] = (conv_H_output * conv_W_output * df_conv["conv_CO"]) * 8
y_conv = df_conv['time_conv']

X_conv_normalized = scaler_conv.transform(X_conv)
y_conv_pred = reg_conv.predict(X_conv_normalized)

print("The real time cost for conv layer: ", sum(y_conv))
print("The predict time cost for conv layer: ", sum(y_conv_pred))
print("Difference is: ", sum(y_conv_pred) - sum(y_conv))

''' Predict relu '''
reg_relu = load(LR_folderPath + "relu_" + name + "_LR_model.joblib")
scaler_relu = load(LR_folderPath + "relu_" + name + "_scaler.joblib")

X_relu = pd.DataFrame()
X_relu["FLOPs"] = df_relu['relu_coeff']
y_relu = df_relu['time_relu']

X_relu_normalized = scaler_relu.transform(X_relu)
y_relu_pred = reg_relu.predict(X_relu_normalized)

print("The real time cost for relu layer: ", sum(y_relu))
print("The predict time cost for relu layer: ", sum(y_relu_pred))
print("Difference is: ", sum(y_relu_pred) - sum(y_relu))

''' Predict maxpool '''
reg_maxpool = load(LR_folderPath + "mp_" + name + "_LR_model.joblib")
scaler_maxpool = load(LR_folderPath + "mp_" + name + "_scaler.joblib")

X_maxpool = pd.DataFrame()
mp_H_output = (df_maxpool['maxpool_H'] + df_maxpool["maxpool_zPadHLeft"] + df_maxpool["maxpool_zPadHRight"] - df_maxpool["maxpool_ksizeH"]) / df_maxpool["maxpool_strideH"] + 1
mp_W_output = (df_maxpool['maxpool_W'] + df_maxpool["maxpool_zPadWLeft"] + df_maxpool["maxpool_zPadWRight"] - df_maxpool["maxpool_ksizeW"]) / df_maxpool["maxpool_strideW"] + 1

X_maxpool['FLOPs'] = (df_maxpool["maxpool_ksizeH"] * df_maxpool["maxpool_ksizeW"]) * (mp_H_output * mp_W_output) * df_maxpool["maxpool_C"]
X_maxpool['IN_MACs'] = (df_maxpool['maxpool_H'] * df_maxpool['maxpool_W']) * df_maxpool["maxpool_C"] * 8
X_maxpool['OUT_MACs'] = (mp_H_output * mp_W_output) * df_maxpool["maxpool_C"] * 8
y_maxpool = df_maxpool['time_maxpool']

X_maxpool_normalized = scaler_maxpool.transform(X_maxpool)
y_maxpool_pred = reg_maxpool.predict(X_maxpool_normalized)

print("The real time cost for maxpool layer: ", sum(y_maxpool))
print("The predict time cost for maxpool layer: ", sum(y_maxpool_pred))
print("Difference is: ", sum(y_maxpool_pred) - sum(y_maxpool))

''' Predict avgpool '''
reg_avgpool = load(LR_folderPath + "ap_" + name + "_LR_model.joblib")
scaler_avgpool = load(LR_folderPath + "ap_" + name + "_scaler.joblib")

X_avgpool = pd.DataFrame()
ap_H_output = (df_avgpool['avgpool_H'] + df_avgpool["avgpool_zPadHLeft"] + df_avgpool["avgpool_zPadHRight"] - df_avgpool["avgpool_ksizeH"]) / df_avgpool["avgpool_strideH"] + 1
ap_W_output = (df_avgpool['avgpool_W'] + df_avgpool["avgpool_zPadWLeft"] + df_avgpool["avgpool_zPadWRight"] - df_avgpool["avgpool_ksizeW"]) / df_avgpool["avgpool_strideW"] + 1

X_avgpool['FLOPs'] = (df_avgpool["avgpool_ksizeH"] * df_avgpool["avgpool_ksizeW"]) * (ap_H_output * ap_W_output) * df_avgpool["avgpool_C"]
X_avgpool['IN_MACs'] = (df_avgpool['avgpool_H'] * df_avgpool['avgpool_W']) * df_avgpool["avgpool_C"] * 8
X_avgpool['OUT_MACs'] = (ap_H_output * ap_W_output) * df_avgpool["avgpool_C"] * 8
y_avgpool = df_avgpool['time_avgpool']

X_avgpool_normalized = scaler_avgpool.transform(X_avgpool)
y_avgpool_pred = reg_avgpool.predict(X_avgpool_normalized)

print("The real time cost for avgpool layer: ", sum(y_avgpool))
print("The predict time cost for avgpool layer: ", sum(y_avgpool_pred))
print("Difference is: ", sum(y_avgpool_pred) - sum(y_avgpool))

''' Predict BN '''
reg_BN = load(LR_folderPath + "BN_" + name + "_LR_model.joblib")
scaler_BN = load(LR_folderPath + "BN_" + name + "_scaler.joblib")

X_BN = pd.DataFrame()
X_BN["FLOPs"] = df_BN['BN_coeff']
y_BN = df_BN['time_BN']

X_BN_normalized = scaler_BN.transform(X_BN)
y_BN_pred = reg_BN.predict(X_BN_normalized)

print("The real time cost for BN layer: ", sum(y_BN))
print("The predict time cost for BN layer: ", sum(y_BN_pred))
print("Difference is: ", sum(y_BN_pred) - sum(y_BN))

''' Predict truncation '''
reg_trunc = load(LR_folderPath + "trunc_" + name + "_LR_model.joblib")
scaler_trunc = load(LR_folderPath + "trunc_" + name + "_scaler.joblib")

X_trunc = pd.DataFrame()
X_trunc["FLOPs"] = df_truncation['trunc_coeff']
y_trunc = df_truncation['time_trunc']

X_trunc_normalized = scaler_trunc.transform(X_trunc)
y_trunc_pred = reg_trunc.predict(X_trunc_normalized)

print("The real time cost for trunc layer: ", sum(y_trunc))
print("The predict time cost for trunc layer: ", sum(y_trunc_pred))
print("Difference is: ", sum(y_trunc_pred) - sum(y_trunc))

''' Predict FC '''
reg_FC = load(LR_folderPath + "FC_" + name + "_LR_model.joblib")
scaler_FC = load(LR_folderPath + "FC_" + name + "_scaler.joblib")

X_FC = pd.DataFrame()
X_FC['FLOPs'] = (2 * df_FC["FC_CI"] * df_FC["FC_CO"]) + df_FC["FC_CO"]
# sizeof(uint64_t) is unsigned 64-bit integer -> 8 bytes
X_FC['IN_MACs'] = df_FC["FC_CI"] * 8
X_FC['PAR_MACs'] = (df_FC["FC_CI"] + 1) * df_FC["FC_CO"] * 8
X_FC['OUT_MACs'] = df_FC["FC_CO"] * 8
y_FC = df_FC['time_FC']

X_FC_normalized = scaler_FC.transform(X_FC)
y_FC_pred = reg_FC.predict(X_FC_normalized)

print("The real time cost for FC layer: ", sum(y_FC))
print("The predict time cost for FC layer: ", sum(y_FC_pred))
print("Difference is: ", sum(y_FC_pred) - sum(y_FC))

end_time = time.time()
prediction_time = end_time - start_time

''' Total time difference '''
time_all = 1691397285055-1691397053779
# time_all = sum(y_conv) + sum(y_relu) + sum(y_maxpool) + sum(y_avgpool) + sum(y_BN) + sum(y_trunc) + sum(y_FC)
time_predict = sum(y_conv_pred) + sum(y_relu_pred) + sum(y_maxpool_pred) + sum(y_avgpool_pred) + sum(y_BN_pred) + sum(y_trunc_pred) + sum(y_FC_pred)
print("The real all runtime is: ", time_all)
print("The predict all runtime is: ", time_predict)
print("Difference is: ", time_predict - time_all)
print("Time for prediction is: ", prediction_time * 1000, " ms")