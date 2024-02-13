import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from joblib import load
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Set variable here
show_server = 1 # 0: client; 1: server
RNN_folderPath = "/home/eloise/eloise/script/analysis/layer_onlyTime_WAN/predict_RNN/"

if show_server:
    name = "server"
    folderpath = "/home/eloise/eloise/result_WAN/result-server/"
else:
    name = "client"
    folderpath = "/home/eloise/eloise/result_WAN/result-client/"

samples_sparse = []
time_seq = []
# conv: 0-12 (13)
# relu: 13 (1)
# maxpool: 14-29 (16)
# avgpool: 30-45 (16)
# BN: 46 (1)
# Matmul: 47-49 (3)
# Trunc: 50 (1)
feature_length = 51

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

'''
Generate the features for whole network (samples_sparse, time_steps, features)
model -> str: which network model (eg. "sqnet", "resnet", "densenet121")
startIdx -> int: start index -> range(start, end+1)
endIdx -> int: end index
'''
def generate(model, startIdx, endIdx):
    Flag_trunc = False
    folder = folderpath + model + "/"
    for idx in range(startIdx, endIdx+1):
        log_filepath = folder + "data_" + str(idx) + '/log_' + name + '.txt'

        with open(log_filepath) as f:
            # Initialize a list to store the time steps for the current sample
            time_steps_for_sample = []
            time_seq_for_sample = []
            for line in f:
                text = line.strip().split()
                # First decode the start layers: "Current time of after StartComputation = 1691397053779"
                if text[0] == "Current" and text[3] == "after" and text[4] == "StartComputation":
                    start_all_layers = float(text[-1])
                # Last decode the end of protocol: "Current time of end protocol = 1691397285055"
                if text[0] == "Current" and text[3] == "end" and text[4] == "protocol":
                    end_all_layers = float(text[-1])

                if text[0] == "Current" and text[3] == "start":
                    if text[-3] == "conv" or text[-3] == "relu" or text[-3] == "maxpool" or text[-3] == "avgpool" or text[-3] == "matmul" or text[-3] == "BN1" or text[-3] == "BN2":
                        start = float(text[-1])
                    if Flag_trunc:
                        end_trunc = float(text[-1])
                        time_seq_for_sample.append([end_trunc - start_trunc])
                        Flag_trunc = False
                if text[0] == "Current" and text[3] == "end":
                    if text[-3] == "conv" or text[-3] == "relu" or text[-3] == "maxpool" or text[-3] == "avgpool" or text[-3] == "matmul" or text[-3] == "BN1" or text[-3] == "BN2":
                        end = float(text[-1])
                        time_seq_for_sample.append([end - start])
                    if text[4] != "protocol":
                        start_trunc = float(text[-1])
                    if text[4] == "protocol":
                        if Flag_trunc:
                            end_trunc = float(text[-1])
                            time_seq_for_sample.append([end_trunc - start_trunc])
                            Flag_trunc = False
                # conv: 0-12 (13)
                if text[0] == "HomConv" and text[1][0] == "#":
                    features_conv = [int(text[i].split("=")[-1][:-1]) for i in range(3, 15)]
                    features_conv.append(int(text[15].split("=")[-1]))

                    padding_length = feature_length - len(features_conv)
                    features_conv = np.pad(features_conv, (0, padding_length), mode='constant')
                    time_steps_for_sample.append(features_conv)

                    # print(features_conv, features_conv.shape)
                # relu: 13 (1)
                if text[0] == "Relu" and text[1][0] == "#":
                    features_relu = [int(text[3])]

                    features_relu = np.pad(features_relu, (13, 37), mode='constant')
                    time_steps_for_sample.append(features_relu)

                    # print(features_relu, features_relu.shape)
                # maxpool: 14-29 (16)
                if text[0] == "Maxpool" and text[1] != "data":
                    features_maxpool = [int(text[i].split("=")[-1][:-1]) for i in range(3, 18)]
                    features_maxpool.append(int(text[18].split("=")[-1]))

                    features_maxpool = np.pad(features_maxpool, (14, 21), mode='constant')
                    time_steps_for_sample.append(np.array(features_maxpool))

                    # print(features_maxpool, features_maxpool.shape)
                # avgpool: 30-45 (16)
                if text[0] == "AvgPool" and text[1] != "data":
                    features_avgpool = [int(text[i].split("=")[-1][:-1]) for i in range(3, 18)]
                    features_avgpool.append(int(text[18].split("=")[-1]))

                    features_avgpool = np.pad(features_avgpool, (30, 5), mode='constant')
                    time_steps_for_sample.append(np.array(features_avgpool))

                    # print(features_avgpool, features_avgpool.shape)
                # BN: 46 (1)
                if text[0] == "HomBN1" and text[1][0] == "#":
                    BN1_C = int(text[4].split("[")[-1][:-1])
                    BN1_H = int(text[5][:-1])
                    BN1_W = int(text[6][:-1])
                    BN_coeff_value = BN1_C * BN1_H * BN1_W
                    features_BN = [BN_coeff_value]

                    features_BN = np.pad(features_BN, (46, 4), mode='constant')
                    time_steps_for_sample.append(features_BN)

                    # print(features_BN, features_BN.shape)
                if text[0] == "HomBN2" and text[1][0] == "#":
                    features_BN = [int(text[6])]

                    features_BN = np.pad(features_BN, (46, 4), mode='constant')
                    time_steps_for_sample.append(features_BN)

                    # print(features_BN, features_BN.shape)
                # Matmul: 47-49 (3)
                if text[0] == "Matmul" and text[1] != "data":
                    features_matmul = [int(text[i].split("=")[-1][:-1]) for i in range(3, 6)]

                    features_matmul = np.pad(features_matmul, (47, 1), mode='constant')
                    time_steps_for_sample.append(features_matmul)

                    # print(features_matmul, features_matmul.shape)
                # Trunc: 50 (1)
                if text[0] == "Truncate" and text[1][0] == "#":
                    Flag_trunc = True
                    features_truncation = [int(text[3])]

                    features_truncation = np.pad(features_truncation, (50, 0), mode='constant')
                    time_steps_for_sample.append(features_truncation)

                    # print(features_truncation, features_truncation.shape)

            samples_sparse.append(time_steps_for_sample)
            time_seq.append(time_seq_for_sample)

    return end_all_layers, start_all_layers

if __name__ == "__main__":
    end_all_layers, start_all_layers = generate("densenet121", 1, 1)
    samples_sparse = np.array(samples_sparse)
    time_seq = np.array(time_seq)
    
    start_time = time.time()
    # Apply Min-Max scaling
    scaler_RNN = load(RNN_folderPath + "RNN_" + name + "_scaler.joblib")
    num_samples, max_time_steps, num_features = samples_sparse.shape
    X_normalized = scaler_RNN.transform(samples_sparse.reshape(-1, num_features)).reshape(samples_sparse.shape)

    # Predict
    best_model_RNN = tf.keras.models.load_model(RNN_folderPath + 'best_model_RNN_' + name + '.keras')
    y_pred = best_model_RNN.predict(X_normalized)
    
    end_time = time.time()
    prediction_time = end_time - start_time

    filtered_test, filtered_pred = remove_consecutive_zeros(time_seq[0], y_pred[0])

    time_all = end_all_layers - start_all_layers
    print("The real all runtime is: ", time_all, "ms")
    print("The predict all runtime is: ", sum(filtered_pred), "ms")
    print("Difference is: ", sum(filtered_pred) - time_all, "ms")
    mae = mean_absolute_error(filtered_test, filtered_pred)
    print("Mean Absolute Error (MAE):", mae)
    print("Error percentage is: ", abs(sum(filtered_pred) - time_all)/time_all * 100, "%")
    print("Time for prediction is: ", prediction_time * 1000, " ms")

    for i in range(len(filtered_pred)):
        print("Layer:", i, "Real value:", filtered_test[i], "Predict value:", filtered_pred[i], "Error: ", filtered_test[i]-filtered_pred[i])
    