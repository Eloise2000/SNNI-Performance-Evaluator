import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set variable here
show_server = 1 # 0: client; 1: server
folderpath = "/home/eloise/eloise/result/"
    
if show_server:
    name = "server"
else:
    name = "client"

save_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_RNN/" + name + "/"

samples_sparse = []
time = []
time_seq = []
# conv: 0-12 (13)
# relu: 13 (1)
# maxpool: 14-29 (16)
# avgpool: 30-45 (16)
# BN: 46 (1)
# Matmul: 47-49 (3)
# Trunc: 50 (1)
feature_length = 51

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
            time.append(end_all_layers - start_all_layers)
            time_seq.append(time_seq_for_sample)

if __name__ == "__main__":
    generate("short1", 1, 15)
    generate("short2", 1, 15)
    generate("resnet50", 1, 15)
    generate("densenet121", 1, 15)
    generate("sqnet", 1, 15)
    generate("conv1", 1, 15)
    generate("conv2", 1, 15)
    generate("conv3", 1, 15)
    generate("conv4", 1, 15)
    generate("conv5", 1, 15)
    generate("conv6", 1, 15)
    generate("conv7", 1, 15)
    generate("conv8", 1, 15)
    generate("conv9", 1, 15)
    generate("ap1", 1, 15)
    generate("ap2", 1, 15)
    generate("ap3", 1, 15)
    generate("ap4", 1, 15)
    generate("ap5", 1, 15)
    generate("ap6", 1, 15)
    generate("ap7", 1, 15)
    generate("ap8", 1, 15)
    generate("ap9", 1, 15)
    generate("convap1", 1, 15)
    generate("convap2", 1, 15)
    generate("convap3", 1, 15)
    generate("convap4", 1, 15)
    generate("convap5", 1, 15)
    generate("convap6", 1, 15)
    generate("convap7", 1, 15)
    generate("convap8", 1, 15)
    generate("convap9", 1, 15)
    generate("convmp1", 1, 15)
    generate("convmp2", 1, 15)
    generate("convmp3", 1, 15)
    generate("convmp4", 1, 15)
    generate("convmp5", 1, 15)
    generate("mp1", 1, 15)
    generate("mp2", 1, 15)
    generate("mp3", 1, 15)
    generate("mp4", 1, 15)
    generate("mp5", 1, 15)
    generate("mp6", 1, 15)
    generate("mp7", 1, 15)
    generate("mp8", 1, 15)
    generate("mp9", 1, 15)


    # Padding the dataset
    max_time_step = max(len(sample) for sample in samples_sparse)
    for i in range(len(samples_sparse)):
        for _ in range(max_time_step - len(samples_sparse[i])):
            samples_sparse[i].append(np.zeros(feature_length))
            time_seq[i].append(np.zeros(1))

    samples_sparse = np.array(samples_sparse)
    time = np.array(time)
    time_seq = np.array(time_seq)

    
    # print(samples_sparse)
    # print(time_seq)
    # print(samples_sparse.shape)
    # print(time_seq.shape)
    np.save(save_folder + "samples_sparse.npy", samples_sparse)
    # np.save(save_folder + "time.npy", time)
    np.save(save_folder + "time_seq.npy", time_seq)