import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Init the plot list
BN2_C = []
BN2_H = []
BN2_W = []

time_cost = []

# Set variable here
show_server = 0 # 0: client; 1: server
target_folder = "/home/eloise/eloise/script/analysis/layer_onlyTime_WAN/dataset/data_layer/"

if show_server:
    col_name = "server"
    folderpath = "/home/eloise/eloise/result_WAN/result-server/"
else:
    col_name = "client"
    folderpath = "/home/eloise/eloise/result_WAN/result-client/"

'''
Generate the features for convolution layer
model -> str: which network model (eg. "sqnet", "resnet", "densenet121")
startIdx -> int: start index -> range(start, end+1)
endIdx -> int: end index
'''
def generate(model, startIdx, endIdx):
    Flag_BN2 = False
    folder = folderpath + model + "/"
    for idx in range(startIdx, endIdx+1):
        if show_server:
            log_filepath = folder + "data_" + str(idx) + '/log_server.txt'
        else:
            log_filepath = folder + "data_" + str(idx) + '/log_client.txt'

        # Read the layer file and process Relu layers
        start = []
        end = []

        with open(log_filepath) as f:
            for line in f:
                text = line.strip().split()
                if text[0] == "Current" and text[3] == "start" and text[-3] == "BN2":
                    start.append(float(text[-1]))
                if text[0] == "Current" and text[3] == "end" and text[-3] == "BN2":
                    end.append(float(text[-1]))
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
                        
                        BN2_C.append(conv_CI)
                        BN2_H.append(conv_H)
                        BN2_W.append(conv_W)
                if text[0] == "AvgPool" and text[1] != "data":
                    if Flag_BN2:
                        Flag_BN2 = False
                        avgpool_imgH = int(text[16].split("=")[-1][:-1])
                        avgpool_imgW = int(text[17].split("=")[-1][:-1])
                        avgpool_C1 = int(text[18].split("=")[-1])

                        if avgpool_imgH * avgpool_imgW * avgpool_C1 != BN2_coeff: 
                            raise ValueError("The multiplication doesn't fit with total neuron!")
                        
                        BN2_C.append(avgpool_imgH)
                        BN2_H.append(avgpool_imgW)
                        BN2_W.append(avgpool_C1)
                    
        for i in range(len(start)):
            time_cost.append(end[i] - start[i])

if __name__ == "__main__":
    generate("resnet50", 1, 10)
    generate("densenet121", 1, 10)

    # write to file (nan means not valid) 
    df = pd.DataFrame(list(map(np.array, zip(BN2_C, BN2_H, BN2_W, time_cost))),
                        columns=["BN2_C", "BN2_H", "BN2_W", 'time_cost'], dtype=object)

    print(df.head(30))
    print(df.shape)
    target_filename = target_folder + "BN2_onlyTime_3D_" + col_name + ".csv"
    df.to_csv(target_filename, sep='\t', na_rep=np.nan)