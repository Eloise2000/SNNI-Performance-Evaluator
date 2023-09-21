import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set variable here
log_filepath = "/home/eloise/eloise/result/sqnet/data_1/log_client.txt"
show_server = 0 # 0: client; 1: server

# Init the conv data list
conv_N = []
conv_H = []
conv_W = []
conv_CI = []
conv_FH = []
conv_FW = []
conv_CO = []
conv_zPadHLeft=[]
conv_zPadHRight=[]
conv_zPadWLeft=[]
conv_zPadWRight=[]
conv_strideH=[]
conv_strideW=[]
time_conv = []

# Init the maxpool data list
maxpool_N = []
maxpool_H = []
maxpool_W = []
maxpool_C = []
maxpool_ksizeH = []
maxpool_ksizeW = []
maxpool_zPadHLeft = []
maxpool_zPadHRight = []
maxpool_zPadWLeft = []
maxpool_zPadWRight = []
maxpool_strideH = []
maxpool_strideW = []
maxpool_N1 = []
maxpool_imgH = []
maxpool_imgW = []
maxpool_C1 = []
time_mp = []

# Init the avgpool data list
avgpool_N = []
avgpool_H = []
avgpool_W = []
avgpool_C = []
avgpool_ksizeH = []
avgpool_ksizeW = []
avgpool_zPadHLeft = []
avgpool_zPadHRight = []
avgpool_zPadWLeft = []
avgpool_zPadWRight = []
avgpool_strideH = []
avgpool_strideW = []
avgpool_N1 = []
avgpool_imgH = []
avgpool_imgW = []
avgpool_C1 = []
time_ap = []

# Init the relu data list
relu_coeff = []
time_relu = []

# Read the layer file and process Relu layers

with open(log_filepath) as f:
    for line in f:
        text = line.strip().split()
        if text[0] == "Current" and text[3] == "start":
            start = float(text[-1])
        if text[0] == "Current" and text[3] == "end":
            end = float(text[-1])
            if text[-3] == "conv": time_conv.append(end - start)
            if text[-3] == "maxpool": time_mp.append(end - start)
            if text[-3] == "avgpool": time_ap.append(end - start)
            if text[-3] == "relu": time_relu.append(end - start)
        if text[0] == "HomConv" and text[1][0] == "#":
            conv_N.append(int(text[3].split("=")[-1][:-1]))
            conv_H.append(int(text[4].split("=")[-1][:-1]))
            conv_W.append(int(text[5].split("=")[-1][:-1]))
            conv_CI.append(int(text[6].split("=")[-1][:-1]))
            conv_FH.append(int(text[7].split("=")[-1][:-1]))
            conv_FW.append(int(text[8].split("=")[-1][:-1]))
            conv_CO.append(int(text[9].split("=")[-1][:-1]))
            # conv_S.append(int(text[10].split("=")[-1][:-1]))
            # conv_Padding.append(int(text[-2])) # 0: Padding VALID (0 0 0 0); # 1: Padding SAME (1 1 1 1)
            conv_zPadHLeft.append(int(text[10].split("=")[-1][:-1]))
            conv_zPadHRight.append(int(text[11].split("=")[-1][:-1]))
            conv_zPadWLeft.append(int(text[12].split("=")[-1][:-1]))
            conv_zPadWRight.append(int(text[13].split("=")[-1][:-1]))
            conv_strideH.append(int(text[14].split("=")[-1][:-1]))
            conv_strideW.append(int(text[15].split("=")[-1]))
        if text[0] == "Maxpool" and text[1] != "data":
            maxpool_N.append(int(text[3].split("=")[-1][:-1]))
            maxpool_H.append(int(text[4].split("=")[-1][:-1]))
            maxpool_W.append(int(text[5].split("=")[-1][:-1]))
            maxpool_C.append(int(text[6].split("=")[-1][:-1]))
            maxpool_ksizeH.append(int(text[7].split("=")[-1][:-1]))
            maxpool_ksizeW.append(int(text[8].split("=")[-1][:-1]))
            
            maxpool_zPadHLeft.append(int(text[9].split("=")[-1][:-1]))
            maxpool_zPadHRight.append(int(text[10].split("=")[-1][:-1]))
            maxpool_zPadWLeft.append(int(text[11].split("=")[-1][:-1]))
            maxpool_zPadWRight.append(int(text[12].split("=")[-1][:-1]))
            maxpool_strideH.append(int(text[13].split("=")[-1][:-1]))
            maxpool_strideW.append(int(text[14].split("=")[-1][:-1]))
            maxpool_N1.append(int(text[15].split("=")[-1][:-1]))
            maxpool_imgH.append(int(text[16].split("=")[-1][:-1]))
            maxpool_imgW.append(int(text[17].split("=")[-1][:-1]))
            maxpool_C1.append(int(text[18].split("=")[-1]))
        if text[0] == "AvgPool" and text[1] != "data":
            avgpool_N.append(int(text[3].split("=")[-1][:-1]))
            avgpool_H.append(int(text[4].split("=")[-1][:-1]))
            avgpool_W.append(int(text[5].split("=")[-1][:-1]))
            avgpool_C.append(int(text[6].split("=")[-1][:-1]))
            avgpool_ksizeH.append(int(text[7].split("=")[-1][:-1]))
            avgpool_ksizeW.append(int(text[8].split("=")[-1][:-1]))
            
            avgpool_zPadHLeft.append(int(text[9].split("=")[-1][:-1]))
            avgpool_zPadHRight.append(int(text[10].split("=")[-1][:-1]))
            avgpool_zPadWLeft.append(int(text[11].split("=")[-1][:-1]))
            avgpool_zPadWRight.append(int(text[12].split("=")[-1][:-1]))
            avgpool_strideH.append(int(text[13].split("=")[-1][:-1]))
            avgpool_strideW.append(int(text[14].split("=")[-1][:-1]))
            avgpool_N1.append(int(text[15].split("=")[-1][:-1]))
            avgpool_imgH.append(int(text[16].split("=")[-1][:-1]))
            avgpool_imgW.append(int(text[17].split("=")[-1][:-1]))
            avgpool_C1.append(int(text[18].split("=")[-1]))
        if text[0] == "Relu" and text[1][0] == "#":
            relu_coeff.append(int(text[3]))

df_conv = pd.DataFrame(list(map(np.array, zip(conv_N, conv_H, conv_W, conv_CI, conv_FH, conv_FW, conv_CO, 
                        conv_zPadHLeft, conv_zPadHRight, conv_zPadWLeft, conv_zPadWRight, conv_strideH, conv_strideW, time_cost))),
                        columns=["conv_N", "conv_H", "conv_W", "conv_CI", "conv_FH", "conv_FW", "conv_CO", 
                        "conv_zPadHLeft", "conv_zPadHRight", "conv_zPadWLeft", "conv_zPadWRight", "conv_strideH", "conv_strideW",'time_cost'], dtype=object)

if __name__ == "__main__":
    folderpath = "/home/eloise/eloise/result/"
    generate("short1", 1, 15)