import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set variable here
show_server = 1 # 0: client; 1: server
# target_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_origin/"
target_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_without_sqnet/"

# Init the plot list
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

time_cost = []

if show_server:
    col_name = "server"
else:
    col_name = "client"

'''
Generate the features for convolution layer
model -> str: which network model (eg. "sqnet", "resnet", "densenet121")
startIdx -> int: start index -> range(start, end+1)
endIdx -> int: end index
'''
def generate(model, startIdx, endIdx):
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
                if text[0] == "Current" and text[3] == "start" and text[-3] == "conv":
                    start.append(float(text[-1]))
                if text[0] == "Current" and text[3] == "end" and text[-3] == "conv":
                    end.append(float(text[-1]))
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

        for i in range(len(start)):
            time_cost.append(end[i] - start[i])

if __name__ == "__main__":
    folderpath = "/home/eloise/eloise/result/"
    generate("short1", 1, 15)
    generate("short2", 1, 15)
    generate("resnet50", 1, 15)
    generate("densenet121", 1, 15)
    # generate("sqnet", 1, 15)
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

    # write to file (nan means not valid) 
    # print(relu_coeff[0:5])
    # print(time_cost[0:5])
    # print(CPU_avg[0:5])
    df = pd.DataFrame(list(map(np.array, zip(conv_N, conv_H, conv_W, conv_CI, conv_FH, conv_FW, conv_CO, 
                        conv_zPadHLeft, conv_zPadHRight, conv_zPadWLeft, conv_zPadWRight, conv_strideH, conv_strideW, time_cost))),
                        columns=["conv_N", "conv_H", "conv_W", "conv_CI", "conv_FH", "conv_FW", "conv_CO", 
                        "conv_zPadHLeft", "conv_zPadHRight", "conv_zPadWLeft", "conv_zPadWRight", "conv_strideH", "conv_strideW",'time_cost'], dtype=object)

    print(df.head(30))
    target_filename = target_folder + "conv_onlyTime_" + col_name + ".csv"
    # print(target_filename)
    df.to_csv(target_filename, sep='\t', na_rep=np.nan)