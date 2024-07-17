"""
Author: Eloise Zhang
Date: 12 July 2024
Description: This script processes the raw TXT format log data, extracting Convolutional features and runtime into a CSV format.
"""

import pandas as pd
import numpy as np

# Set variable here
show_server = 1 # 0: client; 1: server

target_folder = "./data_WAN/"

if show_server:
    col_name = "server"
    folderpath = "./result_server_WAN/"
else:
    col_name = "client"
    folderpath = "./result_client_WAN/"

# Init the plot list
conv_N = []
conv_HI = []
conv_WI = []
conv_CI = []
conv_HO = []
conv_WO = []
conv_CO = []
conv_FH = []
conv_FW = []
conv_S = []
conv_Padding=[]

conv_start_time=[]
conv_end_time=[]
conv_time_cost=[]

'''
Generate the features for convolution layer
model -> str: which network model (eg. "convmp_rs17", "convmp_rs23")
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

        # Read the layer file and process conv layers
        with open(log_filepath) as f:
            for line in f:
                text = line.strip().split()
                if text[0][:-2] == "Layer" and text[1][:-1] == "Conv":
                    conv_N.append(int(text[4].split("=")[-1][:-1]))
                    conv_HI.append(int(text[5].split("=")[-1][:-1]))
                    conv_WI.append(int(text[6].split("=")[-1][:-1]))
                    conv_CI.append(int(text[7].split("=")[-1][:-1]))
                    conv_HO.append(int(text[8].split("=")[-1][:-1]))
                    conv_WO.append(int(text[9].split("=")[-1][:-1]))
                    conv_CO.append(int(text[10].split("=")[-1][:-1]))
                    conv_FH.append(int(text[11][-2]))
                    conv_FW.append(int(text[12][0]))
                    conv_S.append(int(text[13].split("=")[-1][:-1]))
                    conv_Padding.append(text[14].split("=")[-1]) # "valid"/"same"
                    conv_start_time.append(float(text[23]))
                    conv_end_time.append(float(text[26]))
                    # conv_time_cost.append(float("{:10.4f}".format(float(text[29])))) # format to 10.4
                    conv_time_cost.append(float(text[29])) # not format so far

if __name__ == "__main__":
    generate("convmp_rs17", 1, 5)
    generate("convmp_rs23", 1, 5)
    generate("convmp_rs36", 1, 5)
    generate("convmp_rs42", 1, 5)
    generate("convmp_rs49", 1, 5)
    generate("convmp_rs53", 1, 5)
    generate("convmp_rs62", 1, 5)
    generate("convmp_rs79", 1, 5)
    generate("convmp_rs81", 1, 5)
    generate("convmp_rs97", 1, 5)

    # write to file (nan means not valid) 
    df = pd.DataFrame(list(map(np.array, zip(conv_N, conv_HI, conv_WI, conv_CI, conv_HO, conv_WO, conv_CO, 
                                    conv_FH, conv_FW, conv_S, conv_Padding,
                                    conv_start_time, conv_end_time, conv_time_cost))),
                        columns=["conv_N", "conv_HI", "conv_WI", "conv_CI", "conv_HO", "conv_WO", "conv_CO", 
                                    "conv_FH", "conv_FW", "conv_S", "conv_Padding",
                                    "conv_start_time", "conv_end_time", "conv_time_cost"], dtype=object)

    print(df.head(30))
    target_filename = target_folder + "conv_onlyTime_" + col_name + ".csv"
    print(target_filename)
    df.to_csv(target_filename, sep='\t', na_rep=np.nan)