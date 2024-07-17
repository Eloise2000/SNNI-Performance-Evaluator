"""
Author: Eloise Zhang
Date: 12 July 2024
Description: This script processes the raw TXT format log data, extracting BatchNormalization Layer features and runtime into a CSV format.
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
bn_N = []
bn_H = []
bn_W = []
bn_C = []
bn_eps = []
bn_momentum = []

bn_start_time=[]
bn_end_time=[]
bn_time_cost=[]

'''
Generate the features for maxpool layer
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

        # Read the layer file and process maxpool layers
        with open(log_filepath) as f:
            for line in f:
                text = line.strip().split()
                if text[0][:-2] == "Layer" and text[1][:-1] == "BatchNorm2d":
                    bn_N.append(int(text[4].split("=")[-1][:-1]))
                    bn_H.append(int(text[5].split("=")[-1][:-1]))
                    bn_W.append(int(text[6].split("=")[-1][:-1]))
                    bn_C.append(int(text[7].split("=")[-1][:-1]))
                    bn_eps.append(float(text[8].split("=")[-1][:-1]))
                    bn_momentum.append(float(text[9].split("=")[-1]))
                    
                    bn_start_time.append(float(text[18]))
                    bn_end_time.append(float(text[21]))
                    # bn_time_cost.append(float("{:10.4f}".format(float(text[29])))) # format to 10.4
                    bn_time_cost.append(float(text[24])) # not format so far

if __name__ == "__main__":
    generate("avgbn_rs17", 1, 5)
    generate("avgbn_rs23", 1, 5)
    generate("avgbn_rs36", 1, 5)
    generate("avgbn_rs42", 1, 5)
    generate("avgbn_rs49", 1, 5)
    generate("avgbn_rs53", 1, 5)
    generate("avgbn_rs62", 1, 5)
    generate("avgbn_rs79", 1, 5)
    generate("avgbn_rs81", 1, 5)
    generate("avgbn_rs97", 1, 5)

    # write to file (nan means not valid) 
    df = pd.DataFrame(list(map(np.array, zip(bn_N, bn_H, bn_W, bn_C, bn_eps, bn_momentum, 
                                    bn_start_time, bn_end_time, bn_time_cost))),
                        columns=["bn_N", "bn_H", "bn_W", "bn_C", "bn_eps", "bn_momentum", 
                                    "bn_start_time", "bn_end_time", "bn_time_cost"], dtype=object)

    print(df.head(30))
    target_filename = target_folder + "bn_onlyTime_" + col_name + ".csv"
    print(target_filename)
    df.to_csv(target_filename, sep='\t', na_rep=np.nan)