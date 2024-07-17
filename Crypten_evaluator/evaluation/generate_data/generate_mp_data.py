"""
Author: Eloise Zhang
Date: 12 July 2024
Description: This script processes the raw TXT format log data, extracting MaxPool Layer features and runtime into a CSV format.
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
mp_N = []
mp_HI = []
mp_WI = []
mp_CI = []
mp_HO = []
mp_WO = []
mp_CO = []
mp_F = []
mp_S = []
mp_Padding=[]
mp_ceil_mode = []

mp_start_time=[]
mp_end_time=[]
mp_time_cost=[]

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
                if text[0][:-2] == "Layer" and text[1][:-1] == "MaxPool":
                    mp_N.append(int(text[4].split("=")[-1][:-1]))
                    mp_HI.append(int(text[5].split("=")[-1][:-1]))
                    mp_WI.append(int(text[6].split("=")[-1][:-1]))
                    mp_CI.append(int(text[7].split("=")[-1][:-1]))
                    mp_HO.append(int(text[8].split("=")[-1][:-1]))
                    mp_WO.append(int(text[9].split("=")[-1][:-1]))
                    mp_CO.append(int(text[10].split("=")[-1][:-1]))
                    mp_F.append(int(text[11].split("=")[-1][:-1]))
                    mp_S.append(int(text[12].split("=")[-1][:-1]))
                    mp_Padding.append(text[13].split("=")[-1][:-1]) # all 0
                    mp_ceil_mode.append(text[14].split("=")[-1])

                    mp_start_time.append(float(text[23]))
                    mp_end_time.append(float(text[26]))
                    # mp_time_cost.append(float("{:10.4f}".format(float(text[29])))) # format to 10.4
                    mp_time_cost.append(float(text[29])) # not format so far

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
    df = pd.DataFrame(list(map(np.array, zip(mp_N, mp_HI, mp_WI, mp_CI, mp_HO, mp_WO, mp_CO, 
                                    mp_F, mp_S, mp_Padding, mp_ceil_mode,
                                    mp_start_time, mp_end_time, mp_time_cost))),
                        columns=["mp_N", "mp_HI", "mp_WI", "mp_CI", "mp_HO", "mp_WO", "mp_CO", 
                                    "mp_F", "mp_S", "mp_Padding", "mp_ceil_mode",
                                    "mp_start_time", "mp_end_time", "mp_time_cost"], dtype=object)

    print(df.head(30))
    target_filename = target_folder + "maxpool_onlyTime_" + col_name + ".csv"
    print(target_filename)
    df.to_csv(target_filename, sep='\t', na_rep=np.nan)