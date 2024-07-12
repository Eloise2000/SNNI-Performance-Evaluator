import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Init the plot list
trunc_coeff = []
time_cost = []

# Set variable here
show_server = 1 # 0: client; 1: server
target_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_origin/"
# target_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_without_sqnet/"

if show_server:
    col_name = "server"
else:
    col_name = "client"

'''
Generate the features for relu layer
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

        # Read the layer file and process trunc layers
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
                        time_cost.append(end - start)
                        Flag = False
                if text[0] == "Current" and text[3] == "start":
                    if Flag:
                        end = float(text[-1])
                        time_cost.append(end - start)
                        Flag = False
                if text[0] == "Truncate" and text[1][0] == "#":
                    trunc_coeff.append(int(text[3]))
                    Flag = True

if __name__ == "__main__":
    # Cheetah
    folderpath = "/home/eloise/eloise/result/"
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

    df = pd.DataFrame(list(map(np.array, zip(trunc_coeff, time_cost))),
                        columns=['trunc_coeff','time_cost'], dtype=object)

    print(df.head(30))
    target_filename = target_folder + "trunc_onlyTime_" + col_name + ".csv"
    df.to_csv(target_filename, sep='\t', na_rep=np.nan)