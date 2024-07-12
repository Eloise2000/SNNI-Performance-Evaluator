import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Init the plot list
relu_coeff = []
time_cost = []

# Set variable here
show_server = 0 # 0: client; 1: server
target_folder = "/home/eloise/eloise/script/analysis/layer_onlyTime_LAN/dataset/data_layer/"

if show_server:
    col_name = "server"
    folderpath = "/home/eloise/eloise/result_LAN/result-server/"
else:
    col_name = "client"
    folderpath = "/home/eloise/eloise/result_LAN/result-client/"

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

        # Read the layer file and process Relu layers
        start = []
        end = []

        with open(log_filepath) as f:
            for line in f:
                text = line.strip().split()
                if text[0] == "Current" and text[3] == "start" and text[-3] == "relu":
                    start.append(float(text[-1]))
                if text[0] == "Current" and text[3] == "end" and text[-3] == "relu":
                    end.append(float(text[-1]))
                if text[0] == "Relu" and text[1][0] == "#":
                    relu_coeff.append(int(text[3]))

        for i in range(len(start)):
            time_cost.append(end[i] - start[i])

if __name__ == "__main__":
    # Cheetah
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

    df = pd.DataFrame(list(map(np.array, zip(relu_coeff, time_cost))),
                        columns=['relu_coeff','time_cost'], dtype=object)

    print(df.head(30))
    target_filename = target_folder + "relu_onlyTime_" + col_name + ".csv"
    df.to_csv(target_filename, sep='\t', na_rep=np.nan)