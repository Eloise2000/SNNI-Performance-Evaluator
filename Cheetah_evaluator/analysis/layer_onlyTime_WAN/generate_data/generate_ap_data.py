import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Init the plot list
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

time_cost = []

# Set variable here
show_server = 1 # 0: client; 1: server
target_folder = "/home/eloise/eloise/script/analysis/layer_onlyTime_WAN/dataset/data_layer/"

if show_server:
    col_name = "server"
    folderpath = "/home/eloise/eloise/result_WAN/result-server/"
else:
    col_name = "client"
    folderpath = "/home/eloise/eloise/result_WAN/result-client/"

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
                if text[0] == "Current" and text[3] == "start" and text[-3] == "avgpool":
                    start.append(float(text[-1]))
                if text[0] == "Current" and text[3] == "end" and text[-3] == "avgpool":
                    end.append(float(text[-1]))
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

        for i in range(len(start)):
            time_cost.append(end[i] - start[i])

if __name__ == "__main__":
    # Cheetah
    generate("short1", 1, 10)
    generate("short2", 1, 10)
    generate("resnet50", 1, 10)
    generate("densenet121", 1, 10)
    generate("sqnet", 1, 10)
    generate("conv1", 1, 10)
    generate("conv2", 1, 10)
    generate("conv3", 1, 10)
    generate("conv4", 1, 10)
    generate("conv5", 1, 10)
    generate("conv6", 1, 10)
    generate("conv7", 1, 10)
    generate("conv8", 1, 10)
    generate("conv9", 1, 10)
    generate("ap1", 1, 10)
    generate("ap2", 1, 10)
    generate("ap3", 1, 10)
    generate("ap4", 1, 10)
    generate("ap5", 1, 10)
    generate("ap6", 1, 10)
    generate("ap7", 1, 10)
    generate("ap8", 1, 10)
    generate("ap9", 1, 10)
    generate("convap1", 1, 10)
    generate("convap2", 1, 10)
    generate("convap3", 1, 10)
    generate("convap4", 1, 10)
    generate("convap5", 1, 10)
    generate("convap6", 1, 10)
    generate("convap7", 1, 10)
    generate("convap8", 1, 10)
    generate("convap9", 1, 10)
    generate("convmp1", 1, 10)
    generate("convmp2", 1, 10)
    generate("convmp3", 1, 10)
    generate("convmp4", 1, 10)
    generate("convmp5", 1, 10)
    generate("mp1", 1, 10)
    generate("mp2", 1, 10)
    generate("mp3", 1, 10)
    generate("mp4", 1, 10)
    generate("mp5", 1, 10)
    generate("mp6", 1, 10)
    generate("mp7", 1, 10)
    generate("mp8", 1, 10)
    generate("mp9", 1, 10)

    df = pd.DataFrame(list(map(np.array, zip(avgpool_N, avgpool_H, avgpool_W, avgpool_C, avgpool_ksizeH, avgpool_ksizeW, 
                        avgpool_zPadHLeft, avgpool_zPadHRight, avgpool_zPadWLeft, avgpool_zPadWRight,
                        avgpool_strideH, avgpool_strideW, avgpool_N1, 
                        avgpool_imgH, avgpool_imgW, avgpool_C1, time_cost))),
                        columns=["avgpool_N", "avgpool_H", "avgpool_W", "avgpool_C", "avgpool_ksizeH", "avgpool_ksizeW", 
                        "avgpool_zPadHLeft", "avgpool_zPadHRight", "avgpool_zPadWLeft", "avgpool_zPadWRight",
                        "avgpool_strideH", "avgpool_strideW", "avgpool_N1", 
                        "avgpool_imgH", "avgpool_imgW", "avgpool_C1", 'time_cost'], dtype=object)

    print(df.head(30))
    target_filename = target_folder + "avgpool_onlyTime_" + col_name + ".csv"
    df.to_csv(target_filename, sep='\t', na_rep=np.nan)