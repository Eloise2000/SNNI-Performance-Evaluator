import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

''' Phases:
- before StartComputation
- StartComputation (precomputation: BaseOT, communication, etc.)
- run layers
'''

# Set variable here
show_server = 1 # 0: client; 1: server

def process(model, startIdx, endIdx):
    model_before_StartComputation = []
    model_StartComputation = []

    folder = folderpath + model + "/"
    for idx in range(startIdx, endIdx+1):
        if show_server:
            log_filepath = folder + "data_" + str(idx) + '/log_server.txt'
        else:
            log_filepath = folder + "data_" + str(idx) + '/log_client.txt'

        # Read the layer file and process Relu layers
        layer_count = 0

        with open(log_filepath) as f:
            for line in f:
                text = line.strip().split()
                if text[0] == "Current" and text[3] == "start" and text[4] == "protocol":
                    time_start_protocol = float(text[-1])
                if text[0] == "Current" and text[3] == "before" and text[4] == "StartComputation":
                    time_before_StartComputation = float(text[-1])
                if text[0] == "Current" and text[3] == "after" and text[4] == "StartComputation":
                    time_after_StartComputation = float(text[-1])
                if text[0] == "Current" and text[3] == "start" and text[5] == "current":
                    layer_count += 1

        model_before_StartComputation.append(time_before_StartComputation - time_start_protocol)
        model_StartComputation.append(time_after_StartComputation - time_before_StartComputation)

    return model_before_StartComputation, model_StartComputation, layer_count

if __name__ == "__main__":
    folderpath = "/home/eloise/eloise/result/"
    short1_before_StartComputation = []
    short2_before_StartComputation = []
    mp2_before_StartComputation = []
    resnet50_before_StartComputation = []
    densenet121_before_StartComputation = []
    sqnet_before_StartComputation = []

    short1_StartComputation = []
    short2_StartComputation = []
    mp2_StartComputation = []
    resnet50_StartComputation = []
    densenet121_StartComputation = []
    sqnet_StartComputation = []

    short1_before_StartComputation, short1_StartComputation, short1_layer = process("short1", 1, 20)
    short2_before_StartComputation, short2_StartComputation, short2_layer = process("short2", 1, 20)
    mp2_before_StartComputation, mp2_StartComputation, mp2_layer = process("mp2", 1, 20)
    resnet_before_StartComputation, resnet_StartComputation, resnet50_layer = process("resnet50", 1, 7)
    densenet121_before_StartComputation, densenet121_StartComputation, densenet121_layer = process("densenet121", 1, 10)
    sqnet_before_StartComputation, sqnet_StartComputation, sqnet_layer = process("sqnet", 1, 20)

    boxplot_data_before_StartComputation = [short1_before_StartComputation, short2_before_StartComputation, mp2_before_StartComputation, resnet_before_StartComputation, densenet121_before_StartComputation, sqnet_before_StartComputation]
    boxplot_data_StartComputation = [short1_StartComputation, short2_StartComputation, mp2_StartComputation, resnet_StartComputation, densenet121_StartComputation, sqnet_StartComputation]

    # Plot boxplot_data_before_StartComputation
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, 1, 1)
    labels = [short1_layer, short2_layer, mp2_layer, resnet50_layer, densenet121_layer, sqnet_layer]

    ax.boxplot(boxplot_data_before_StartComputation, labels = labels, showmeans=True)
    ax.set_xlabel("Network size (num of layers)")
    ax.set_ylabel("Time (ms)")
    # ax.set_xticklabels( labels, rotation=45 )
    plt.title("Before StartComputation Phase")
    plt.show()
    plt.savefig("Before_StartComputation_phase.png")

    # Plot boxplot_data_before_StartComputation
    # fig = plt.figure(figsize=(15, 6))
    # ax = fig.add_subplot(1, 1, 1)
    # labels = [short1_layer, short2_layer, mp2_layer, resnet50_layer, densenet121_layer, sqnet_layer]

    # ax.boxplot(boxplot_data_StartComputation, labels = labels, showmeans=True)
    # ax.set_xlabel("Network size (num of layers)")
    # ax.set_ylabel("Time (ms)")
    # # ax.set_xticklabels( labels, rotation=45 )
    # plt.title("StartComputation Phase")
    # plt.show()
    # plt.savefig("StartComputation_phase.png")
