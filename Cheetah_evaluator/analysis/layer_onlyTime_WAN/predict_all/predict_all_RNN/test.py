import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from joblib import load
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Set variable here
show_server = 1 # 0: client; 1: server
RNN_folderPath = "/home/eloise/eloise/script/analysis/layer_onlyTime_WAN/predict_RNN/"

if show_server:
    name = "server"
    folderpath = "/home/eloise/eloise/result_WAN/result-server/"
else:
    name = "client"
    folderpath = "/home/eloise/eloise/result_WAN/result-client/"

'''
Generate the features for whole network (samples_sparse, time_steps, features)
model -> str: which network model (eg. "sqnet", "resnet", "densenet121")
startIdx -> int: start index -> range(start, end+1)
endIdx -> int: end index
'''
def generate(model, startIdx, endIdx):
    Flag_trunc = False
    folder = folderpath + model + "/"
    total_time = 0
    for idx in range(startIdx, endIdx+1):
        log_filepath = folder + "data_" + str(idx) + '/log_' + name + '.txt'

        with open(log_filepath) as f:
            # Initialize a list to store the time steps for the current sample
            time_steps_for_sample = []
            time_seq_for_sample = []
            for line in f:
                text = line.strip().split()
                # First decode the start layers: "Current time of after StartComputation = 1691397053779"
                if text[0] == "Current" and text[3] == "after" and text[4] == "StartComputation":
                    start_all_layers = float(text[-1])
                # Last decode the end of protocol: "Current time of end protocol = 1691397285055"
                if text[0] == "Current" and text[3] == "end" and text[4] == "protocol":
                    end_all_layers = float(text[-1])

        total_time += end_all_layers - start_all_layers

    return total_time/(endIdx - startIdx + 1)

if __name__ == "__main__":
    # avg_time = generate("densenet121", 1, 10)
    avg_time = generate("resnet50", 1, 10)
    print("avg time is:", avg_time)