import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from joblib import dump

''' Initialize variables here '''
show_server = 0
if show_server:
    name = "server"
else: 
    name = "client"

target_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_origin/"
# target_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_without_sqnet/"
layer_filepath = target_folder + "maxpool_onlyTime_" + name + ".csv"
df = pd.read_csv(layer_filepath, delimiter="\s+")
folder_path = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/predict_regression/LR_model/"
# folder_path = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/predict_regression/LR_model_without_sqnet/"

'''
maxpool columns= "maxpool_N", "maxpool_H", "maxpool_W", "maxpool_C", "maxpool_ksizeH", "maxpool_ksizeW", 
                "maxpool_zPadHLeft", "maxpool_zPadHRight", "maxpool_zPadWLeft", "maxpool_zPadWRight",
                "maxpool_strideH", "maxpool_strideW", "maxpool_N1", 
                "maxpool_imgH", "maxpool_imgW", "maxpool_C1", 'time_cost'
'''

# print(df.head(10))
# X = df.loc[:, ["maxpool_N", "maxpool_H", "maxpool_W", "maxpool_C", "maxpool_ksizeH", "maxpool_ksizeW"]]

### Process on dataset
print("*** Number of data before processing: ", df.size)
# df = df[(df['time_cost'] > 0) &(df['time_cost'] < 20000)]
df = df[df['time_cost'] > 0]
print("*** Number of data after processing: ", df.size)

### Training and testing
X = pd.DataFrame()
mp_H_output = (df['maxpool_H'] + df["maxpool_zPadHLeft"] + df["maxpool_zPadHRight"] - df["maxpool_ksizeH"]) / df["maxpool_strideH"] + 1
mp_W_output = (df['maxpool_W'] + df["maxpool_zPadWLeft"] + df["maxpool_zPadWRight"] - df["maxpool_ksizeW"]) / df["maxpool_strideW"] + 1

'''physical operations'''
X['FLOPs'] = (df["maxpool_ksizeH"] * df["maxpool_ksizeW"]) * (mp_H_output * mp_W_output) * df["maxpool_C"]
# sizeof(uint64_t) is unsigned 64-bit integer -> 8 bytes
X['IN_MACs'] = (df['maxpool_H'] * df['maxpool_W']) * df["maxpool_C"] * 8
X['OUT_MACs'] = (mp_H_output * mp_W_output) * df["maxpool_C"] * 8

for i in range(len(mp_H_output)):
    print(mp_H_output[i], df['maxpool_imgH'][i])