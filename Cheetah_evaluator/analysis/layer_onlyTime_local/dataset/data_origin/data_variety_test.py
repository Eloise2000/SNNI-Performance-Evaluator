import pandas as pd

show_server = 1
if show_server:
    name = "server"
else: 
    name = "client"

# layer_filepath = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/conv_onlyTime_"+name+".csv"
# layer_filepath = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/maxpool_onlyTime_"+name+".csv"
# layer_filepath = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/relu_onlyTime_"+name+".csv"
layer_filepath = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/avgpool_onlyTime_"+name+".csv"
df = pd.read_csv(layer_filepath, delimiter="\s+")

# columns_to_group_by = [
#     "conv_N", "conv_H", "conv_W", "conv_CI",
#     "conv_FH", "conv_FW", "conv_CO",
#     "conv_zPadHLeft", "conv_zPadHRight",
#     "conv_zPadWLeft", "conv_zPadWRight",
#     "conv_strideH", "conv_strideW"
# ]

# columns_to_group_by = ["maxpool_N", "maxpool_H", "maxpool_W", "maxpool_C", "maxpool_ksizeH", "maxpool_ksizeW", 
#                         "maxpool_zPadHLeft", "maxpool_zPadHRight", "maxpool_zPadWLeft", "maxpool_zPadWRight",
#                         "maxpool_strideH", "maxpool_strideW", "maxpool_N1", 
#                         "maxpool_imgH", "maxpool_imgW", "maxpool_C1"]

# columns_to_group_by = ["relu_coeff"]

columns_to_group_by = ["avgpool_N", "avgpool_H", "avgpool_W", "avgpool_C", "avgpool_ksizeH", "avgpool_ksizeW", 
                        "avgpool_zPadHLeft", "avgpool_zPadHRight", "avgpool_zPadWLeft", "avgpool_zPadWRight",
                        "avgpool_strideH", "avgpool_strideW", "avgpool_N1", 
                        "avgpool_imgH", "avgpool_imgW", "avgpool_C1"]

'''Check data'''
# duplicate_counts = df.groupby(columns_to_group_by)
# for name, group in duplicate_counts:
#     print(name)
#     print(group["time_cost"].max() - group["time_cost"].min())
# print(duplicate_counts)

'''Print out'''
duplicate_counts = df.groupby(columns_to_group_by).size().reset_index(name='count')
duplicate_counts.to_csv("duplicated_data_ap.csv", index=False)