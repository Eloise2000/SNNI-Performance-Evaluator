import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from joblib import dump
import time

''' Initialize variables here '''
show_server = 1
if show_server:
    name = "server"
else: 
    name = "client"

target_folder = "/home/eloise/eloise/script/analysis/layer_onlyTime_WAN/dataset/data_layer/"
folder_path = "/home/eloise/eloise/script/analysis/layer_onlyTime_WAN/predict_regression/LR_model/"

layer_filepath = target_folder + "conv_onlyTime_" + name + ".csv"
df = pd.read_csv(layer_filepath, delimiter="\s+")

'''"conv_N", "conv_H", "conv_W", "conv_CI", "conv_FH", "conv_FW", "conv_CO", 
    "conv_zPadHLeft", "conv_zPadHRight", "conv_zPadWLeft", "conv_zPadWRight", 
    "conv_strideH", "conv_strideW","time_cost"'''

# print(df.head(10))
# X = df.loc[:, ["conv_N", "conv_H", "conv_W", "conv_CI", "conv_FH", "conv_FW", "conv_CO", "conv_S", "conv_Padding"]]

### Process on dataset
print("*** Number of data before processing: ", df.size)
df = df[df['time_cost'] > 0]
print("*** Number of data after processing: ", df.size)

### Training and testing
X = pd.DataFrame()
# X = df.loc[:, ["conv_N", "conv_H", "conv_W", "conv_CI", "conv_FH", "conv_FW", "conv_CO", 
#     "conv_zPadHLeft", "conv_zPadHRight", "conv_zPadWLeft", "conv_zPadWRight", 
#     "conv_strideH", "conv_strideW"]]

conv_H_output = (df['conv_H'] + df["conv_zPadHLeft"] + df["conv_zPadHRight"] - df["conv_FH"]) / df["conv_strideH"] + 1
conv_W_output = (df['conv_W'] + df["conv_zPadWLeft"] + df["conv_zPadWRight"] - df["conv_FW"]) / df["conv_strideW"] + 1

'''features of the input vector'''
# X['HW'] = df['conv_H'] * df['conv_W']
# X["conv_CI"] = df["conv_CI"]
# X['HoutWout'] = conv_H_output * conv_W_output
# X['FHFW'] = df["conv_FH"] * df["conv_FW"]
# X['strideHstrideW"'] = df["conv_strideH"] * df["conv_strideW"]

'''physical operations'''
'''first attempt
X['FLOPs'] = (2 * df["conv_FH"] * df["conv_FW"] * df["conv_CI"]) * conv_H_output * conv_W_output * df["conv_CO"]
X['IN_MACs'] = (df['conv_H'] * df['conv_W'] * df["conv_CI"]) * 8
X['PAR_MACs'] = (df["conv_CI"] * df["conv_FH"] * df["conv_FW"] * df["conv_CO"]) * 8
X['OUT_MACs'] = (conv_H_output * conv_W_output * df["conv_CO"]) * 8
'''

X['FLOPs'] = (2 * df["conv_FH"] * df["conv_FW"] * df["conv_CI"]) * conv_H_output * conv_W_output * df["conv_CO"]
X['IN_MACs'] = (df['conv_H'] * df['conv_W'] * df["conv_CI"]) * 8
X['PAR_MACs'] = (df["conv_CI"] * df["conv_FH"] * df["conv_FW"] * df["conv_CO"]) * 8
X['OUT_MACs'] = (conv_H_output * conv_W_output * df["conv_CO"]) * 8

'''
X['FLOPs'] = (2 * df["conv_FH"] * df["conv_FW"] * df["conv_CI"]) * conv_H_output * conv_W_output * df["conv_CO"]
# sizeof(uint64_t) is unsigned 64-bit integer -> 8 bytes
X['IN_MACs'] = (df['conv_H'] * df['conv_W'] * df["conv_CI"]) * 8
# X['PAR_MACs'] = (df["conv_CI"] * df["conv_CO"] * (df["conv_FH"] * df["conv_FW"] + 1)) * 8
# X['PAR_MACs'] = (df["conv_CI"] * df["conv_FH"] * df["conv_FW"] * df["conv_CO"] + df["conv_CO"]) * 8
X['PAR_MACs'] = (df["conv_CI"] * df["conv_FH"] * df["conv_FW"] * df["conv_CO"]) * 8
X['PAR_MACs2'] = df["conv_CO"]
# X['PAR_MACs3'] = (df["conv_CI"] * df["conv_CO"]) * 8
X['OUT_MACs'] = (conv_H_output * conv_W_output * df["conv_CO"]) * 8
# X['IN'] = df["conv_CI"]
# X['OUT'] = df["conv_CO"] # 0.82
# X['INOUT'] = df["conv_CI"] * df["conv_CO"] # 0.79
'''

'''protocol related'''
# X['NHWCI'] = df['conv_N'] * df['conv_H'] * df['conv_W'] * df['conv_CI'] # From code
# X['FWFHCICO'] = df["conv_FH"] * df["conv_FW"] * df["conv_CI"] * df["conv_CO"] # From code
# X['NHoutWoutCO'] = df['conv_N'] * conv_H_output * conv_W_output * df["conv_CO"] # From code

y = df['time_cost']

# Normalize the features in X
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Shuffle the data for 5-cross validation
X_normalized, y = shuffle(X_normalized, y, random_state=1)

reg = LinearRegression(positive=True).fit(X_normalized, y)
# print("score is: ", reg.score(X_normalized, y))

''' Dump the result '''
# dump(reg, folder_path + 'conv_' + name + '_LR_model.joblib')
# dump(scaler, folder_path + 'conv_' + name + '_scaler.joblib')

# Get the regression coefficients
coefficients = reg.coef_

# Print the coefficients
print("*** Regression Coefficients ***")
for i, coef in enumerate(coefficients):
    print("Coefficient for X_normalized{}: {}".format(i+1, coef))
print("Intercept: {}".format(reg.intercept_))

print("*** Results ***")

# Calculate the mean of y
mean_y = np.mean(y)
print("Mean of y:", mean_y)

# Calculate Mean Absolute Error (MAE)
y_pred = reg.predict(X_normalized)
mae = mean_absolute_error(y, y_pred)
print("Mean Absolute Error (MAE):", mae)

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y - y_pred) / y)) * 100
print("Mean Absolute Percentage Error (MAPE):", mape)


# plt.scatter(y, np.abs((y - y_pred) / y) * 100, color='blue', label='Predicted vs. Actual')
# plt.xlabel('Actual')
# plt.ylabel('MAPE')
# plt.title('Actual vs. Mean Absolute Percentage Error (MAPE) Values')
# plt.legend()
# plt.savefig('mape_conv_CICO.png')  # Provide the desired file name and extension
# plt.savefig('mape_conv_MoreThan1000.png')  # Provide the desired file name and extension
# plt.savefig('mape_conv_MoreThan500.png')  # Provide the desired file name and extension
# plt.savefig('mape_conv_all.png')  # Provide the desired file name and extension


# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate Root Mean Square Percentage Error (RMSPE)
rmspe = np.sqrt(np.mean(np.square(((y - y_pred) / y)), axis=0)) * 100
print("Root Mean Square Percentage Error (RMSPE):", rmspe)

# Calculate R2 Score
r2 = r2_score(y, y_pred)
print("R2 Score:", r2)

# Perform 5-fold cross-validation on the training set
cv_scores_mae = cross_val_score(reg, X_normalized, y, cv=5, scoring='neg_mean_absolute_error')
mae_cv = -cv_scores_mae.mean()

cv_scores = cross_val_score(reg, X_normalized, y, cv=5, scoring='neg_mean_absolute_percentage_error')
mape_cv = -cv_scores.mean() * 100

# Perform 5-fold cross-validation on the training set for RMSE
cv_scores_rmse = cross_val_score(reg, X_normalized, y, cv=5, scoring='neg_mean_squared_error')
rmse_cv = np.sqrt(-cv_scores_rmse.mean())

# Define a function to calculate RMSPE
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2)) * 100

# Perform 5-fold cross-validation on the training set for RMSPE
cv_scores_rmspe = cross_val_score(reg, X_normalized, y, cv=5, scoring=make_scorer(rmspe, greater_is_better=False))
rmspe_cv = -np.mean(cv_scores_rmspe)

cv_scores_r2 = cross_val_score(reg, X_normalized, y, cv=5, scoring='r2')
r2_cv = cv_scores_r2.mean()

print("*** 5-fold Cross-Validation Results ***")
print("5-fold cross-validation result (MAE): ", mae_cv)
print("5-fold cross-validation result (MAPE): ", mape_cv)
print("5-fold cross-validation result (RMSE): ", rmse_cv)
print("5-fold cross-validation result (RMSPE): ", rmspe_cv)
print("5-fold cross-validation result (R2): ", r2_cv)

### Create a scatter plot
# plt.scatter(y, y_pred-y, color='blue', label='Predicted vs. Actual')
# plt.scatter(y, y_pred, color='blue', label='Predicted vs. Actual')
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.title('Actual vs. Predicted Values')
# plt.legend()
# plt.savefig('scatter_plot2.png')  # Provide the desired file name and extension

''' Measure prediction time '''
df_row = df.iloc[30]
# Create a single data point for prediction
X_single = pd.DataFrame()

# Start prediction
start_time = time.time()

conv_H_output_row = (df_row['conv_H'] + df_row["conv_zPadHLeft"] + df_row["conv_zPadHRight"] - df_row["conv_FH"]) / df_row["conv_strideH"] + 1
conv_W_output_row = (df_row['conv_W'] + df_row["conv_zPadWLeft"] + df_row["conv_zPadWRight"] - df_row["conv_FW"]) / df_row["conv_strideW"] + 1

'''physical operations'''
X_single['FLOPs'] = [(2 * df_row["conv_FH"] * df_row["conv_FW"] * df_row["conv_CI"]) * conv_H_output_row * conv_W_output_row * df_row["conv_CO"]]
X_single['IN_MACs'] = [(df_row['conv_H'] * df_row['conv_W'] * df_row["conv_CI"]) * 8]
X_single['PAR_MACs'] = [(df_row["conv_CI"] * df_row["conv_FH"] * df_row["conv_FW"] * df_row["conv_CO"]) * 8]
X_single['OUT_MACs'] = [(conv_H_output_row * conv_W_output_row * df_row["conv_CO"]) * 8]

# X_single['FLOPs'] = (2 * df_row["conv_FH"] * df_row["conv_FW"] * df_row["conv_CI"]) * conv_H_output_row * conv_W_output_row * df_row["conv_CO"]
# X_single['IN_MACs'] = (df_row['conv_H'] * df_row['conv_W'] * df_row["conv_CI"]) * 8
# X_single['PAR_MACs'] = (df_row["conv_CI"] * df_row["conv_FH"] * df_row["conv_FW"] * df_row["conv_CO"]) * 8
# X_single['OUT_MACs'] = (conv_H_output_row * conv_W_output_row * df_row["conv_CO"]) * 8

X_single_normalized = scaler.transform(X_single)

# Predict the time for the single data point
predicted_time = reg.predict(X_single_normalized)
end_time = time.time()

real_time = df_row['time_cost']
# Calculate the time taken for prediction
prediction_time = end_time - start_time

# print(start_time)
# print(end_time)
# Print the predicted time and prediction time
print("Predicted Time:", predicted_time)
print("Real Time:", real_time)
print("Diff Time:", real_time - predicted_time)
print("Prediction Time:", prediction_time * 1000, "ms")

### Old Experiments
# X["CI-1"] = df["conv_CI"] - 1
# X['addition'] = conv_H_output * conv_W_output * (df["conv_FH"] * df["conv_FW"] - 1) * df["conv_CI"] * df["conv_CO"]
# X['multiplication'] = conv_H_output * conv_W_output * (df["conv_FH"] * df["conv_FW"]) * df["conv_CI"] * df["conv_CO"]


'''
funcReconstruct2PCCons(nullptr, inputArr, N * H * W * CI);
funcReconstruct2PCCons(nullptr, filterArr, FH * FW * CI * CO);
funcReconstruct2PCCons(nullptr, outArr, N * newH * newW * CO);
'''

# X = pd.DataFrame()
# X['HW'] = df['conv_H'] * df['conv_W']
# # X['HWPadding'] = df['conv_H'] * df['conv_W'] * df['conv_Padding']
# # X['HWFHFW'] = df['conv_H'] * df['conv_W'] * df["conv_FH"] * df["conv_FW"]

# X['FHFWCI'] = df["conv_FH"] * df["conv_FW"] * df["conv_CI"]
# X['FHFWCO'] = df["conv_FH"] * df["conv_FW"] * df["conv_CO"]
# # X['CICO'] = df["conv_CI"] * df["conv_CO"]
# # X['PaddingCICO'] = df['conv_Padding'] * df["conv_CI"] * df["conv_CO"]

# X['PaddingCI'] = df['conv_Padding'] * df["conv_CI"]
# X['PaddingCO'] = df['conv_Padding'] * df["conv_CO"]
# X['PaddingFHFW'] = df['conv_Padding'] * df["conv_FH"] * df["conv_FW"]

# X['H/S'] = df['conv_H'] / df["conv_S"]
# X['W/S'] = df['conv_W'] / df["conv_S"]
# X['CICO/S'] = df["conv_CI"] * df["conv_CO"] / df['conv_S']
# # X['FHFW/S'] = df["conv_FH"] * df["conv_FW"] / df['conv_S']

# X['Padding/S'] = df['conv_Padding'] / df["conv_S"]

# # X['HCO'] = df['conv_H'] * df['conv_CO']
# # X['NHWCO'] = df['conv_N'] * df['conv_H'] * df['conv_W'] * df['conv_CO']

# conv_H_output = (df['conv_H'] + 2*df["conv_Padding"] - df["conv_FH"]) / df["conv_S"] + 1
# conv_W_output = (df['conv_W'] + 2*df["conv_Padding"] - df["conv_FW"]) / df["conv_S"] + 1
# X['NHWCI'] = df['conv_N'] * df['conv_H'] * df['conv_W'] * df['conv_CI'] # From code
# X['HCI'] = df['conv_H'] * df['conv_CI']
# # X['WCI'] = df['conv_W'] * df['conv_CI']
# X['HWCO'] = df['conv_H'] * df['conv_W'] * df['conv_CO']
# X['HWCICO'] = df['conv_H'] * df['conv_W'] * df['conv_CI'] * df['conv_CO']
# X['HWCICO/S'] = df['conv_H'] * df['conv_W'] * df['conv_CI'] * df['conv_CO'] / df["conv_S"]
# # X['HCICO'] = df['conv_H'] * df['conv_CI'] * df['conv_CO']
# X['FWFHCICO'] = df["conv_FH"] * df["conv_FW"] * df["conv_CI"] * df["conv_CO"] # From code
# X['FWFHCICO/S'] = df["conv_FH"] * df["conv_FW"] * df["conv_CI"] * df["conv_CO"] / df["conv_S"]
# # X['FWCICO/S'] = df["conv_FW"] * df["conv_CI"] * df["conv_CO"] / df["conv_S"]
# # X['FHCICO/S'] = df["conv_FH"] * df["conv_CI"] * df["conv_CO"] / df["conv_S"]
# X['NHoutWoutCO'] = df['conv_N'] * conv_H_output * conv_W_output * df["conv_CO"] # From code

# # X['conv_Hout'] = conv_H_output
# # X['conv_Wout'] = conv_W_output

# # X['HoutWout'] = conv_H_output * conv_W_output