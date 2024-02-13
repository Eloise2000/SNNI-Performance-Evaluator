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

target_folder = "/home/eloise/eloise/script/analysis/layer_onlyTime_LAN/dataset/data_layer/"
layer_filepath_BN1 = target_folder + "BN1_onlyTime_" + name + ".csv"
layer_filepath_BN2 = target_folder + "BN2_onlyTime_3D_" + name + ".csv"
df_BN1 = pd.read_csv(layer_filepath_BN1, delimiter="\s+")
df_BN2 = pd.read_csv(layer_filepath_BN2, delimiter="\s+")
folder_path = "/home/eloise/eloise/script/analysis/layer_onlyTime_LAN/predict_regression/LR_model/"

'''"BN1_C", "BN1_H", "BN1_W", "time_cost"'''
'''"BN2_C", "BN2_H", "BN2_W", "time_cost"'''

# print(df.head(10))
# X = df.loc[:, ["conv_N", "conv_H", "conv_W", "conv_CI", "conv_FH", "conv_FW", "conv_CO", "conv_S", "conv_Padding"]]

### Process on dataset
print("*** Number of BN1 data before processing: ", df_BN1.shape)
df_BN1 = df_BN1[df_BN1['time_cost'] > 0]
print("*** Number of BN1 data after processing: ", df_BN1.shape)

print("*** Number of BN2 data before processing: ", df_BN2.shape)
df_BN2 = df_BN2[df_BN2['time_cost'] > 0]
print("*** Number of BN2 data after processing: ", df_BN2.shape)

### Training and testing
X = pd.DataFrame()

'''physical operations'''
df_BN1 = df_BN1.rename(columns={'BN1_C': 'C', 'BN1_H': 'H', 'BN1_W': 'W'})
df_BN2 = df_BN2.rename(columns={'BN2_C': 'C', 'BN2_H': 'H', 'BN2_W': 'W'})
df = pd.concat([df_BN1, df_BN2], axis = 0, ignore_index=True)

X['FLOPs'] = df["C"] * df["H"] * df["W"]
X['IN_MACs'] = df["C"] * df["H"] * df["W"]
X['PAR_MACs'] = df["C"]
X['OUT_MACs'] = df["C"] * df["H"] * df["W"]

y = df['time_cost']
# print(X)
# print(y)

# Normalize the features in X
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# plt.scatter(X, y, color='red')
# # plt.scatter(X_normalized, y_pred, color='blue', label='Predicted vs. Actual')
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# # plt.title('Actual vs. Predicted Values')
# plt.legend()
# plt.savefig('scatter_plot5.png')  # Provide the desired file name and extension


# Shuffle the data for 5-cross validation
X_normalized, y = shuffle(X_normalized, y, random_state=1)

reg = LinearRegression(positive=True).fit(X_normalized, y)
# print("score is: ", reg.score(X_normalized, y))

''' Dump the result '''
# dump(reg, folder_path + 'BN_' + name + '_LR_model.joblib')
# dump(scaler, folder_path + 'BN_' + name + '_scaler.joblib')

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

## Create a scatter plot
# plt.scatter(X_normalized, y, color='red', label='Actual')
# plt.scatter(X_normalized, y_pred, color='blue', label='Predicted')
# plt.xlabel('BN neurons (normalized)')
# plt.ylabel('Time Cost (ms)')
# plt.title('Actual vs. Predicted Values')
# plt.legend()
# plt.savefig('scatter_plot_BN.png')  # Provide the desired file name and extension


''' Measure prediction time '''
df_row = df.iloc[3]
# Create a single data point for prediction
X_single = pd.DataFrame()

'''physical operations'''
X_single['FLOPs'] = [df_row["C"] * df_row["H"] * df_row["W"]]
X_single['IN_MACs'] = [df_row["C"] * df_row["H"] * df_row["W"]]
X_single['PAR_MACs'] = [df_row["C"]]
X_single['OUT_MACs'] = [df_row["C"] * df_row["H"] * df_row["W"]]

# Start prediction
start_time = time.time()
X_single_normalized = scaler.transform(X_single)

# Predict the time for the single data point
predicted_time = reg.predict(X_single_normalized)
end_time = time.time()

real_time = df_row['time_cost']
# Calculate the time taken for prediction
prediction_time = end_time - start_time

# Print the predicted time and prediction time
print("Predicted Time:", predicted_time)
print("Real Time:", real_time)
print("Diff Time:", real_time - predicted_time)
print("Prediction Time:", prediction_time * 1000, "ms")
