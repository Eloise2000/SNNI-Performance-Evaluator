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

layer_filepath = target_folder + "maxpool_onlyTime_" + name + ".csv"
df = pd.read_csv(layer_filepath, delimiter="\s+")

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
# (df['maxpool_imgH'] + df["maxpool_zPadHLeft"] + df["maxpool_zPadHRight"] - df["maxpool_ksizeH"]) / df["maxpool_strideH"] + 1
mp_H_output = df['maxpool_H']
# (df['maxpool_imgW'] + df["maxpool_zPadWLeft"] + df["maxpool_zPadWRight"] - df["maxpool_ksizeW"]) / df["maxpool_strideW"] + 1
mp_W_output = df['maxpool_W']

'''physical operations'''
X['FLOPs'] = (df["maxpool_ksizeH"] * df["maxpool_ksizeW"]) * (mp_H_output * mp_W_output) * df["maxpool_C"]
# sizeof(uint64_t) is unsigned 64-bit integer -> 8 bytes
X['IN_MACs'] = (df['maxpool_imgH'] * df['maxpool_imgW']) * df["maxpool_C"] * 8
X['OUT_MACs'] = (mp_H_output * mp_W_output) * df["maxpool_C"] * 8

y = df['time_cost']

# Normalize the features in X
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Shuffle the data for 10-cross validation
X_normalized, y = shuffle(X_normalized, y, random_state=1)

# reg = LinearRegression(positive=True).fit(X_normalized, y)
reg = LinearRegression(positive=True, fit_intercept=False).fit(X_normalized, y)
# reg = Lasso(positive=True).fit(X_normalized, y)
# print("score is: ", reg.score(X_normalized, y))

''' Dump the result '''
# dump(reg, folder_path + 'mp_' + name + '_LR_model.joblib')
# dump(scaler, folder_path + 'mp_' + name + '_scaler.joblib')

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

### Create a scatter plot
# plt.scatter(y, y_pred-y, color='blue', label='Predicted vs. Actual')
# plt.scatter(y, y_pred, color='blue', label='Predicted vs. Actual')
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.title('Actual vs. Predicted Values')
# plt.legend()
# plt.savefig('scatter_plot2.png')  # Provide the desired file name and extension

''' Measure prediction time '''
df_row = df.iloc[3]
# Create a single data point for prediction
X_single = pd.DataFrame()

# Start prediction
start_time = time.time()

mp_H_output_row = df_row['maxpool_H']
mp_W_output_row = df_row['maxpool_W']

'''physical operations'''
X_single['FLOPs'] = [(df_row["maxpool_ksizeH"] * df_row["maxpool_ksizeW"]) * (mp_H_output_row * mp_W_output_row) * df_row["maxpool_C"]]
X_single['IN_MACs'] = [(df_row['maxpool_imgH'] * df_row['maxpool_imgW']) * df_row["maxpool_C"] * 8]
X_single['OUT_MACs'] = [(mp_H_output_row * mp_W_output_row) * df_row["maxpool_C"] * 8]

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