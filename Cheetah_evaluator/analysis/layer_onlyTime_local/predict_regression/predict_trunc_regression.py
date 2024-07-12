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
layer_filepath = target_folder + "trunc_onlyTime_" + name + ".csv"
df = pd.read_csv(layer_filepath, delimiter="\s+")
folder_path = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/predict_regression/LR_model/"
# folder_path = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/predict_regression/LR_model_without_sqnet/"

'''
trunc columns= "trunc_coeff", 'time_cost'
'''

# print(df.head(10))
# X = df.loc[:, ["maxpool_N", "maxpool_H", "maxpool_W", "maxpool_C", "maxpool_ksizeH", "maxpool_ksizeW"]]

### Process on dataset
print("*** Number of data before processing: ", df.size)
df = df[df['time_cost'] > 0]
print("*** Number of data after processing: ", df.size)

### Training and testing
X = pd.DataFrame()

'''physical operations'''
X['FLOPs'] = df["trunc_coeff"]

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
dump(reg, folder_path + 'trunc_' + name + '_LR_model.joblib')
dump(scaler, folder_path + 'trunc_' + name + '_scaler.joblib')

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

# Perform 10-fold cross-validation on the training set
cv_scores_mae = cross_val_score(reg, X_normalized, y, cv=10, scoring='neg_mean_absolute_error')
mae_cv = -cv_scores_mae.mean()

cv_scores = cross_val_score(reg, X_normalized, y, cv=10, scoring='neg_mean_absolute_percentage_error')
mape_cv = -cv_scores.mean() * 100

# Perform 10-fold cross-validation on the training set for RMSE
cv_scores_rmse = cross_val_score(reg, X_normalized, y, cv=10, scoring='neg_mean_squared_error')
rmse_cv = np.sqrt(-cv_scores_rmse.mean())

# Define a function to calculate RMSPE
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2)) * 100

# Perform 10-fold cross-validation on the training set for RMSPE
cv_scores_rmspe = cross_val_score(reg, X_normalized, y, cv=10, scoring=make_scorer(rmspe, greater_is_better=False))
rmspe_cv = -np.mean(cv_scores_rmspe)

cv_scores_r2 = cross_val_score(reg, X_normalized, y, cv=10, scoring='r2')
r2_cv = cv_scores_r2.mean()

print("*** 10-fold Cross-Validation Results ***")
print("10-fold cross-validation result (MAE): ", mae_cv)
print("10-fold cross-validation result (MAPE): ", mape_cv)
print("10-fold cross-validation result (RMSE): ", rmse_cv)
print("10-fold cross-validation result (RMSPE): ", rmspe_cv)
print("10-fold cross-validation result (R2): ", r2_cv)

### Create a scatter plot
# plt.scatter(y, y_pred-y, color='blue', label='Predicted vs. Actual')
# plt.scatter(y, y_pred, color='blue', label='Predicted vs. Actual')
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.title('Actual vs. Predicted Values')
# plt.legend()
# plt.savefig('scatter_plot2.png')  # Provide the desired file name and extension