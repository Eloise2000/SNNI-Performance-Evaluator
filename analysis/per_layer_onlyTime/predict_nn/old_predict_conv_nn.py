import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

show_server = 0
if show_server:
    name = "server"
else: 
    name = "client"

'''
conv: conv_N, conv_H, conv_W, conv_CI, conv_FH, conv_FW, conv_CO, conv_S, conv_Padding, time_cost, CPU_avg, RAM_max, power_avg, energy
N, meta.ishape.height(), meta.ishape.width(),
meta.ishape.channels(), meta.fshape.height(), meta.fshape.width(),
meta.n_filters, meta.stride,
(meta.padding == gemini::Padding::VALID ? "VALID" : "SAME"), zPadHLeft,
zPadHRight, zPadWLeft, zPadWRight)
'''
layer_filepath = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/conv_onlyTime_"+name+".csv"
df = pd.read_csv(layer_filepath, delimiter="\s+")

# print(df.head(10))
x = df.loc[:, ["conv_N", "conv_H", "conv_W", "conv_CI", "conv_FH", "conv_FW", "conv_CO", 
    "conv_zPadHLeft", "conv_zPadHRight", "conv_zPadWLeft", "conv_zPadWRight", 
    "conv_strideH", "conv_strideW"]]
y = df['time_cost']

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Print the shapes of the training and test sets
print('Training data shape:', x_train.shape, y_train.shape)
print('Test data shape:', x_test.shape, y_test.shape)

# Define the input data shape and number of output units
input_shape = (13,) # example input shape for conv
num_outputs = 1 # example number of output units

# Define the MLP architecture
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_outputs)
])

# model = tf.keras.Sequential([
#     layers.Dense(10, activation='relu', input_shape=input_shape),
#     layers.BatchNormalization(),
#     layers.Dropout(0.3),
#     layers.Dense(10, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.3),
#     layers.Dense(num_outputs)
# ])

# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error')

# Train the model
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

# # Evaluate the model on the test data
# test_loss = model.evaluate(x_test, y_test)
# print('Test loss:', test_loss)

y_pred = model.predict(x_test)

# Calculate RMSE, MAE, and R2 using the predicted and actual target values
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('RMSE:', rmse)
print('MAE:', mae)
print('R2:', r2)

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss for '+name)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()