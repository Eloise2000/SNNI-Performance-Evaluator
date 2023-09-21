# Tensorflow 2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from kerastuner import HyperModel, RandomSearch

class MyHyperModel(HyperModel):
    def __init__(self, input_shape, num_outputs):
        self.input_shape = input_shape
        self.num_outputs = num_outputs

    def build(self, hp):
        model = tf.keras.Sequential()
        
        # Tune the number of hidden layers and units
        for i in range(hp.Int('num_hidden_layers', min_value=1, max_value=5)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32), activation='relu'))
        
        model.add(layers.Dense(self.num_outputs))
        
        # Tune the optimizer and learning rate
        optimizer = hp.Choice('optimizer', ['adam', 'sgd'])
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

        return model

# Load and preprocess the data
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

input_shape = (13,)  # Input shape for your data
num_outputs = 1  # Number of output units

# Define the hyperparameter tuning search space
hypermodel = MyHyperModel(input_shape, num_outputs)

# Define the tuner
tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=10,  # Number of trials to perform
    directory='tuner_results',  # Directory to store results
    project_name='my_tuning_project'  # Name of the tuning project
)

tuner.reload()
tuner.get_best_models(num_models=2)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the best model
best_model = tuner.hypermodel.build(best_hps)

# Compile the best model
best_model.compile(optimizer=best_hps.get('optimizer'), loss='mean_squared_error', metrics=['mae'])

# Train the best model
history = best_model.fit(x_train, y_train, epochs=500, validation_data=(x_val, y_val))

# Evaluate the best model
# test_loss, test_mae = best_model.evaluate(x_test, y_test)
# print('Test loss:', test_loss)
# print('Test MAE:', test_mae)

y_pred = best_model.predict(x_test)

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
plt.title('Conv Training and Validation Loss for '+name)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('conv_best_plot.png')