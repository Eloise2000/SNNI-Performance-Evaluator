# Tensorflow 2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from kerastuner import HyperModel, RandomSearch
import tensorflow as tf
from tensorflow.keras import layers
import time

class MyHyperModel(HyperModel):
    def __init__(self, input_shape, num_outputs):
        self.input_shape = input_shape
        self.num_outputs = num_outputs

    def build(self, hp):
        model = tf.keras.Sequential()
        
        # Tune the number of hidden layers and units
        for i in range(hp.Int('num_hidden_layers', min_value=1, max_value=2)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=128, step=32), activation='relu'))
            # model.add(layers.BatchNormalization())

        model.add(layers.Dense(self.num_outputs))
        
        # Tune the optimizer and learning rate
        # optimizer = hp.Choice('optimizer', ['adam', 'sgd'])
        optimizer = 'adam'
        # learning_rate = hp.Float('learning_rate', min_value=1e-3, max_value=1e-2)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

        return model

show_server = 1
if show_server:
    name = "server"
    project_used ='230925-2-used'
else: 
    name = "client"
    project_used ='230925'

'''"BN1_C", "BN1_H", "BN1_W", "time_cost"'''
'''"BN2_coeff", "time_cost"'''

target_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_origin/"
# target_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_without_sqnet/"
layer_filepath_BN1 = target_folder + "BN1_onlyTime_" + name + ".csv"
layer_filepath_BN2 = target_folder + "BN2_onlyTime_" + name + ".csv"
df_BN1 = pd.read_csv(layer_filepath_BN1, delimiter="\s+")
df_BN1 = df_BN1[df_BN1['time_cost'] > 0]
df_BN2 = pd.read_csv(layer_filepath_BN2, delimiter="\s+")
df_BN2 = df_BN2[df_BN2['time_cost'] > 0]

df = pd.DataFrame()
df['BN_coeff'] = df_BN1["BN1_C"] * df_BN1["BN1_H"] * df_BN1["BN1_W"]
df['BN_coeff'] += df_BN2['BN2_coeff']
df['time_cost'] = df_BN1['time_cost']
df['time_cost'] += df_BN2['time_cost']

# print(df.head(10))
x = df.loc[:, ["BN_coeff"]]
y = df['time_cost']

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Define the input data shape and number of output units
input_shape = (1,) # example input shape for maxpool
num_outputs = 1 # example number of output units

# Define the hyperparameter tuning search space
hypermodel = MyHyperModel(input_shape, num_outputs)

# Define the tuner
tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=10,  # Number of trials to perform
    directory='tuner_results_' + name,  # Directory to store results
    project_name=project_used  # Name of the tuning project
)

tuner.reload()

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build and compile the final model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)
best_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the best model
history = best_model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val))

df_row = df_BN2.iloc[300]
x_single = df_row[['BN2_coeff']].values.reshape(1, -1)
y_single = df_row['time_cost']

# Start prediction
start_time = time.time()
y_single_pred = best_model.predict(x_single)
end_time = time.time()

# Calculate the time taken for prediction
prediction_time = end_time - start_time

print("Predicted Time:", y_single_pred)
print("Real Time:", y_single)
print("Diff Time:", y_single - y_single_pred)
print("Prediction Time:", prediction_time * 1000, "ms")