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
        num_hidden_layers = hp.Int("num_hidden_layers", min_value=1, max_value=3)
        # num_hidden_layers = hp.Int("num_hidden_layers", min_value=1, max_value=4)

        model = tf.keras.Sequential()
        
        # Add the first dense layer with input shape
        model.add(layers.Dense(units=hp.Int('units_0', min_value=16, max_value=512, step=16), activation = 'relu', 
                               input_shape=self.input_shape))

        # Tune the number of hidden layers and units
        # for i in range(hp.Int('num_hidden_layers', min_value=0, max_value=4)):
        for i in range(num_hidden_layers):
            # model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32), activation='relu'))
            model.add(layers.Dense(units=hp.Int(f"units_{i+1}", min_value=16, max_value=512, step=16), activation = 'relu'))
            # model.add(layers.BatchNormalization())

        model.add(layers.Dense(self.num_outputs))
        
        # Tune the optimizer and learning rate
        # optimizer = hp.Choice('optimizer', ['adam', 'sgd'])
        optimizer = 'adam'
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

        model.summary()
        return model

show_server = 1
if show_server:
    name = "server"
    project_used ='230111-1'
else: 
    name = "client"
    project_used ='230111-1'

'''
relu columns= 'relu_coeff','time_cost'
'''
target_folder = "/home/eloise/eloise/script/analysis/layer_onlyTime_LAN/dataset/data_layer/"
layer_filepath = target_folder + "relu_onlyTime_"+name+".csv"
df = pd.read_csv(layer_filepath, delimiter="\s+")

df = df[df['time_cost'] > 0]
x = df.loc[:, ["relu_coeff"]]
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

NN_folderPath = "/home/eloise/eloise/script/analysis/layer_onlyTime_LAN/predict_NN/"
best_model_relu = tf.keras.models.load_model(f"{NN_folderPath}relu/tuner_results_{name}/{project_used}/best_model_relu_{name}.keras")

df_row = df.iloc[300]
x_single = df_row[["relu_coeff"]].values.reshape(1, -1)
y_single = df_row['time_cost']

# Start prediction
start_time = time.time()
y_single_pred = best_model_relu.predict(x_single)
end_time = time.time()

# Calculate the time taken for prediction
prediction_time = end_time - start_time

print("Predicted Time:", y_single_pred)
print("Real Time:", y_single)
print("Diff Time:", y_single - y_single_pred)
print("Prediction Time:", prediction_time * 1000, "ms")