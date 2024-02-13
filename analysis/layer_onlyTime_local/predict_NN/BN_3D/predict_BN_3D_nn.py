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
        for i in range(hp.Int('num_hidden_layers', min_value=1, max_value=4)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=256, step=32), activation='relu'))
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
else: 
    name = "client"

'''"BN1_C", "BN1_H", "BN1_W", "time_cost"'''
'''"BN2_C", "BN2_H", "BN2_W", "time_cost"'''

target_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_origin/"
# target_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_without_sqnet/"
layer_filepath_BN1 = target_folder + "BN1_onlyTime_" + name + ".csv"
layer_filepath_BN2 = target_folder + "BN2_onlyTime_3D_" + name + ".csv"
df_BN1 = pd.read_csv(layer_filepath_BN1, delimiter="\s+")
df_BN1 = df_BN1[df_BN1['time_cost'] > 0]
df_BN2 = pd.read_csv(layer_filepath_BN2, delimiter="\s+")
df_BN2 = df_BN2[df_BN2['time_cost'] > 0]

df_BN1 = df_BN1.rename(columns={'BN1_C': 'C', 'BN1_H': 'H', 'BN1_W': 'W'})
df_BN2 = df_BN2.rename(columns={'BN2_C': 'C', 'BN2_H': 'H', 'BN2_W': 'W'})
df = pd.concat([df_BN1, df_BN2], axis = 0, ignore_index=True)

# print(df.head(10))
x = df.loc[:, ['C', 'H', 'W']]
y = df['time_cost']
# print(df)
# print(x)
# print(y)
# print(x.shape)
# print(y.shape)
# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Print the shapes of the training and test sets
print('Training data shape:', x_train.shape, y_train.shape)
print('Test data shape:', x_test.shape, y_test.shape)

# Define the input data shape and number of output units
input_shape = (1,) # example input shape for avgpool
num_outputs = 1 # example number of output units

# Define the hyperparameter tuning search space
hypermodel = MyHyperModel(input_shape, num_outputs)

# Define the tuner
tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=10,  # Number of trials to perform
    directory='tuner_results_' + name,  # Directory to store results
    project_name='231013'  # Name of the tuning project
)

# Search for the best hyperparameters
tuner.search(x_train, y_train, epochs=100, validation_data=(x_val, y_val), verbose=2)  # Set verbose to 2

# Print the summary of each trial with R2 scores
tuner.results_summary()
    
# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build and compile the final model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)
# best_model.compile(optimizer=best_hps.get('optimizer'), loss='mean_squared_error', metrics=['mae'])
best_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the best model
history = best_model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val))

y_pred = best_model.predict(x_test)

# Print the best hyperparameters
print("Best Hyperparameters:")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")

# Calculate RMSE, MAE, and R2 using the predicted and actual target values
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate MAPE
# print(y_pred.shape)
# print(y_test.shape)
y_pred = y_pred.reshape(-1)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Calculate RMSPE
rmspe = np.sqrt(np.mean(np.square(((y_test - y_pred) / y_test)))) * 100

print('MAE:', mae)
print('MAPE:', mape)
print('RMSE:', rmse)
print('RMSPE:', rmspe)
print('R2:', r2)

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Avgpool Training and Validation Loss for '+name)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('BN_best_plot_' + name + '.png')

best_model.save('best_model_BN_' + name + '.keras')