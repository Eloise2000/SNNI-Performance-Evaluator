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

'''
avgpool: "avgpool_N", "avgpool_H", "avgpool_W", "avgpool_C", "avgpool_ksizeH", "avgpool_ksizeW"
'''
target_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_origin/"
# target_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_without_sqnet/"
layer_filepath = target_folder + "avgpool_onlyTime_"+name+".csv"
df = pd.read_csv(layer_filepath, delimiter="\s+")

# print(df.head(10))
x = df.loc[:, ["avgpool_N", "avgpool_H", "avgpool_W", "avgpool_C", "avgpool_ksizeH", "avgpool_ksizeW", 
                "avgpool_zPadHLeft", "avgpool_zPadHRight", "avgpool_zPadWLeft", "avgpool_zPadWRight",
                "avgpool_strideH", "avgpool_strideW", "avgpool_N1", 
                "avgpool_imgH", "avgpool_imgW", "avgpool_C1"]]
y = df['time_cost']

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Print the shapes of the training and test sets
print('Training data shape:', x_train.shape, y_train.shape)
print('Test data shape:', x_test.shape, y_test.shape)

# Define the input data shape and number of output units
input_shape = (16,) # example input shape for avgpool
num_outputs = 1 # example number of output units

# Define the hyperparameter tuning search space
hypermodel = MyHyperModel(input_shape, num_outputs)

# Define the tuner
tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=20,  # Number of trials to perform
    directory='tuner_results_' + name,  # Directory to store results
    project_name='230925'  # Name of the tuning project
)

# Search for the best hyperparameters
tuner.search(x_train, y_train, epochs=100, validation_data=(x_val, y_val), verbose=2)  # Set verbose to 2

# Print the summary of each trial with R2 scores
tuner.results_summary()

# print("R2 Scores for each trial:")
# for i, r2 in enumerate(trial_r2_scores):
#     print(f"Trial {i + 1}: R2 = {r2:.4f}")
    
# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build and compile the final model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)
# best_model.compile(optimizer=best_hps.get('optimizer'), loss='mean_squared_error', metrics=['mae'])
best_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the best model
history = best_model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val))

# Evaluate the best model
# test_loss, test_mae = best_model.evaluate(x_test, y_test)
# print('Test loss:', test_loss)
# print('Test MAE:', test_mae)

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
plt.savefig('avgpool_best_plot_' + name + '.png')

best_model.save('best_model_ap_' + name + '.keras')