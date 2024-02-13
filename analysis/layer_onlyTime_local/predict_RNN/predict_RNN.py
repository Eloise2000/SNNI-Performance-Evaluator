import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Activation, LSTM, TimeDistributed, Masking, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from kerastuner import HyperModel, RandomSearch
from joblib import dump

# Set variable here
show_server = 1 # 0: client; 1: server

if show_server:
    name = "server"
    project_used = '231012'
else:
    name = "client"
    project_used = '231011'

save_folder = "/home/eloise/eloise/script/analysis/per_layer_onlyTime/dataset/data_RNN/" + name + "/"

class MyHyperModel(HyperModel):
    def __init__(self, input_shape, num_outputs):
        self.input_shape = input_shape
        self.num_outputs = num_outputs

    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(Masking(mask_value=0.0, input_shape=(None, 51)))
        model.add(LSTM(units=16, return_sequences=True))
        
        # Tune the number of hidden layers and units
        for i in range(hp.Int('num_hidden_layers', min_value=8, max_value=20)):
            model.add(TimeDistributed(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32), activation='relu')))
        
        model.add(TimeDistributed(Dense(self.num_outputs, activation='linear')))
        optimizer = 'adam'

        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

        return model

samples = np.load(save_folder + "samples_sparse.npy")
time = np.load(save_folder + "time_seq.npy")

print("samples shape is: ", samples.shape)
print("time shape is: ", time.shape)

# X_train and X_test should have shape (samples, time_steps, features)
num_samples, max_time_steps, num_features = samples.shape

# Apply Min-Max scaling
scaler = MinMaxScaler()
samples_scaled = scaler.fit_transform(samples.reshape(-1, num_features)).reshape(samples.shape)

# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-5 * 10**(epoch / 5)  # Adjust this function as needed
# )

# Define the input data shape and number of output units
input_shape = (num_features,) # example input shape for avgpool
num_outputs = 1 # example number of output units

# Define the hyperparameter tuning search space
hypermodel = MyHyperModel(input_shape, num_outputs)

# Define the tuner
tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=15,  # Number of trials to perform
    directory='tuner_results_' + name,  # Directory to store results
    project_name=project_used  # Name of the tuning project
)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(samples_scaled, time, test_size=0.2, random_state=42)
# Further split the training set into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Search for the best hyperparameters
tuner.search(x_train, y_train, epochs=50, validation_data=(x_val, y_val), batch_size=32, verbose=2)  # Set verbose to 2

# Print the summary of each trial with R2 scores
tuner.results_summary()

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build and compile the final model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)
# best_model.compile(optimizer=best_hps.get('optimizer'), loss='mean_squared_error', metrics=['mae'])
best_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the best model
history = best_model.fit(x_train, y_train, epochs=300, validation_data=(x_val, y_val), batch_size=32)

# Print the best hyperparameters
print("Best Hyperparameters:")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")

# Evaluate the model on the test set
loss = best_model.evaluate(x_test, y_test)
y_pred = best_model.predict(x_test)

for i in range(len(y_test)):
    print("Real data: ", y_test[i], "Predict data: ", y_pred[i], "Diff: ", y_pred[i] - y_test[i])

np.save("y_test.npy", y_test)
np.save("y_pred.npy", y_pred)

# Print the test loss
print(f"Test Loss: {loss}")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('RNN_best_plot_' + name + '.png')

dump(scaler, 'RNN_' + name + '_scaler.joblib')
best_model.save('best_model_RNN_' + name + '.keras')