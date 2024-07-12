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
        num_hidden_layers = hp.Int("num_hidden_layers", min_value=2, max_value=3)

        model = tf.keras.Sequential()
        
        # Add the first dense layer with input shape
        model.add(layers.Dense(units=hp.Int('units_0', min_value=16, max_value=640, step=16), activation = 'relu', 
                               input_shape=self.input_shape))

        # Tune the number of hidden layers and units
        # for i in range(hp.Int('num_hidden_layers', min_value=0, max_value=4)):
        for i in range(num_hidden_layers):
            # model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32), activation='relu'))
            model.add(layers.Dense(units=hp.Int(f"units_{i+1}", min_value=16, max_value=640, step=16), activation = 'relu'))
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
else: 
    name = "client"

'''
maxpool: "maxpool_N", "maxpool_H", "maxpool_W", "maxpool_C", "maxpool_ksizeH", "maxpool_ksizeW"
'''
target_folder = "/home/eloise/eloise/script/analysis/layer_onlyTime_LAN/dataset/data_layer/"
layer_filepath = target_folder + "maxpool_onlyTime_"+name+".csv"
df = pd.read_csv(layer_filepath, delimiter="\s+")

# print(df.head(10))
x = df.loc[:, ["maxpool_N", "maxpool_H", "maxpool_W", "maxpool_C", "maxpool_ksizeH", "maxpool_ksizeW", 
                "maxpool_zPadHLeft", "maxpool_zPadHRight", "maxpool_zPadWLeft", "maxpool_zPadWRight",
                "maxpool_strideH", "maxpool_strideW", "maxpool_N1", 
                "maxpool_imgH", "maxpool_imgW", "maxpool_C1"]]
y = df['time_cost']

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Print the shapes of the training and test sets
print('Training data shape:', x_train.shape, y_train.shape)
print('Test data shape:', x_test.shape, y_test.shape)

# Define the input data shape and number of output units
input_shape = (16,) # example input shape for maxpool
num_outputs = 1 # example number of output units

# Define the hyperparameter tuning search space
hypermodel = MyHyperModel(input_shape, num_outputs)

# Define the tuner
tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=20,  # Number of trials to perform
    directory='tuner_results_' + name,  # Directory to store results
    project_name='230111-1'  # Name of the tuning project
)

# Search for the best hyperparameters
tuner.search(x_train, y_train, epochs=100, validation_data=(x_val, y_val), verbose=2)  # Set verbose to 2

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build and compile the final model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)
best_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the best model
history = best_model.fit(x_train, y_train, epochs=300, batch_size=32, validation_data=(x_val, y_val))

y_pred = best_model.predict(x_test)

# Search for the best hyperparameters
for trial, trial_info in enumerate(tuner.oracle.get_best_trials(num_trials=10)):
    # Extract hyperparameters for the current trial
    trial_hps = trial_info.hyperparameters

    print(f"\nHyperparameters for Trial {trial + 1}:")
    for param, value in trial_hps.values.items():
        print(f"{param}: {value}")

# Build the model by calling build() method on a batch of data
best_model.build(x_train.shape)  # Use the shape of the input data
# Print the best model architecture
print("\n**********Best Model Architecture:**********")
best_model.summary()

# Print the best hyperparameters
print("\n**********Best Hyperparameters:**********")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")

# Calculate RMSE, MAE, and R2 using the predicted and actual target values
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate MAPE
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
plt.title('Maxpool Training and Validation Loss for '+name)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('maxpool_best_plot_' + name + '.png')

best_model.save('best_model_mp_' + name + '.keras')