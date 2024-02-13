import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Activation, LSTM, TimeDistributed, Masking, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

# Set variable here
show_server = 1 # 0: client; 1: server

if show_server:
    name = "server"
else:
    name = "client"

# samples = np.load(save_folder + "samples.npy")
samples = np.load(save_folder + "samples_sparse.npy")
# time = np.load(save_folder + "time.npy")
time = np.load(save_folder + "time_seq.npy")

print("samples shape is: ", samples.shape)
print("time shape is: ", time.shape)
# Assuming X_train, y_train, X_test, y_test are your training and testing data
# X_train and X_test should have shape (samples, time_steps, features)

num_samples, max_time_steps, num_features = samples.shape

samples_2d = samples.reshape(-1, num_features)
# Apply Min-Max scaling
scaler = MinMaxScaler()
samples_2d_scaled = scaler.fit_transform(samples_2d)
# Reshape back to 3D
samples_scaled = samples_2d_scaled.reshape(num_samples, max_time_steps, num_features)

print("samples_scaled shape: ", samples_scaled.shape)

# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-5 * 10**(epoch / 5)  # Adjust this function as needed
# )

model = Sequential([
    Masking(mask_value=0.0, input_shape=(None, 51)),
    LSTM(units=16, return_sequences=True),
    TimeDistributed(Dense(128, activation='relu')),
    TimeDistributed(Dense(312, activation='relu')),
    TimeDistributed(Dense(312, activation='relu')),
    TimeDistributed(Dense(128, activation='relu')),
    TimeDistributed(Dense(1, activation='linear'))
])

custom_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # Set your desired learning rate
model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
# model.compile(optimizer='adam', loss='mean_squared_error')

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(samples_scaled, time, test_size=0.2, random_state=42)
# Further split the training set into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Train the model
# history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=32, callbacks=[lr_scheduler])
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=300, batch_size=32)

# Evaluate the model on the test set
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

print("x_test shape: ", x_test.shape)
print("y_test shape: ", y_test.shape)
print("y_pred shape: ", y_pred.shape)

for i in range(len(y_test)):
    print("Real data: ", y_test[i], "Predict data: ", y_pred[i], "Diff: ", y_pred[i] - y_test[i])

# Print the test loss
print(f"Test Loss: {loss}")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('RNN_best_plot_' + name + '.png')
