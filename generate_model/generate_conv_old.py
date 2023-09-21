import tensorflow as tf
import random

conv_weights = []
powers_of_2_large = [16, 32, 64, 128, 256, 512, 1024]
powers_of_2_mid = [16, 32, 64]
powers_of_2_small = [16, 32]
kernel_options = [1, 3, 5, 7]
padding_options = ['valid', 'same']  # List of possible padding options

def random_generate_large():
    filters_num = random.choice(powers_of_2_large)
    kernel_size = random.choice(kernel_options)
    strides_conv = random.randint(1, 2)
    padding_option = random.choice(padding_options)  # Choose random padding option
    return filters_num, kernel_size, strides_conv, padding_option

def random_generate_mid():
    filters_num = random.choice(powers_of_2_mid)
    kernel_size = random.choice(kernel_options)
    strides_conv = random.randint(1, 2)
    padding_option = random.choice(padding_options)  # Choose random padding option
    return filters_num, kernel_size, strides_conv, padding_option

def random_generate_small():
    filters_num = random.choice(powers_of_2_small)
    kernel_size = random.choice(kernel_options)
    strides_conv = random.randint(1, 2)
    padding_option = random.choice(padding_options)  # Choose random padding option
    return filters_num, kernel_size, strides_conv, padding_option

# Define the TensorFlow model
input_tensor = tf.placeholder(tf.float32, shape=[1, 224, 224, 3], name='input')

conv_H = 224
conv_C = 3
for _ in range(10):  # Assuming 8 convolutional layers
    if conv_H > 200: # Make sure not too much computation
        filters_num, kernel_size, strides_conv, padding_option = random_generate_small()
    elif conv_H > 100: # Make sure not too much computation
        filters_num, kernel_size, strides_conv, padding_option = random_generate_mid()
    else:
        filters_num, kernel_size, strides_conv, padding_option = random_generate_large()
    
    conv_layer = tf.layers.conv2d(
        input_tensor,
        filters=filters_num,
        kernel_size=[kernel_size, kernel_size],
        strides=[strides_conv, strides_conv],
        padding=padding_option,  # Use the chosen padding option
        # activation=tf.nn.relu,
        activation=None,
        name=f'conv{_ + 1}'  # Naming each layer sequentially
    )
    
    conv_H, conv_C = conv_layer.shape[1], conv_layer.shape[3]

    print(filters_num, kernel_size, strides_conv, padding_option)
    print(f"Layer name: {conv_layer.name}")
    print(f"Layer shape: {conv_layer.shape}")

    conv_weight = tf.Variable(
        tf.random.normal([kernel_size, kernel_size, conv_layer.shape[-1], filters_num]),
        name=f'w_conv{_ + 1}'  # Naming each weight tensor sequentially
    )
    conv_weights.append(conv_weight)

    if (conv_H*conv_H)*conv_C < 810000:
        relu_layer = tf.nn.relu(conv_layer)
        input_tensor = relu_layer
        print("cur: ", (conv_H*conv_H)*conv_C)
        print(f"Layer name: {relu_layer.name}")
    else:
        input_tensor = conv_layer

flatten = tf.layers.flatten(input_tensor, name='flatten')
fc1 = tf.layers.dense(flatten, units=1024, activation=tf.nn.relu, name='fc1')
logits = tf.layers.dense(fc1, units=10, name='logits')

# Initialize the TensorFlow session and global variables
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Get the shape of each layer
    # for layer in tf.compat.v1.get_default_graph().get_operations():
    #     print(layer.name, layer.outputs)
        
    # Get the value of the weight tensors
    conv_weights_value = sess.run(conv_weights)

    # Convert the TensorFlow model to a .pb file
    output_node_names = ['logits/BiasAdd']
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)
    with tf.gfile.GFile('model_conv3.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())