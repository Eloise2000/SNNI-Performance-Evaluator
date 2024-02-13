import tensorflow as tf
import random

conv_weights = []
powers_of_2_all = [16, 32, 64, 128, 256, 512, 1024]
kernel_options_conv = [1, 3, 5, 7]
kernel_options_pool = [2,3]
padding_options = ['valid', 'same']  # List of possible padding options

def random_generate(powers_of_2):
    filters_num = random.choice(powers_of_2)
    kernel_size_conv = random.choice(kernel_options_conv)
    strides_conv = random.randint(1, 2)
    padding_option_conv = random.choice(padding_options)  # Choose random padding option
    kernel_size_pool = random.choice(kernel_options_pool)
    strides_pool = random.randint(1, 2)
    padding_option_pool = random.choice(padding_options)  # Choose random padding option
    return filters_num, kernel_size_conv, strides_conv, padding_option_conv, kernel_size_pool, strides_pool, padding_option_pool

# Define the TensorFlow model
input_tensor = tf.placeholder(tf.float32, shape=[1, 224, 224, 3], name='input')

conv_H = 224
conv_C = 3
for _ in range(5):  # Assuming 8 convolutional layers
    th = 810000//(conv_H*conv_H)
    print(th, conv_H)
    cur_powers_of_2 = [num for num in powers_of_2_all if num <= th]
    filters_num, kernel_size_conv, strides_conv, padding_option_conv, kernel_size_pool, strides_pool, padding_option_pool = random_generate(cur_powers_of_2)
    
    conv_layer = tf.layers.conv2d(
        input_tensor,
        filters=filters_num,
        kernel_size=[kernel_size_conv, kernel_size_conv],
        strides=[strides_conv, strides_conv],
        padding=padding_option_conv,  # Use the chosen padding option
        activation=tf.nn.relu,
        # activation=None,
        name=f'conv{_ + 1}'  # Naming each layer sequentially
    )

    pool_layer = tf.layers.max_pooling2d(
        conv_layer, 
        pool_size=[kernel_size_pool, kernel_size_pool], 
        strides=strides_pool, 
        padding=padding_option_pool,
        name=f'pool{_ + 1}'  # Naming each layer sequentially
    )
    
    conv_H, conv_C = conv_layer.shape[1], conv_layer.shape[3]

    # print(filters_num, kernel_size, strides_conv, padding_option)
    print(f"Layer name: {conv_layer.name}")
    print(f"Layer shape: {conv_layer.shape}")
    print(f"Layer name: {pool_layer.name}")
    print(f"Layer shape: {pool_layer.shape}")

    conv_weight = tf.Variable(
        tf.random.normal([kernel_size_conv, kernel_size_conv, conv_layer.shape[-1], filters_num]),
        name=f'w_conv{_ + 1}'  # Naming each weight tensor sequentially
    )
    conv_weights.append(conv_weight)
    input_tensor = pool_layer

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
    with tf.gfile.GFile('model_convmp5.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())