# mp1, mp3, mp4
import tensorflow as tf
import random

kernel_options = [2,3]
padding_options = ['valid', 'same']  # List of possible padding options

def random_generate():
    kernel_size = random.choice(kernel_options)
    strides_pool = random.randint(1, 2)
    padding_option = random.choice(padding_options)  # Choose random padding option
    return kernel_size, strides_pool, padding_option

# Define the TensorFlow model
input_tensor = tf.placeholder(tf.float32, shape=[1, 224, 224, 3], name='input')

for _ in range(10):  # Assuming 8 convolutional layers
    kernel_size, strides_pool, padding_option = random_generate()
    
    pool_layer = tf.layers.average_pooling2d(
        input_tensor, 
        pool_size=[kernel_size, kernel_size], 
        strides=strides_pool, 
        padding=padding_option,
        name=f'pool{_ + 1}'  # Naming each layer sequentially
    )

    print(kernel_size, strides_pool, padding_option)
    print(f"Layer name: {pool_layer.name}")
    print(f"Layer shape: {pool_layer.shape}")

    # Random add relu layer after
    if random.randint(0, 1):
        relu_layer = tf.nn.relu(pool_layer)
        input_tensor = relu_layer
    else: 
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

    # Convert the TensorFlow model to a .pb file
    output_node_names = ['logits/BiasAdd']
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)
    with tf.gfile.GFile('model_ap1.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())