"""
Author: Eloise Zhang
Date: 12 July 2024
Description: This script generates several neural network models with Conv2d, Relu and MaxPool layers.
"""

import crypten
import crypten.nn as cnn
import random
import torch
import time
import argparse

class DynamicModel(cnn.Module):
    def __init__(self, random_seed=None):
        super(DynamicModel, self).__init__()

        # Random generate options
        self.powers_of_2 = [16, 32, 64, 128, 256, 512, 1024]
        self.empirical_map = {100:1024, 128:512, 200:256, 256:128} # HI:CO_max
        self.kernel_options_conv = [1, 3, 5, 7]
        self.kernel_options_pool = [2,3]
        self.stride_options = [1,2]
        self.padding_options = ['valid', 'same']

        # Save the random seed for reproducibility
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)

        # Initialize an empty list to store layers
        self.layers = cnn.ModuleList()

        # Randomly determine the number of layers (conv + relu + maxpool) within a range
        num_layers = random.randint(5, 10)
        in_channels = 3
        in_size = 256

        for _ in range(num_layers):
            # Randomly generate the parameters
            conv_out_channels, conv_kernel_size, conv_strides, conv_padding, pool_kernel_size, pool_strides = self.random_generate(in_size)
            # For Designing layers
            if conv_padding == "valid": in_size = (in_size - conv_kernel_size) // conv_strides + 1
            print(f"Current size of Conv is ({in_size},{in_size}), in channel is {in_channels}, out channel is {conv_out_channels}, kernel size is {conv_kernel_size}, stride is {conv_strides}, padding is {conv_padding}")

            if in_size <= 0:
                break

            # Add convolutional layer
            self.layers.append(cnn.Conv2d(in_channels, conv_out_channels, kernel_size=conv_kernel_size, stride=conv_strides, padding=conv_padding))
            in_channels = conv_out_channels  # Update in_channels for the next layer

            # Randomly decide whether to add ReLU layer
            if random.random() < 0.8 and in_size > 0:
                self.layers.append(cnn.ReLU())

            # Only add MaxPool layer when flop is not too large
            if self.check_maxpool(in_size, in_channels, pool_kernel_size, pool_strides):
                # For Designing layers
                in_size = (in_size - pool_kernel_size) // pool_strides + 1
                print(f"Current size of MaxPool is {in_size}, channel is {in_channels}")

                if in_size <= 0:
                    break

                self.layers.append(cnn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_strides))

                
        # self.layers.append(cnn.Linear(16 * 128 * 128, 1))

    def check_maxpool(self, in_size, in_channels, pool_kernel_size, pool_strides):
        out_size = (in_size - pool_kernel_size) // pool_strides + 1
        return out_size * out_size * pool_kernel_size * pool_kernel_size * in_channels < 48880000

    def random_generate(self, in_size):
        # Get the maximum possible CO
        CO_max = 64
        for size, CO in self.empirical_map.items():
            if in_size <= size:
                CO_max = CO
                break

        # Randomly select conv parameters
        if random.random() < 0.2:
            powers_of_2 = [num for num in self.powers_of_2 if num <= CO_max]
            conv_out_channels = random.choice(powers_of_2)
            # print("Test1", powers_of_2, conv_out_channels)
        else:
            conv_out_channels = random.randint(16, CO_max)
            # print("Test2", conv_out_channels)
        conv_kernel_size = random.choice(self.kernel_options_conv)
        conv_strides = random.choice(self.stride_options)
        if conv_strides != 1: conv_padding = "valid" # padding='same' is not supported for strided convolutions
        else: conv_padding = random.choice(self.padding_options)

        # Randomly select maxpool parameters
        pool_kernel_size = random.choice(self.kernel_options_pool)
        pool_strides = random.choice(self.stride_options)
        return conv_out_channels, conv_kernel_size, conv_strides, conv_padding, pool_kernel_size, pool_strides

    def forward(self, x):
        conv_cnt = 0
        relu_cnt = 0
        mp_cnt = 0
        for i, layer in enumerate(self.layers):
            # Conv2d layer
            if isinstance(layer, cnn.Conv2d):
                conv_cnt += 1
                # For experiment
                N, CI, HI, WI = x.size()
                # CO = layer.weight.size(0)
                kernel_size = tuple(layer.weight.size()[2:])
                stride = layer.stride
                padding = layer.padding

                start_time = time.time()
                x = layer(x) # forward
                end_time = time.time()

                N, CO, HO, WO = x.size()
                print(f"Layer{i}: Conv{conv_cnt} - Features: Batch_Size={N}, In_Height={HI}, In_Width={WI}, In_Channels={CI}, Out_Height={HO}, Out_Width={WO}, Out_Channels={CO}, Kernel_Size={kernel_size}, Stride={stride}, Padding={padding} - Output_Shape: {x.size()} - Start_time: {start_time} - End_time: {end_time} - Time_used: {(end_time-start_time)*1000} ms")

            # Relu layer
            if isinstance(layer, cnn.ReLU):
                relu_cnt += 1
                N, C, H, W = x.size()

                start_time = time.time()
                x = layer(x) # forward
                end_time = time.time()

                print(f"Layer{i}: ReLU{relu_cnt} - Features: Batch_Size={N}, Height={H}, Width={W}, Channels={C} - Output_Shape: {x.size()} - Start_time: {start_time} - End_time: {end_time} - Time_used: {(end_time-start_time)*1000} ms")

            # MaxPool layer
            if isinstance(layer, cnn.MaxPool2d):
                mp_cnt += 1
                # For experiment
                N, CI, HI, WI = x.size()
                kernel_size = layer.kernel_size
                stride = layer.stride
                padding = layer.padding
                ceil_mode = layer.ceil_mode

                start_time = time.time()
                x = layer(x) # forward
                end_time = time.time()

                N, CO, HO, WO = x.size()
                print(f"Layer{i}: MaxPool{mp_cnt} - Features: Batch_Size={N}, In_Height={HI}, In_Width={WI}, In_Channels={CI}, Out_Height={HO}, Out_Width={WO}, Out_Channels={CO}, Kernel_Size={kernel_size}, Stride={stride}, Padding={padding}, ceil_mode={ceil_mode} - Output_Shape: {x.size()} - Start_time: {start_time} - End_time: {end_time} - Time_used: {(end_time-start_time)*1000} ms")

        return x

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Generates several neural network models with Conv2d, Relu and MaxPool layers.')

    # Add the argument
    parser.add_argument('--random_seed', type=int, required=True, help='an integer for setting the model random seed')
    parser.add_argument('--folderpath', type=str, required=True, help="Folder path to save the model")
    parser.add_argument("--test", type=bool, required=True, help="Flag indicating test mode.")

    # Parse the arguments
    try:
        args = parser.parse_args()

        if (args.random_seed and args.folderpath) is not None:
            random_seed = args.random_seed
            save_path = args.folderpath
            Test_Flag = args.test
            print(f"Random seed set to {random_seed}")
            print(f"Folder path set to {save_path}")
            print(f"Test set to {Test_Flag}")

    except argparse.ArgumentError as e:
        print(f"Error: {e}")
        raise

    # Prepare crypten
    print("Start Crypten...")
    crypten.init()

    # Create an instance of the dynamic model
    dynamic_model = DynamicModel(random_seed=random_seed)
    
    # Save the non-encrypted model with random initiated parameters
    crypten.save_from_party({
        'model_config': {
            'random_seed': dynamic_model.random_seed,
        },
        'model_state_dict': dynamic_model.state_dict(),
    }, f"{save_path}convmp_model_rs{random_seed}.pth", src=0)
    print(f"Non-encrypted model saved in {save_path}convmp_model_rs{random_seed}.pth with random seed {random_seed}.")

    if Test_Flag:
        print("Start testing...")
        private_model = dynamic_model.encrypt()

        print("\nNetwork encrypted:")
        for _, curr_module in private_model._modules.items():
            for idx, module in curr_module._modules.items():
                print(idx, module)

        # Test the model with a dummy input
        dummy_input = crypten.randn(1, 3, 256, 256)  # Batch size of 1, 3 channels, 256x256 image

        print("\nTest on prediction:")
        startpre_time = time.time()
        result_enc = private_model.forward(dummy_input)
        endpre_time = time.time()
        print(f"Time for prediction cost: {(endpre_time - startpre_time)*1000} ms")
        print(f"Output shape: {result_enc.get_plain_text().shape}")