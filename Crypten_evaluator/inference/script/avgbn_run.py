"""
Author: Eloise Zhang
Date: 12 July 2024
Description: This script runs SNNI in 2PC, and tests the inference runtime for Conv2d, Relu, AvgPool and BatchNorm2d layers.
"""

import crypten
import crypten.nn as cnn
import torch
import crypten.communicator as comm
import random
import time
import copy
import sys

class DynamicModel(cnn.Module):
    def __init__(self, random_seed=None):
        super(DynamicModel, self).__init__()

        # Random generate options
        self.powers_of_2 = [16, 32, 64, 128, 256, 512, 1024]
        self.empirical_map = {100:1024, 128:512, 200:256, 256:128} # HI:CO_max
        self.kernel_options_conv = [1, 3, 5, 7]
        self.kernel_options_pool = [2,3]
        self.stride_options = [1,2,3]
        self.padding_options_conv = ['valid', 'same']

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

            if in_size <= 0:
                break

            # Add convolutional layer
            self.layers.append(cnn.Conv2d(in_channels, conv_out_channels, kernel_size=conv_kernel_size, stride=conv_strides, padding=conv_padding))
            print(f"Current size of Conv is ({in_size},{in_size}), in channel is {in_channels}, out channel is {conv_out_channels}, kernel size is {conv_kernel_size}, stride is {conv_strides}, padding is {conv_padding}")
            
            in_channels = conv_out_channels  # Update in_channels for the next layer

            # Add BatchNorm2d layer
            self.layers.append(cnn.BatchNorm2d(in_channels))
            print(f"Current channel of BatchNorm2d is {in_channels}")

            # Add ReLU layer
            self.layers.append(cnn.ReLU())

            # Add AvgPool layer
            # 20% kernel size is image size
            if random.random() < 0.2:
                self.layers.append(cnn.AvgPool2d(kernel_size=in_size))
                in_size = 1
            else:
                in_size = (in_size - pool_kernel_size) // pool_strides + 1
                if in_size <= 0:
                    break
                self.layers.append(cnn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_strides))
            print(f"Current size of AvgPool is {in_size}, channel is {in_channels}")
    
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
        else:
            conv_out_channels = random.randint(16, CO_max)

        conv_kernel_size = random.choice(self.kernel_options_conv)
        conv_strides = random.choice(self.stride_options)
        if conv_strides != 1: conv_padding = "valid" # padding='same' is not supported for strided convolutions
        else: conv_padding = random.choice(self.padding_options_conv)

        # Randomly select avgpool parameters
        pool_kernel_size = random.choice(self.kernel_options_pool)
        pool_strides = random.choice(self.stride_options)
        return conv_out_channels, conv_kernel_size, conv_strides, conv_padding, pool_kernel_size, pool_strides

    def forward(self, x):
        conv_cnt = 0
        bn_cnt = 0
        relu_cnt = 0
        ap_cnt = 0
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

            # AvgPool layer
            if isinstance(layer, cnn.AvgPool2d):
                ap_cnt += 1
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
                print(f"Layer{i}: AvgPool{ap_cnt} - Features: Batch_Size={N}, In_Height={HI}, In_Width={WI}, In_Channels={CI}, Out_Height={HO}, Out_Width={WO}, Out_Channels={CO}, Kernel_Size={kernel_size}, Stride={stride}, Padding={padding}, ceil_mode={ceil_mode} - Output_Shape: {x.size()} - Start_time: {start_time} - End_time: {end_time} - Time_used: {(end_time-start_time)*1000} ms")

            # BatchNorm2d layer
            if isinstance(layer, cnn.BatchNorm2d):
                bn_cnt += 1
                # For experiment
                N, C, H, W = x.size()
                eps = layer.eps
                momentum = layer.momentum

                start_time = time.time()
                x = layer(x) # forward
                end_time = time.time()

                print(f"Layer{i}: BatchNorm2d{bn_cnt} - Features: Batch_Size={N}, Height={H}, Width={W}, Channels={C}, eps={eps}, momentum={momentum} - Output_Shape: {x.size()} - Start_time: {start_time} - End_time: {end_time} - Time_used: {(end_time-start_time)*1000} ms")

        return x

if __name__ == "__main__":
    SERVER = 0
    CLIENT = 1

    # Check if the random seed is provided
    if len(sys.argv) != 2:
        print("Please add the random seed value for convmp.")
        sys.exit(1)

    random_seed = int(sys.argv[1])

    # Prepare crypten
    print("Start Crypten...")
    crypten.init()
    
    num_thread = torch.get_num_threads()
    print(f"Init finished, num_thread is {num_thread}")

    rank = comm.get().get_rank()
    print(f"Rank is {rank}")

    # Load the model architecture and state dictionary
    # `crypten.load_from_party` has problem, create the logic by my own, src = 0
    model_src = SERVER # SERVER has model
    if rank == SERVER:
        model_folder_path = "./generate_model/models"
        checkpoint = torch.load(f"{model_folder_path}avgbn_model_rs{random_seed}.pth")
        random_seed = checkpoint['model_config']['random_seed']
        private_model = DynamicModel(random_seed=random_seed)

        # Zero out the tensors / modules to hide loaded data from broadcast (isinstance(private_model, cnn.Module) = True)
        # private_model_zeros = copy.deepcopy(private_model)
        # for p in private_model_zeros.parameters():
        #     p.data.fill_(0)
        # comm.get().broadcast_obj(private_model_zeros, model_src)
        private_model.src = model_src
        private_model.load_state_dict(checkpoint['model_state_dict'])

    if rank ==  CLIENT:
        crypten.common.serial.register_safe_class(DynamicModel)
        # private_model = comm.get().broadcast_obj(None, model_src)
        private_model = DynamicModel(random_seed=random_seed)

    private_model.encrypt(src = model_src)
    print("Network encrypted:")
    for _, curr_module in private_model._modules.items():
        for idx, module in curr_module._modules.items():
            print(idx, module)

    # Test the model with a dummy input
    dummy_input = crypten.randn(1, 3, 256, 256)  # Batch size of 1, 3 channels, 256x256 image

    startpre_time = time.time()
    result_enc = private_model.forward(dummy_input)
    endpre_time = time.time()
    print(f"Start time: {startpre_time} End time: {endpre_time}")
    print(f"Total time for prediction cost: {(endpre_time - startpre_time)*1000} ms")
    print("Output shape:", result_enc.get_plain_text().shape)
