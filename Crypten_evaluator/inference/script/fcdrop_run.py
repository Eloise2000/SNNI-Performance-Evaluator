"""
Author: Eloise Zhang
Date: 12 July 2024
Description: This script runs SNNI in 2PC, and tests the inference runtime for Flatten, Linear and Dropout layers.
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

        # Save the random seed for reproducibility
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)

        # Initialize an empty list to store layers
        self.layers = cnn.ModuleList()

        # Flatten layer (starting from the second dimension)
        self.layers.append(cnn.Flatten(axis=1))

        # Randomly determine the number of layers (conv + relu + maxpool) within a range
        num_layers = random.randint(15, 20)
        in_size = 256*256*3

        for _ in range(num_layers):
            out_size = random.randint(2, 1024)
            self.layers.append(cnn.Linear(in_features=in_size, out_features=out_size))
            in_size = out_size

            cur_p = random.choice([i * 0.1 for i in range(1,10)])
            self.layers.append(cnn.Dropout(p = cur_p))

        self.layers.append(cnn.Linear(in_features=in_size, out_features=1))

    def forward(self, x):
        flatten_cnt = 0
        linear_cnt = 0
        dropout_cnt = 0

        for i, layer in enumerate(self.layers):
            if isinstance(layer, cnn.Flatten):
                flatten_cnt += 1
                N, CI, HI, WI = x.size()

                start_time = time.time()
                x = layer(x)
                end_time = time.time()

                axis = layer.axis
                N, FO = x.size()
                print(f"Layer{i}: Flatten{flatten_cnt} - Features: Batch_Size={N}, In_Height={HI}, In_Width={WI}, In_Channels={CI}, Out_Features={FO} - Output_Shape: {x.size()} - Start_time: {start_time} - End_time: {end_time} - Time_used: {(end_time-start_time)*1000} ms")

            if isinstance(layer, cnn.Linear):
                linear_cnt += 1
                N, FI = x.size()
                
                start_time = time.time()
                x = layer(x)
                end_time = time.time()

                N, FO = x.size()
                print(f"Layer{i}: Linear{linear_cnt} - Features: Batch_Size={N}, In_Features={FI}, Out_Features={FO} - Output_Shape: {x.size()} - Start_time: {start_time} - End_time: {end_time} - Time_used: {(end_time-start_time)*1000} ms")

            if isinstance(layer, cnn.Dropout):
                dropout_cnt += 1

                start_time = time.time()
                x = layer(x)
                end_time = time.time()

                N, F = x.size()
                p = layer.p
                print(f"Layer{i}: Dropout{dropout_cnt} - Features: Batch_Size={N}, Features={F}, P={p} - Output_Shape: {x.size()} - Start_time: {start_time} - End_time: {end_time} - Time_used: {(end_time-start_time)*1000} ms")

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
        checkpoint = torch.load(f"{model_folder_path}fcdrop_model_rs{random_seed}.pth")
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
