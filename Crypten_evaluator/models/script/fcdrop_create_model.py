import crypten
import crypten.nn as cnn
import random
import torch
import time

'''
Linear (here)
Dropout2d (here)
# Softmax (not here)
# AvgPool2d (not here)
# BatchNorm2d (not here)
'''

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
    # Prepare crypten
    print("Start Crypten...")
    crypten.init()

    # Create an instance of the dynamic model
    random_seed = 42
    dynamic_model = DynamicModel(random_seed=random_seed)

    # Save the non-encrypted model
    save_path = "/home/eloise/crypten/models/"
    crypten.save_from_party({
        'model_config': {
            'random_seed': dynamic_model.random_seed,
        },
        'model_state_dict': dynamic_model.state_dict(),
    }, f"{save_path}fcdrop_model_rs{random_seed}.pth", src=0)
    print("Non-encrypted Model Saved.")

    private_model = dynamic_model.encrypt()

    print("\nNetwork encrypted:")
    for _, curr_module in private_model._modules.items():
        print("Test here:", curr_module)
        for idx, module in curr_module._modules.items():
            print(idx, module)

    # Test the model with a dummy input
    dummy_input = crypten.randn(1, 3, 256, 256)  # Batch size of 1, 3 channels, 256x256 image

    print("\nTest on prediction:")
    startpre_time = time.time()
    result_enc = private_model.forward(dummy_input)
    endpre_time = time.time()
    print(f"Time for prediction cost: {(endpre_time - startpre_time)*1000} ms")
    print("Output shape:", result_enc.get_plain_text().shape)
        

