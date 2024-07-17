# Evaluator for Crypten <!-- omit from toc -->
## Introduction <!-- omit from toc -->
Here is a step-by-step tutorial on how to build and run the evaluator for Secure Neural Network Inference (SNNI) using the secure computing protocol achieved by Crypten. [CrypTen](https://github.com/facebookresearch/CrypTen?tab=readme-ov-file) is a framework for Privacy-Preserving Machine Learning, built on PyTorch and based on Secure Multiparty Computation. In this tutorial, we will guide you through constructing various network architectures, executing secure inference within Crypten, collecting runtime data of each layer for each architecture, and developing analytical prediction models to estimate the runtime of each layer. This work can be applied in [Neural Architecture Search](https://en.wikipedia.org/wiki/Neural_architecture_search) for exploring efficient neural network architectures for secure inference.

## Contents <!-- omit from toc -->
- [How to Run the Experiment](#how-to-run-the-experiment)
- [Getting Started](#getting-started)
  - [Installing CrypTen](#installing-crypten)
  - [Building Models on the Server Side](#building-models-on-the-server-side)
  - [Running SNNI in 2PC setting](#running-snni-in-2pc-setting)
    - [Quick Start](#quick-start)
    - [Setting up Throttle](#setting-up-throttle)
    - [For Batch Testing](#for-batch-testing)
    - [Other Tests](#other-tests)
  - [Building Models for Performance Evaluation](#building-models-for-performance-evaluation)
    - [Data Pre-processing](#data-pre-processing)
    - [Build Analytical Model](#build-analytical-model)
- [Troubleshooting](#troubleshooting)
  - [Error Code 12: Cannot Allocate Memory](#error-code-12-cannot-allocate-memory)


## How to Run the Experiment
The experiment includes three main steps:
1. Constructing various neural network models
2. Running secure neural network inference in a 2PC setting
3. Building analytical models for performance evaluation

## Getting Started
### Installing CrypTen

For Linux or Mac, run the following command:

```sh
pip install crypten
```

### Building Models on the Server Side

Navigate to the `generate_model` directory to generate random PyTorch models and save as `.pth` files.

```sh
cd generate_model
```

#### For Layers: Conv, Relu, AvgPool, and BatchNorm <!-- omit from toc -->

```sh
bash run_multiple_seeds_avgbn.sh
```

#### For Layers: Conv, Relu, and MaxPool <!-- omit from toc -->
```sh
bash run_multiple_seeds_convmp.sh
```

#### For Layers: Linear and Dropout <!-- omit from toc -->
```sh
bash run_multiple_seeds_fcdrop.sh
```

#### Notes: <!-- omit from toc -->

- **Random Model Initialization:** All model parameters are randomly initialized. For specific models, define them directly in the `create_model` file using PyTorch.
  
- **Random Seeds:** The bash scripts use sample random seeds. Modify these scripts to use additional seeds as needed.
  
- **Convolutional Layer Architecture:** Convolutional layers are defined in two separate model files to provide architectural variety.
  
- **Testing Flag (`TEST_FLAG`):** This flag determines whether the generated models are tested during secure inference in a single party environment to catch errors.
  
- **Empirical Map:** The `empirical_map` is manually configured for random generation of Convolutional layers to avoid memory allocation errors. Refer to the [troubleshooting](#troubleshooting) for more details.

### Running SNNI in 2PC setting
#### Quick Start
Navigate to the `inference` directory to run inference in 2PC setting.

1. Include `common.sh` and `script/*_run.py` scripts on both the client and server sides, and install Crypten on both sides.

2. Include the `.pth` pretrained model file in the **server** `/generate_model/models` directory.

3. Modify `common.sh`:
   - For both the client and server:
     - Change `MASTER_ADDR` to the IP address of the master.
   - For the client:
     - Set `RANK=0`.
   - For the server:
     - Set `RANK=1`.

4. Run the following commands on both the client and server sides:
    ```sh
    bash common.sh
    python3 script/*_run.py <seed>
    ```
    Replace `<seed>` with the same random seed used for building the model architecture.

#### Notes: <!-- omit from toc -->
- The pretrained models should be only in the server side, as they are considered as the server's property in SNNI.

- For demonstration purposes, we use a dummy input for inference. If you intend to use real input data, ensure it remains only on the client side. Initiate an all-zero input of the same size on the server side.

- During inference, the features for each layer and their corresponding runtime will be printed out.

#### Setting up Throttle

If you wish to set up bandwidth and ping latency in SNNI by manipulating traffic control (tc) settings on a network interface in Linux, you can use the `throttle.sh` file.

- Setup bandwidth and ping latency to mimic LAN (Local Area Network):
    ```sh
    bash throttle.sh lan
    ```
    For demonstration, this sets bandwidth to 7 Gbps and ping latency to 0.3 ms.

- Setup bandwidth and ping latency to mimic WAN (Wide Area Network):
    ```sh
    bash throttle.sh wan
    ```
    For demonstration, this sets bandwidth to 1 Gbps and ping latency to 10 ms.

- Delete the throttle:
    ```sh
    bash throttle.sh del
    ```

- Test throttle:
    - Check latency: `ping <server_ip>` (replace `<server_ip>` with actual server IP)
    - Check bandwidth:
        - Server side: `iperf3 -s`
        - Client side: `iperf3 -c <server_ip> -t 10`

#### For Batch Testing
If you want to run multiple random seeds several times to gather runtime data for each layer, refer to the `run_*.sh` bash files. These scripts demonstrate how to collect data and save the log out file on the server side.

#### Other Tests
Apart from runtime collection, other tests can be conducted during SNNI. For example, you can test memory cost and CPU usage using `psrecord`, or test energy cost during SNNI. Sample plots using `psrecord` can be found in the `sample_plot` folder.

### Building Models for Performance Evaluation

Navigate to the `evaluation` directory to process the raw data and build analytical models for each layer.

#### Data Pre-processing

First, we pre-process the layer runtime data from logged TXT file format to CSV file format for each layer, including the layer's features and runtime.

```sh
python3 ./generate_data/generate_<layer>_data.py
```
Modify `<layer>` to the specific layer you are processing.

#### Build Analytical Model

## Troubleshooting

### Error Code 12: Cannot Allocate Memory

This runtime error can occur when the model requires more memory allocation for secure computing than the hardware can provide. It can be detected during the test of secure inference.

To mitigate this issue, an `empirical_map` has been created for the Convolutional layer to avoid memory allocation errors in an 8GB RAM CPU-only environment. Additionally, empirical constraints are applied to the MaxPool layer in the `check_maxpool` function.
