# Evaluator for Crypten
## Introduction
Here is a step-by-step tutorial on how to build and run the evaluator for Secure Neural Network Inference (SNNI) using the secure computing protocol achieved by Crypten. [CrypTen](https://github.com/facebookresearch/CrypTen?tab=readme-ov-file) is a framework for Privacy-Preserving Machine Learning, built on PyTorch and based on Secure Multiparty Computation. In this tutorial, we will guide you through constructing various network architectures, executing secure inference within Crypten, collecting runtime data of each layer for each architecture, and developing analytical prediction models to estimate the runtime of each layer. This work can be applied in [Neural Architecture Search](https://en.wikipedia.org/wiki/Neural_architecture_search) for exploring efficient neural network architectures for secure inference.

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

Navigate to the `generate_model` directory to generate random PyTorch models saved as `.pth` files.

```sh
cd generate_model
```

#### For Layers: Conv, Relu, AvgPool, and BatchNorm

```sh
bash run_multiple_seeds_avgbn.sh
```

#### For Layers: Conv, Relu, and MaxPool
```sh
bash run_multiple_seeds_convmp.sh
```

#### For Layers: Linear and Dropout
```sh
bash run_multiple_seeds_fcdrop.sh
```

#### Notes:

- **Random Model Initialization:** All model parameters are randomly initialized. For specific models, define them directly in the `create_model` file using PyTorch.
  
- **Random Seeds:** The bash scripts use sample random seeds. Modify these scripts to use additional seeds as needed.
  
- **Convolutional Layer Architecture:** Convolutional layers are defined in two separate model files to provide architectural variety.
  
- **Testing Flag (`TEST_FLAG`):** This flag determines whether the generated models are tested during secure inference in a single party environment to catch errors.
  
- **Empirical Map:** The `empirical_map` is manually configured for random generation of Convolutional layers to avoid memory allocation errors. Refer to the [troubleshooting](#troubleshooting) for more details.





## Troubleshooting

### Error Code 12: Cannot Allocate Memory

This runtime error can occur when the model requires more memory allocation for secure computing than the hardware can provide. It can be detected during the test of secure inference.

To mitigate this issue, an `empirical_map` has been created for the Convolutional layer to avoid memory allocation errors in an 8GB RAM CPU-only environment. Additionally, empirical constraints are applied to the MaxPool layer in the `check_maxpool` function.
