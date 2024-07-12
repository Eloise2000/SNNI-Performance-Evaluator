# Introduction
Here is a step-by-step tutorial on how to build and run the evaluator for Secure Neural Network Inference (SNNI) using the secure computing protocol achieved by Crypten. [CrypTen](https://github.com/facebookresearch/CrypTen?tab=readme-ov-file) is a framework for Privacy-Preserving Machine Learning, built on PyTorch and based on Secure Multiparty Computation. In this tutorial, we will guide you through constructing various network architectures, executing secure inference within Crypten, collecting runtime data of each layer for each architecture, and developing analytical prediction models to estimate the runtime of each layer. This work can be applied in [Neural Architecture Search](https://en.wikipedia.org/wiki/Neural_architecture_search) for exploring efficient neural network architectures for secure inference.

# How to Run the Experiment
The experiment includes three main steps:
1. Constructing various neural network architectures
2. Running secure neural network inference in a 2PC setting
3. Building analytical models for performance evaluation

# Getting Started
1. Installing CrypTen

    For Linux or Mac, run the following command:

    ```sh
    pip install crypten
    ```
