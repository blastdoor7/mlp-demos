# Neural Network Experiments in C

This repository contains **from-scratch neural network implementations in C**, starting with a simple XOR MLP demo. The goal is to explore, learn, and experiment with various neural network architectures and training techniques without using external libraries.

## Current Demos

- **XOR MLP (`xor_mlp.c`)**  
  - 2 input neurons → 2 hidden neurons → 1 output neuron  
  - Sigmoid activation function  
  - Trained with backpropagation  
  - Fully from-scratch forward and backward passes  

*(More demos coming soon: e.g., toy regression, MNIST digits, different activations, and small-scale experiments.)*

## Usage

Compile and run each demo with GCC:

```bash
gcc -O2 xor_mlp.c -o xor_mlp -lm
./xor_mlp
