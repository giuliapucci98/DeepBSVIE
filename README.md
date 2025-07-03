# DeepBSVIE

This repository contains a deep learning framework for solving Backward Stochastic Volterra Integral Equations (BSVIEs). The implementation is based on a recursive backward representation introduced in the paper:  
Deep BSVIEs Parametrization and Learning-Based Applications  
http://arxiv.org/abs/2507.01948

The repository is structured to handle both standard and reflected BSVIEs through separate branches.

## Overview

We implement a discrete-time neural scheme to approximate the solution of BSVIEs. The method leverages feedforward neural networks, trained recursively on a two-time grid, to learn the solution fields (Y(t), Z(t,s)). The solver accommodates both classical and reflected BSVIEs, making it suitable for applications in time-inconsistent stochastic control and recursive utilities.

## Repository Structure

- `train.py` – Training script for the neural solver; saves learned network parameters.  
- `test.py` – Evaluation script to test the performance of the trained networks.  
- `BSVIE.py` – Defines the BSVIE model, discretization scheme, and neural network architecture.

## Branches

- `main` – Standard BSVIE solver  
- `reflected` – Extension to reflected BSVIEs with obstacle constraints

## Reference

If you find this code useful, please consider citing:

Agram, N. and Pucci, G. (2025). Deep BSVIEs Parametrization and Learning-Based Applications.  
arXiv:2507.01948  
http://arxiv.org/abs/2507.01948
