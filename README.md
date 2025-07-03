# DeepBSVIE

This repository implements a **discrete-time learning scheme** for solving **Backward Stochastic Volterra Integral Equations (BSVIEs)** based on a recursive backward representation introduced in: [Deep BSVIEs Parametrization and Learning-Based Applications](http://arxiv.org/abs/2507.01948). The repository is organized into two branches to handle standard and reflected BSVIEs, respectively.


## 🔍 Overview

The code provides a deep learning-based approach to approximate solutions of BSVIEs. It uses feedforward neural networks trained in discrete time to learn the solution of these path-dependent integral equations, with the option to handle reflected versions of BSVIEs.

## 📂 Repository Structure

- `train.py`  
  Trains the neural networks and saves the model parameters.

- `test.py`  
  Loads the trained parameters and evaluates the network performance.

- `BSVIE.py`  
  Contains the mathematical model specification, neural network architecture, and the solver for the BSVIE.

## 🌿 Branches

- **`main`** – Implements the scheme for standard BSVIEs.
- **`reflected`** – Includes the extension to **reflected BSVIEs**.
