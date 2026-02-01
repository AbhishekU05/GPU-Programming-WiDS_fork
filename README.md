# GPU Acceleration of IC Thermal Simulation

## Overview
This project accelerates a 2-D integrated-circuit (IC) thermal simulation using GPU programming.
The workload solves the heat diffusion equation on a 1024×1024 grid over 1000 time steps.

We implement:
- A CPU baseline (NumPy)
- A custom CUDA kernel
- A Triton kernel

and benchmark their performance.

## Physics Model
We solve the discrete heat equation with a source term:

T_{i,j}^{t+1} = T_{i,j}^t + αΔt · ∇²T_{i,j} + P_{i,j}

where the Laplacian ∇²T is approximated using a 5-point stencil.

Boundary cells are held fixed.

## Requirements
- NVIDIA GPU (tested on GTX 1650 Ti)
- CUDA Toolkit
- Python 3.9+
- PyTorch
- Triton

## How to Run
```bash
python cpu_baseline.py
python benchmark.py
