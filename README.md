# CUDA & GPU Programming — Code Overview

This repository contains CUDA programs and supporting files written to explore GPU programming, memory behavior, and kernel execution using NVIDIA CUDA.

---

## CPU Reference Code

### 2D Thermal Simulation (CPU)
- Implements a finite-difference heat diffusion update on a 2D grid
- Each grid cell updates its temperature using its four neighbors and a power term
- Runs iteratively over multiple timesteps
- Used as a reference CPU implementation for correctness and timing

---

## CUDA Programs

### Vector Addition (`vector_add.cu`)
- Performs element-wise addition of two input vectors on the GPU
- Allocates device memory using `cudaMalloc`
- Copies input data from host to device using `cudaMemcpy`
- Launches a one-dimensional CUDA grid
- Uses bounds checking inside the kernel
- Copies the output vector back to the host

---

### Element-wise Multiply and Scale (`elementwise_mul_scale.cu`)
- Multiplies two vectors element-wise and applies a scalar scaling factor
- Uses the same memory allocation, transfer, and kernel launch structure as vector addition
- Includes bounds checking for safety

---

### ReLU Activation (`relu.cu`)
- Applies the ReLU operation (`max(0, x)`) to each element of an input vector
- One CUDA thread processes one vector element
- Uses explicit bounds checking
- Transfers results back to the host

---

## Memory Access Pattern Experiments

### Coalesced Global Memory Kernel
- Threads in a warp access consecutive memory locations
- Demonstrates standard contiguous global memory access
- Used to study warp-level memory behavior

---

### Non-coalesced Global Memory Kernel
- Threads access global memory with a stride
- Uses the same computation as the coalesced kernel
- Intended to contrast memory access patterns

---

## Shared Memory Kernel

### Sliding Window Sum (Shared Memory)
- Computes a local sliding-window sum over a vector
- Loads data into shared memory before computation
- Uses `__syncthreads()` to synchronize threads in a block
- Each thread computes one output element
- Includes bounds checks and shared memory indexing

---

## Build & Run

All CUDA programs are compiled using `nvcc`:

```bash
nvcc filename.cu -o output
./output
