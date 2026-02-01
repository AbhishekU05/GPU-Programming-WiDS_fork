# GPU Programming (Weeks 1–6)

## Overview
This repository contains my work for the GPU Programming module, spanning Weeks 1 through 6.
The focus of the course is understanding GPU architectures, identifying performance bottlenecks,
and accelerating real computational workloads using CUDA and Triton.

The work progresses from basic GPU concepts and profiling to implementing and benchmarking
custom GPU kernels for a realistic numerical workload.

---

## Week 1–2: Foundations and CPU Baselines
The initial weeks focus on:
- Understanding GPU execution models (threads, blocks, warps)
- Identifying compute- and memory-bound workloads
- Writing clean CPU baseline implementations

Emphasis is placed on measuring performance correctly and understanding why naive CPU
implementations often fail to scale for large numerical problems.

---

## Week 3–4: GPU Kernels and Performance Analysis
In these weeks, performance-critical sections of code are ported to the GPU using CUDA.
Key topics include:
- Mapping data-parallel problems to GPU threads
- Memory access patterns and coalescing
- Kernel launch configuration
- Correct benchmarking using device synchronization

Custom CUDA kernels are implemented and compared against CPU baselines to quantify speedup.

---

## Week 5: Triton and Higher-Level GPU Programming
This phase introduces Triton as a higher-level alternative to CUDA.
The same computational kernels are reimplemented in Triton to study:
- Programmer productivity vs control
- Performance parity with hand-written CUDA
- Block-based execution and memory access

CUDA and Triton implementations are benchmarked under identical conditions.

---

## Week 6: Final Mini-Project — IC Thermal Simulation
The final project applies all previous concepts to a realistic workload: a two-dimensional
integrated circuit (IC) thermal simulation.

The simulation solves the heat diffusion equation using a finite-difference stencil over a
1024×1024 grid for 1000 time steps. Three implementations are evaluated:
- A CPU baseline using NumPy and Python loops
- A custom CUDA kernel compiled via a PyTorch extension
- A Triton kernel implementing the same stencil computation

Performance results demonstrate over four orders of magnitude speedup on the GPU compared
to the CPU baseline. CUDA and Triton achieve nearly identical runtimes, indicating a
memory-bound workload with high parallel efficiency.

---

## Key Takeaways
- Correct benchmarking is as important as kernel optimization
- Python loop-based CPU implementations scale poorly for stencil workloads
- GPUs excel at regular, data-parallel numerical computations
- Triton provides performance comparable to CUDA with reduced development complexity
- Understanding memory behavior is critical for interpreting speedups

---

## Notes
- GPU timings exclude one-time compilation overhead
- All GPU benchmarks use explicit device synchronization
- Experiments were conducted on an NVIDIA GTX 1650 Ti
- Single-GPU execution is assumed throughout
