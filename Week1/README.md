
# **Week 1 — GPU Intuition & Compute Foundations**


##  **Learning Goals**

By the end of this week, you should be able to:

* Explain the architectural differences between CPUs and GPUs
* Understand SIMT parallelism, warps, and throughput computing
* Identify workloads that benefit from GPU acceleration
* Understand the CUDA execution hierarchy (threads → blocks → grids)
* Identify at least one potential GPU-accelerable task from your own research or interests

This week builds your mental model so everything later (memory, performance, kernels) fits together.


## **Concepts Covered This Week**

* CPU vs GPU architecture
* SIMD vs SIMT
* Warps, streaming multiprocessors, and massive parallelism
* Throughput computing vs latency computing
* CUDA execution hierarchy
* Identifying GPU-friendly workloads
* Profiling intuition: *what part of your code is slow? why?*


## **Required Resources**

### **1. NVIDIA CUDA Programming Guide (Conceptual Introduction)**

Read: **Chapter 1 — Introduction**
[https://docs.nvidia.com/cuda/cuda-programming-guide/](https://docs.nvidia.com/cuda/cuda-programming-guide/)

Focus on:

* What GPUs are optimized for
* Parallel execution model
* Overview of CUDA threads/blocks/grids



### **2. UC Berkeley CS61C — Parallel Processors (Lecture 17)**

Video:
[https://www.youtube.com/watch?v=gychBEOgG8A](https://www.youtube.com/watch?v=gychBEOgG8A)
(If link breaks, search: *CS61C Parallel Processors Lecture 17*)

This gives the “physics layer” of performance:

* Why CPUs stopped getting faster
* Why GPUs thrive on massive parallelism
* Memory bottlenecks and throughput considerations



### **3. Stanford CS231n — Hardware/Software Interface (Lecture 15)**

Slides & Video:
[https://www.youtube.com/watch?v=WGf1f2HbJpE](https://www.youtube.com/watch?v=WGf1f2HbJpE)

Focus on:

* GPU in the modern AI stack
* How frameworks like PyTorch map operations to hardware


##  **Week 1 Assignment (Summary)**

Your full assignment instructions are in `assignment.md`.
Here is a preview of what you will do:

### **Task 1 — Identify a GPU-Accelerable Problem**

Pick a workload you care about:

* Research simulation
* Numerical solver
* ML operation / training bottleneck
* Data analysis pipeline
* Anything computation-heavy in Python

Write a short description of **why** it may benefit from GPU acceleration.



### **Task 2 — GPU Execution Model Diagram**

Draw (on paper or digitally) the CUDA hierarchy:

```
Grid
 └── Blocks
       └── Threads
```

Annotate each level and briefly describe how your chosen workload might map to it.



**Submission folder:**
Place your answers in the `week1` folder:

```
week1/
 ├── assignment.md
 ├── diagrams/ (if any)
 └── notes.md (optional)
```



## **Optional Extra Resources**

* “Even Easier Introduction to CUDA” — NVIDIA Blog
  [https://developer.nvidia.com/blog/even-easier-introduction-cuda/](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
* “How GPUs Work” — Stanford CME193 notes

