# SRCNN From Scratch: ML Systems Optimization with CUDA

Welcome to a hands-on ML Systems Engineering journey — where we take a Super-Resolution Convolutional Neural Network (SRCNN) from a bare-bones MVP to a fully optimized CUDA-powered inference engine.

> **Goal:** This project is not just about building a neural network. It’s about engineering an optimized system — with CUDA kernels, memory hierarchies, and benchmark-driven insights.

---

## Series Overview

This is an 8-part blog series focused on building and optimizing an SRCNN from the ground up, with code, profiling, and GPU performance as the guiding principles. Each part ships working code and measurable improvements.

---

### Project Progression

| Part | Title | Focus |
|------|-------|-------|
| **1** | MVP First: A Minimal Working SRCNN | NumPy/C++ inference-only baseline |
| **2** | Profiling & First CUDA Kernel | Bottlenecks + naive CUDA conv2D |
| **3** | Shared Memory & Tiling | Optimized conv2D kernel |
| **4** | Layer Fusion & Full Inference Path | Conv → ReLU → Conv → ReLU → Conv |
| **5** | Pinned Memory, Streams & CUDA Graphs | Async memory, graph replay |
| **6** | Advanced CUDA Optimization | Occupancy, divergence, batch tuning |
| **7** | Mixed Precision & Tensor Cores | FP16 inference + Tensor Core acceleration |
| **8** | Final Deployment & Benchmark Recap | Jetson, cloud, full system benchmark |

---

## Getting Started

```bash
git clone https://github.com/your-username/srcnn-cuda-opt
cd srcnn-cuda-opt
mkdir build && cd build
cmake ..
make
./superres input.jpg output.jpg

