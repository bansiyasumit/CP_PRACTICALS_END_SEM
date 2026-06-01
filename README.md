# CUDA Python Data Processing (Numba)

This repository contains a collection of Python-based CUDA kernels implemented using `numba.cuda`. It demonstrates fundamental to advanced GPU programming techniques, focusing on parallel execution and memory access efficiency.

## 🚀 Project Overview

1. **Easy (CO2) – Diagonal Extraction:** Extracts the main diagonal from a 1024x1024 matrix using a 1D grid of CUDA threads, reducing an $O(N)$ sequential loop into an $O(1)$ parallel operation per thread.
2. **Moderate (CO3) – Lower Triangular Matrix:** Generates a lower triangular matrix from a 1024x1024 matrix utilizing a 2D mapping of GPU threads over a million individual cells.
3. **Hard (CO4) – Honeycomb Lattice & Memory Analysis:** Procedurally generates a hexagonal interference pattern on a 2048x2048 grid using mathematical wave superposition, designed to demonstrate **coalesced memory access**.

## 💻 How to Run in Google Colab

1. Create a new notebook in [Google Colab](https://colab.research.google.com/).
2. Go to **Runtime > Change runtime type**.
3. Select **T4 GPU** as the Hardware accelerator and click **Save**.
4. Copy the code into Colab cells and execute.

## 🧠 Deep Dive: Memory Access Efficiency

The Honeycomb Lattice generator (Q3) is engineered to maximize GPU memory bandwidth utilization:
* **Row-Major Memory:** In Python/NumPy, 2D arrays are stored continuously row-by-row in physical RAM.
* **Coalesced Memory Access:** By structuring our CUDA thread blocks in a `(16, 16)` 2D grid, threads executing simultaneously within the same warp naturally ask for adjacent memory addresses in the same matrix row.
* **Result:** The GPU bundles these adjacent requests into a single, massive memory transaction, drastically reducing trips to RAM and allowing the kernel to run at near-maximum hardware speed.
