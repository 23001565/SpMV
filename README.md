⭐ Project Overview

The repository includes the following modules:

- mtx2coo.cpp — Convert Matrix Market .mtx files to .coo

- csr.cpp — Convert .coo to CSR + SpMV & SpMVᵀ

- csb.cpp — Convert .coo to CSB + SpMV & SpMVᵀ

These are simple versions designed to illustrate the flow of sparse matrix algorithms.

📂 Project Components
1. 🔧 mtx2coo.cpp

Converts a Matrix Market (.mtx) file into COO (.coo) format.

✔ Features

- Supports common .mtx variations:

- Symmetric matrices (upper/lower triangular storage).

- Pattern matrices (no value column → automatically filled with 1.0).

- Zero entries in .mtx are omitted.

▶ Usage
./mtx2coo <input.mtx> <output.coo>

2. 🧮 csr.cpp

Converts COO format to CSR and performs SpMV computations.

✔ Features

- COO → CSR transformation

- Compute:

  + A × x

  + Aᵀ × x using: atomic operations and temporary auxiliary buffers

- Supports multi-threading via command-line input.

▶ Usage
./csr matrix.coo [num_threads]

3. 📦 csb.cpp

Implements the Compressed Sparse Blocks (CSB) format and its SpMV operations.

✔ Features

- COO → CSB conversion

- Compute: A × x and Aᵀ × x

- Supports block size tuning via beta parameter

▶ Usage
./csb matrix.coo [num_threads] [beta]


⚠️ This implementation is not optimized and serves mainly to demonstrate algorithm flow.

A full, high-performance version requires handling:

- bit-level data layouts
  
- careful thread scheduling

- memory alignment and L2/L3 cache behavior

🔗 References

Original authors' implementation:
[https://people.eecs.berkeley.edu/~aydin/csb/html/files.html]

Simplified CSB reference implementation:
[https://github.com/Luke2336/Compressed-Sparse-Blocks/blob/master/pybind/_csb.cpp]

📊 Experimental Data

The datasets for testing come from the SuiteSparse Matrix Collection:
[https://sparse.tamu.edu/]

✔ Matrices used

- Four square sparse matrices of increasing size: 320k, 680k, 921k, 11M

- The CSB implementation cannot handle the 11M

✔ Data storage

- Due to file size limitations, all .mtx and corresponding .coo files are stored in Google Drive:

🔗 [https://drive.google.com/drive/folders/17KekcCttRVR-pirTUO6Fuj7wKCEX3q2Z?usp=drive_link]

- The folder includes:

  + Original .mtx matrices

  + Generated .coo files (via mtx2coo.cpp)

🏗️ Build Instructions

Compile all programs:

- g++ -O3 -fopenmp csb.cpp -o csb
- g++ -O3 -fopenmp csr.cpp -o csr
- g++ -O3 mtx2coo.cpp -o mtx2coo


📈 Goal of This Project

This project is intended for:

- Understanding sparse matrix format CSB and its' parallel algorithms

- Visualizing the flow of SpMV and SpMVᵀ

- Comparing with CSR format and algorithms


