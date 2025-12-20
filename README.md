# Large-Scale-Clustering---Big-Data-Analysis
## Large-Scale Distributed Clustering with MPI
Parallel K-Means and Hierarchical Clustering on NYC Construction Permit Data


---

## Project Overview

This project implements scalable clustering algorithms using distributed computing to analyze large-scale, high-dimensional data. We parallelize K-Means and Hierarchical Clustering using MPI to efficiently process 3.8 million NYC construction permit records, overcoming the memory and runtime limitations of single-node systems.

The project demonstrates how feature selection, PCA, and message passing enable near real-time clustering on administrative datasets in an HPC environment.

---

## Research Question

Can we cluster millions of records accurately and efficiently using distributed computing?

---

## Dataset

- Source: NYC Open Data – DOB Permit Issuance
- Time Period: 2013–2025
- Size: ~3.8 million records × ~60 columns
- Data Characteristics:
  - High-dimensional
  - Mixed data types (numerical, categorical, spatial, temporal)
  - No ground truth (unsupervised learning)

---

## Motivation

- Identify construction activity patterns across NYC
- Analyze permit processing behavior
- Detect anomalies such as unusual delays
- Apply scalable machine learning techniques to real-world big data

---

## Methods

### Feature Selection and Dimensionality Reduction
- Removed noisy identifiers (names, phone numbers, addresses)
- Reduced features from 60 → 22
- Applied PCA: 22 → 16 components
- Preserved ~96% of total variance

---

### Parallel K-Means with MPI

- Rank 0 initializes centroids using K-Means++
- The dataset is partitioned across MPI processes
- Each process computes local cluster assignments in parallel
- Global centroid updates via MPI_Reduce
- Iterates until convergence

K-Means scales efficiently due to minimal communication overhead and high local computation.

---

### Distributed Hierarchical Clustering

- Partition-based agglomerative clustering
- Ward linkage for distance computation
- Parallel distance calculations across MPI processes
- Global synchronization using MPI_Bcast
- Primarily used for exploratory analysis and dendrogram construction

---

## Validation Metrics

Multiple metrics are used to ensure clustering quality:

- Silhouette Score (cohesion and separation)
- Davies–Bouldin Index (cluster similarity)
- Calinski–Harabasz Index (variance ratio)
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)

Using multiple metrics provides robust validation across different clustering aspects.

---

## Simulation: Proof of Correctness

Synthetic data was generated using `sklearn.make_blobs` to validate correctness:

- 3.8 million samples
- 22 dimensions
- 10 known clusters

Results:
- ARI = 1.000
- NMI = 1.000
- Strong silhouette score
- Converged in 4 iterations

This confirms exact recovery of the true cluster structure.

---

## Performance Results

### K-Means MPI on NYC Data

- Baseline (1 core): 291.6 seconds
- Best time (64 cores): 8.19 seconds
- Speedup: 35.6×

Communication overhead remains low (~4.4%) even at high process counts.

---

### Hierarchical MPI Scaling

- Achieved super-linear speedup (402×) on smaller datasets
- Enabled by improved cache utilization
- Not scalable to the full 3.8M dataset due to memory constraints

---

## Key Findings

- K-Means MPI scales efficiently to millions of records
- Clustering quality is preserved at full scale
- Feature selection and PCA are critical for scalability
- Hierarchical clustering offers speed but sacrifices quality at scale
- Cache effects can produce super-linear speedups

---

## Implementation Details

### HPC Environment
- Cluster: SeaWulf HPC (Stony Brook University)
- Scheduler: SLURM
- Architecture: Distributed-memory, multi-node system

### Software Stack
- Python 3.11.2
- OpenMPI
- mpi4py
- NumPy, scikit-learn, matplotlib

---

## Repository Structure

