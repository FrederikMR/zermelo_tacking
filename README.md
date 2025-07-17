# Time-dependent Zermelo navigation with tacking
This repository contains the implementation for computing pregeodesics for a time-dependent metric as well as computing tack-curves, where it is possible to change metric along the curve.

![Constructed geodesics using GEORCE and similar optimization algorithms](https://github.com/user-attachments/assets/ef87ce54-e80f-4fed-965a-1eb338c146d0)

## Installation and Requirements

The implementations in the GitHub is Python using JAX. To clone the GitHub reporsitory and install packages type the following in the terminal

```
git clone https://github.com/FrederikMR/zermelo_tacking.git
cd zermelo_tacking
pip install -r requirements.txt
```

The first line clones the repository, the second line moves you to the location of the files, while the last line install the packages used in repository.

## Code Structure

The following shows the structure of the code. All general implementations of geometry and optimization algorithms can be found in the "geometry" folder for both the Riemannian and Finsler case.

    .
    ├── load_manifold.py                   # Load manifolds and points for connecting pregeodesics and tack-curves
    ├── tacking.py                         # Computes pregeodesics and tack curves
    ├── tacking_estimation.ipynb           # Plots the tack curves and pregeodesics
    ├── tacking_cpu                         # Contains all results for cpu runs
    ├── tacking_gpu                         # Contains all results for gpu runs
    ├── geometry                           # Contains implementation of time-dependent Finsler manifolds as well as pregeodesic and tacking optimization algorithms, inlcuding GEORCE-H
    └── README.md

## Reproducing Experiments

All experiments can be re-produced by running the notebooks and the tacking.py package for the given manifold, hyper-parameters and optimization method.

## Logging

All experimental results for the runtime and length estimates are saved as .pkl files in the folder "tacking_gpu" and "tacking_cpu".

## Reference

If you want to use GEORCE for scientific purposes, please cite:



