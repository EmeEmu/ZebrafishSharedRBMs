Here's a draft README for your repository:

---

# RBM-Multi-Zebra Paper

This repository contains the code and data (managed with `git-annex`) for the Multi-Fish RBM paper by Matteo Dommanget-Kott. It is written primarily in Julia and includes Jupyter notebooks for analysis.

---

## Layout of the Repository

The repository is organized as follows:

```
.
├── Analysis/                       # Analysis notebooks
├── Data/                           # Storage directory for data 
├── Figures/                        # Figures notebooks + output figures 
│   ├── Panels/                     # Output directory for individual panels 
│   └── setup.jl                    # Imports, paths, and color conventions
├── Misc_Code/                      # Storage directory for miscellaneous code
├── Misc_Data/                      # Storage directory for miscellaneous data and intermediate steps
├── Models/                         # Storage directory for RBMs
├── Modules/                        # A collection of useful code from other repositories
├── Project.toml                    # Julia dependencies
├── Manifest.toml                   # Julia reproducible environment
└── README.md                       # You are here!
```

---

## Instructions for Cloning the Repository

To clone this repository, including its submodules, run the following commands:

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/EmeEmu/MultiFishRBM.git

# If you already cloned the repository without submodules, initialize and update submodules
cd MultiFishRBM
git submodule update --init --recursive
```

---


## Setting Up Julia and Using `Pluto.jl`

### 1. Install Julia

This repository requires Julia (version 1.11.5 or later). 

- Download the appropriate Julia version for your platform from the [official Julia website](https://julialang.org/downloads/) or using [Juliaup](https://github.com/JuliaLang/juliaup) to manage multiple version of Julia.
- Follow the installation instructions provided on the website.

### 2. Install Required Julia Packages

Once Julia is installed, open a Julia REPL and add the necessary packages, starting with [Pluto.jl](https://plutojl.org/) for Julia notebooks :

```julia
using Pkg
Pkg.add("Pluto")
```

The repository includes its own environment for package dependencies (`Project.toml` and `Manifest.toml`). You can instantiate the project's environment as follows:

```julia
cd("/path/to/repo/")
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
This will install and pre-compile all the necessary packages.

### 3. Running `Pluto.jl`

The analysis notebooks in this repository are designed to be run using `Pluto.jl`. To open a notebook:

1. Launch the Pluto server:
   ```julia
   using Pluto
   Pluto.run()
   ```

2. Open the `.jl` notebooks in the `Analysis/` or `Figures/` directories from the Pluto interface.

---

## The Data

TODO

---

## Running the code

The present repository contains all the code and data used to produce the analysis and figures in ????. The following section describes the pipeline to reproduce the analysis.

### 1. Creating the datasets

All datasets are already precomputed, hosted on Zenodo [here](example.com), and available through the `DataDeps` prodedure described above.

The rest of this section describes how to recompute them, from raw data (available from the authors on request). 

#### 1.1. Whole-brain single-neuron datasets

Constuction of Whole-brain single-cell datasets is done with the notebook `Analysis/Dataset_builder.jl`. This creates `DATA_FishName.h5` files stored in `DataAndModels/Data_WBSC/WBSC/`.

#### 1.2. Voxelized datasets

Constuction of Whole-brain voxelized datasets is done with the notebook `Analysis/Dataset_builder.jl`. This creates `VOXgrid.h5` files stored in `DataAndModels/Data_VOX/Voxelgrids/`, and `VOXsize_FishName.h5` files stored in `DataAndModels/Data_VOX/Voxelized/`.



---

## Figures

All figures should be saved as `.svg` files. Ensure to follow the file structure within the `Figures/` directory for organization.

In order to make this easier from Pluto notebooks, a custom function is available at `Misc_Code/fig_saving.jl` which automatically handles figure saving.

Example usage :
```julia
include("Misc_Code/fig_saving.jl")
figure = Figure()
...
save(@figpath("figure_title"), figure)
```

---
