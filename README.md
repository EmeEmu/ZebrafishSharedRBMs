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

### 0. System requirements

All computations were performed on a Ubuntu 20.04 PC with the following specs :

- CPU : AMD Ryzen 9 3950X (16 cores)
- GPU : GeForce RTX 3080Ti (12GB)
- RAM : 128 GB

While not tested, everything should work with lower specs. An important limiting factor is the RAM which needs to be ~60GB for some computations.

### 1. Creating the datasets

All datasets are already precomputed, hosted on Zenodo [here](example.com), and available through the `DataDeps` prodedure described above.

The rest of this section describes how to recompute them, from raw data (available from the authors on request). 

#### 1.1. Voxelized datasets

Constuction of Whole-brain voxelized datasets is done with the notebook `Analysis/Dataset_builder.jl`. This creates `VOXgrid.h5` files stored in `DataAndModels/Data_VOX/Voxelgrids/`, and `VOXsize_FishName.h5` files stored in `DataAndModels/Data_VOX/Voxelized/`.

#### 1.2. Whole-brain single-neuron datasets

Constuction of Whole-brain single-cell datasets is done with the notebook `Analysis/Dataset_builder.jl`. This creates `DATA_FishName.h5` files stored in `DataAndModels/Data_WBSC/WBSC/`.


### 2. Training the RBMs

All RBMs are already precomputed, hosted on Zenodo [here](example.com), and available through the `DataDeps` prodedure described above.

The rest of this section describes how to recompute them. Bare in mind that RBMs are stochastic, therefore retraining will result in different models than those used in the article.
As training can be long, and some computation can require a lot of RAM or disk space, most notebooks contain interactive switches to launch computations on demand, and some cells have been disabled.

#### 2.1. Voxelized RBMs

We provide three notebooks to train and validate voxelized RBMs :

- `Analysis/VoxelizedSingleFishTraining.jl` : train an RBM from a single fish. The fish and training parameters can be selected interactively. Two methods are provided to train a single or repeated trainings.
- `Analysis/VoxelizedMultiFishTraining.jl` : train an RBM from multiple fish. The training parameters can be selected interactively. Two methods are provided to train a single or repeated trainings.
- `Analysis/Voxelized_CrossValidation.jl` : cross validate training hyperparameters for a single-fish training. The fish can be selected interactively.

#### 2.1. Single-Neuron RBMs

We provide three notebooks to train and validate Whole-Brain Single-Cell RBMs :

- `Analysis/WholeBrainSingleFishTraining.jl` : train an RBM from a single fish. The fish and training parameters can be selected interactively. Two methods are provided to train a single or repeated trainings.
- `Analysis/WholeBrainSingleFishStudentTraining.jl` : train a student RBM from a given teacher. Both teacher and students can be selected interactively. In order to run this code you first need to precompute the teacher weight maps using `Analysis/BuildingWeightMaps.jl` (be carefull this requires ~15GB extra disk space per fish).
- `Analysis/BACKUP1/MultiRBMPaper/Analysis/WholeBrainSingleFishAnalysis.jl` : investigate and evaluate a trained RBM. This notebook is not required for figures or other analyses, but is helpfull to evaluate RBMs.

### 3. Figures

#### 3.2. Fig.2 and S2-4

Figure 2 and associated supplementary figures are produced with the notebook `Figures/Fig.Voxelized.jl`. This notebook requires the voxelized dataset and voxelized RBMs, which are available as `DataDeps` or can be recomputed as described previously.

#### 3.3 Fig.3 and S5-8

Figure 3 and associated supplementary figures are produced with the notebook `Figures/Fig_Reinit_training.jl`. This notebook requires the whole-brain single-neuron dataset, teacher and student RBMs, which are available as `DataDeps` or can be recomputed as described previously. It also requires the precomputed metrics between weight maps `WeightDist_6fish_WBSC_*.h5` which is created by `Analysis/Weight_Map_Distance.jl`. For conveniance we also provide this file as a `DataDeps`.


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
