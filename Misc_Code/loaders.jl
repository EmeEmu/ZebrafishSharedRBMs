using Glob

include(joinpath(dirname(Base.current_project()), "datadeps.jl"))
using DataDeps: @datadep_str

# function load_models(dir::String, name::String)
#   paths = glob("*$name*h5", dir)
#   return paths
# end
#
# function load_model(dir::String, name::String)
#   path = load_models(dir, name)
#   @assert length(path) == 1 "multiple models `$name` were found in $dir :\n$path"
#   return path[1]
# end

function load_dataWBSCs(name::String)
  dir = @datadep_str("Data_WBSC/WBSC")
  paths = glob("DATA_*$name*h5", dir)
  return paths
end
function load_dataWBSC(name::String)
  path = load_dataWBSCs(name)
  @assert length(path) == 1 "multiple datas `$name` were found in\n$path"
  return path[1]
end

function load_dataVOXs(name::String, vsize::Float64)
  dir = @datadep_str("Data_Vox/Voxelized")
  paths = glob("VOX$(vsize)*$name*h5", dir)
  return paths
end
function load_dataVOX(name::String, vsize::Float64)
  path = load_dataVOXs(name, vsize::Float64)
  @assert length(path) == 1 "multiple datas `$name` were found in \n$path"
  return path[1]
end

function load_voxgrids(vsize::Float64)
  dir = @datadep_str("Data_Vox/Voxelgrids")
  paths = glob("VOXgrid*vs$(vsize)*h5", dir)
  return paths
end
function load_voxgrid(vsize::Float64)
  path = load_voxgrids(vsize::Float64)
  @assert length(path) == 1 "multiple grids `$vsize` were found in \n$path"
  return path[1]
end


function load_voxRBMs(subdir::AbstractString, name::String)
  if !isempty(subdir)
    ll = "RBMs_Vox/" * subdir
  else
    ll = "RBMs_Vox"
  end
  dir = @datadep_str(ll)
  paths = glob("*$name*h5", dir)
  return paths
end
function load_voxRBM(subdir::AbstractString, name::String)
  path = load_voxRBMs(subdir, name)
  @assert length(path) == 1 "multiple models `$name` were found in $subdir :\n$path"
  return path[1]
end

function load_wbscRBMs(subdir::AbstractString, name::String)
  dir = @datadep_str("RBMs_WBSC/" * subdir)
  paths = glob("*$name*h5*", dir)
  return paths
end
function load_wbscRBM(subdir::AbstractString, name::String)
  path = load_wbscRBMs(subdir, name)
  @assert length(path) == 1 "multiple models `$name` were found in $subdir :\n$path"
  return path[1]
end

function load_miscs(name::String)
  dir = @datadep_str("Misc")
  paths = glob("$name*", dir)
  return paths
end
function load_misc(name::String)
  path = load_miscs(name)
  @assert length(path) == 1 "multiple datas `$name` were found in\n$path"
  return path[1]
end

base_path = joinpath(dirname(Base.current_project()), "DataAndModels")
@assert isdir(base_path)
DATA_WBSC = joinpath(base_path, "Data_WBSC", "WBSC")
DATA_VOX = joinpath(base_path, "Data_Vox", "Voxelized")
DATA_VOXGRID = joinpath(base_path, "Data_Vox", "Voxelgrids")
RBM_VOX = joinpath(base_path, "RBMs_Vox")
RBM_VOXREPEAT = joinpath(base_path, "RBMs_Vox", "Repeats")
RBM_VOXCROSSVAL = joinpath(base_path, "RBMs_Vox", "CrossValidation")
RBM_WBSC_bRBM = joinpath(base_path, "RBMs_WBSC", "bRBMs")
RBM_WBSC_biRBM = joinpath(base_path, "RBMs_WBSC", "biRBMs")
RBM_WBSC_biRBMbefore = joinpath(base_path, "RBMs_WBSC", "biRBMs_before_training")
RBM_WBSC_REPEAT = joinpath(base_path, "RBMs_WBSC", "Repeats")
MISC = joinpath(base_path, "Misc")
