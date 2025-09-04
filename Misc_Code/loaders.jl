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



base_path = joinpath(dirname(Base.current_project()), "DataAndModels")
@assert isdir(base_path)
DATA_WBSC = joinpath(base_path, "Data_WBSC", "WBSC")
DATA_VOX = joinpath(base_path, "Data_Vox", "Voxelized")
DATA_VOXGRID = joinpath(base_path, "Data_Vox", "Voxelgrids")
