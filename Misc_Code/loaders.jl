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

function load_datas(subdir::String, name::String)
  dir = @datadep_str("Data/" * subdir)
  paths = glob("*$name*h5", dir)
  return paths
end

function load_data(subdir::String, name::String)
  path = load_datas(subdir, name)
  @assert length(path) == 1 "multiple datas `$name` were found in $subdir :\n$path"
  return path[1]
end



base_path = joinpath(dirname(Base.current_project()), "DataAndModels")
@assert isdir(base_path)
DATA = joinpath(base_path, "Data")
DATA_WBSC = joinpath(DATA, "WBSC")
DATA_VOX = joinpath(DATA, "Voxelized")
