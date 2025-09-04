# use with `include("datadeps.jl")` in your main script, 
# then call `datadeps("vids1")` to get the path to the data
using DataDeps
using MD5

ENV["DATADEPS_LOAD_PATH"] = joinpath(dirname(Base.current_project()), "DataAndModels/")
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
ENV["DATADEPS_NO_STANDARD_LOAD_PATH"] = true

if !isdir(ENV["DATADEPS_LOAD_PATH"])
  mkdir(ENV["DATADEPS_LOAD_PATH"])
end


register(DataDep(
  "Data_WBSC",
  "Whole-brain single-neuron datasets hosted on Zenodo.",
  "http://134.157.132.30:8000/Data_WBSC/WBSC.zip",
  (md5, "d55b787daa491a0f7da44e0c234e16e4");
  post_fetch_method=unpack
))

register(DataDep(
  "Data_Vox",
  "Whole-brain voxelized datasets hosted on Zenodo.",
  [
    "http://134.157.132.30:8000/Data_Vox/Voxelized.zip",
    "http://134.157.132.30:8000/Data_Vox/Voxelgrids.zip",
  ],
  [
    (md5, "76315cdfa1f0e6d5f91b1333eddec048"),
    (md5, "b2d2c22e22d06cd97c2b62848f14a83c"),
  ];
  post_fetch_method=unpack
))

@info "The data will be downloaded automatically when needed to: $(ENV["DATADEPS_LOAD_PATH"])"

