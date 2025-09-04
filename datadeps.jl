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

register(DataDep(
  "RBMs_Vox",
  "Whole-brain voxelized RBMs hosted on Zenodo.",
  [
    "http://134.157.132.30:8000/RBMs_Vox/vRBMr_multivoxelized_6fish_20.0vox_M40_l2l10.1.h5",
    "http://134.157.132.30:8000/RBMs_Vox/CrossValidation.zip",
    "http://134.157.132.30:8000/RBMs_Vox/Repeats.zip",
  ],
  [
    (md5, "5d21dd6cad889b139e02fcef490b86e3"),
    (md5, "05a155da8a122537f35c622ff9dc2ef3"),
    (md5, "643cab094e17564f28b0acf09b2ae73b"),
  ];
  post_fetch_method=unpack
))

@info "The data will be downloaded automatically when needed to: $(ENV["DATADEPS_LOAD_PATH"])"

