# use with `include("datadeps.jl")` in your main script, 
# then call `datadeps("vids1")` to get the path to the data
using DataDeps
using MD5

ENV["DATADEPS_LOAD_PATH"] = joinpath(dirname(Base.current_project()), "DataAndModels/")
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
ENV["DATADEPS_NO_STANDARD_LOAD_PATH"] = true


register(DataDep(
  "Data",
  "Whole-brain, single-neuron and voxelized datasets hosted on Zenodo.",
  [
    "http://134.157.132.30:8000/Data/WBSC.zip",
    "http://134.157.132.30:8000/Data/Voxelized.zip",
  ],
  [
    (md5, "d55b787daa491a0f7da44e0c234e16e4"),
    (md5, "76315cdfa1f0e6d5f91b1333eddec048"),
  ];
  post_fetch_method=unpack
))

# register(DataDep(
#   "vids1",
#   "videos from source 1 from Zenodo.",
#   [
#     "https://zenodo.org/records/16886970/files/Movie1.mp4",
#     "https://zenodo.org/records/16886970/files/Movie2.mp4",
#   ],
#   [
#     (md5, "0337b86723c795b1bdd4dddb20f19f87"),
#     (md5, "b8f4b5714d2c40517c44b57e70e0126f"),
#   ];
#   # post_fetch_method=identity # (local_filepath)->Any
# ))
#
# register(DataDep(
#   "local_vids1",
#   "local copy videos from source 1 from Zenodo.",
#   "Movie1.mp4",
#   # "https://zenodo.org/records/16886970/files/Movie1.mp4",
#   (md5, "0337b86723c795b1bdd4dddb20f19f87");
#   # post_fetch_method=identity # (local_filepath)->Any
# ))
#
# register(DataDep(
#   "vids2",
#   "videos from source 2 from Zenodo.",
#   "https://zenodo.org/api/records/16886749/files-archive";
#   # (md5, "cd0011a329dccd7d887723ab94e3e19f");
#   post_fetch_method=unpack
# ))

@info "The data will be downloaded automatically when needed to: $(ENV["DATADEPS_LOAD_PATH"])"

