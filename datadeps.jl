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

url = "https://zenodo.org/records/19066472/files/"

register(DataDep(
  "Data_WBSC",
  "Whole-brain single-neuron datasets hosted on Zenodo.",
  url * "WBSC.zip",
  (md5, "4d3fa64556ca55a4d7e706d3f9037140");
  post_fetch_method=unpack
))

register(DataDep(
  "Data_Vox",
  "Whole-brain voxelized datasets hosted on Zenodo.",
  [
    url * "Voxelized.zip",
    url * "Voxelgrids.zip",
  ],
  [
    (md5, "3f1819cf983d3a89aaa96bd5b82d5328"),
    (md5, "2a208240c9983c10d187c4a7b62465d0"),
  ];
  post_fetch_method=unpack
))

register(DataDep(
  "RBMs_Vox",
  "Whole-brain voxelized RBMs hosted on Zenodo.",
  [
    url * "vRBMr_multivoxelized_6fish_20.0vox_M40_l2l10.1.h5",
    url * "CrossValidation_VOX.zip",
    url * "Repeats_VOX.zip",
    url * "Dropped.zip",
  ],
  [
    (md5, "5d21dd6cad889b139e02fcef490b86e3"),
    (md5, "c924805c59ffa74a3b3c2a984eeab9e5"),
    (md5, "2cd8c49fe6ac51ae8aa76277993fa121"),
    (md5, "f03104e1a05002de7517d62e4e1fe315"),
  ];
  post_fetch_method=[
    identity,
    unpack,
    unpack,
    unpack,
  ]
))

register(DataDep(
  "RBMs_WBSC",
  "Whole-brain single-cell RBMs hosted on Zenodo.",
  [
    url * "bRBMs.zip",
    url * "biRBMs.zip",
    url * "biRBMs_before_training.zip",
    url * "Repeats_WBSC.zip",
  ],
  [
    (md5, "012594ab78442f17539ae64ba3d18a58"),
    (md5, "b7aafbf981bc0d78bf0c80d157573cd4"),
    (md5, "bf29aca31e95437beae268c9c6e62ac0"),
    (md5, "abbb56c1d622cba79660becb842d9305"),
  ];
  post_fetch_method=unpack
))

register(DataDep(
  "Misc",
  "Miscellaneous precomputed files hosted on Zenodo.",
  [
    url * "WeightDist_6fish_WBSC_M100_l10.02_l2l10_sigma4_epsilon1.0e-5.h5",
    url * "DeepFakeFreeEnergy_6fish_WBSC_M100_l10.02_l2l10.h5",
    url * "DeepFakeTransferMethods_6fish_WBSC_M100_l10.02_l2l10.h5",
    url * "DeepFakeStats_6fish_WBSC_M100_l10.02_l2l10.h5",
    url * "DeepFakeActivityDistance_6fish_WBSC_M100_l10.02_l2l10.h5",
  ],
  [
    (md5, "5e942a1c19861fd44cc072675d866176"),
    (md5, "690513f0abb2ff7e63174cda77c8ebd8"),
    (md5, "bc5a347f87bcc8618ed39d6de21963a3"),
    (md5, "b0ca8b4ecba9f59905b1b58480e01a74"),
    (md5, "81c7365362701b98eb738dd493a5a8e0"),
  ];
  post_fetch_method=[
    identity,
    identity,
    identity,
    identity,
    identity,
  ]
))

@info "The data will be downloaded automatically when needed to: $(ENV["DATADEPS_LOAD_PATH"])"

