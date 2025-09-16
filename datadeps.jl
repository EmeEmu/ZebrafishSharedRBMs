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

url = "https://zenodo.org/records/17121390/files/"

register(DataDep(
  "Data_WBSC",
  "Whole-brain single-neuron datasets hosted on Zenodo.",
  url*"WBSC.zip",
  (md5, "d55b787daa491a0f7da44e0c234e16e4");
  post_fetch_method=unpack
))

register(DataDep(
  "Data_Vox",
  "Whole-brain voxelized datasets hosted on Zenodo.",
  [
    url*"Voxelized.zip",
    url*"Voxelgrids.zip",
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
    url*"vRBMr_multivoxelized_6fish_20.0vox_M40_l2l10.1.h5",
    url*"CrossValidation_VOX.zip",
    url*"Repeats_VOX.zip",
  ],
  [
    (md5, "5d21dd6cad889b139e02fcef490b86e3"),
    (md5, "05a155da8a122537f35c622ff9dc2ef3"),
    (md5, "643cab094e17564f28b0acf09b2ae73b"),
  ];
  post_fetch_method=[
    identity,
    unpack,
    unpack,
  ]
))

register(DataDep(
  "RBMs_WBSC",
  "Whole-brain single-cell RBMs hosted on Zenodo.",
  [
    url*"bRBMs.zip",
    url*"biRBMs.zip",
    url*"biRBMs_before_training.zip",
    url*"Repeats_WBSC.zip",
  ],
  [
    (md5, "47a9687d41a5a0e001a746e08e7a9c25"),
    (md5, "65ceb9e691a41d8e99cbcfe063ac9f0d"),
    (md5, "3b9572000ea75e2b6467ab6c500bce34"),
    (md5, "b2e98a47f71f27983e6358a83e1be7d6"),
  ];
  post_fetch_method=unpack
))

register(DataDep(
  "Misc",
  "Miscellaneous precomputed files hosted on Zenodo.",
  [
    url*"WeightDist_6fish_WBSC_M100_l10.02_l2l10_sigma4_epsilon1.0e-5.h5",
    url*"DeepFakeFreeEnergy_6fish_WBSC_M100_l10.02_l2l10.h5",
    url*"DeepFakeTransferMethods_6fish_WBSC_M100_l10.02_l2l10.h5",
    url*"DeepFakeStats_6fish_WBSC_M100_l10.02_l2l10.h5",
    url*"DeepFakeActivityDistance_6fish_WBSC_M100_l10.02_l2l10.h5",
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

