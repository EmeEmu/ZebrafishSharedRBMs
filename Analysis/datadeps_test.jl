### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ ae499992-896f-11f0-2889-d3a6f356213c
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ b2c5986b-9944-4430-a1be-c32ec9aedce8
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ dc7a2126-cb59-4a02-b2ad-846ef2808789


# ╔═╡ 84d7b805-0bd8-423b-869e-0305685c83c8


# ╔═╡ abc50cf8-542c-4387-8e1f-c26c0d5d8f6e


# ╔═╡ 228f6750-5470-45b3-9fc4-f56155acf36a
LOAD.load_dataWBSCs("Kinmont")

# ╔═╡ aed3e83d-3060-48fc-a109-2ed7b69d313b
LOAD.load_dataWBSC("Marianne")

# ╔═╡ ba8db32e-6519-43b2-899d-f9dbc9f7d736
LOAD.DATA_WBSC

# ╔═╡ b414ff6d-2f73-466b-abda-ae3d63e070a6


# ╔═╡ ca7cf6c3-f38a-44ca-8dd6-d2b99412144a


# ╔═╡ 2ad87e2e-580a-43db-a8c1-087eda8a993e


# ╔═╡ 470a23ff-490b-4fa7-ae23-5b06583c16d7
LOAD.DATA_VOX

# ╔═╡ 52c36b4c-7d73-4033-a6bf-0c1603fd5e5f
LOAD.load_dataVOX("Marianne", 10.)

# ╔═╡ 6b37b423-6c3e-4c27-97b3-ee07afc701b3
LOAD.DATA_VOXGRID

# ╔═╡ 745720c7-8c7d-44f7-beda-42846229e880
LOAD.load_voxgrid(10.0)

# ╔═╡ d314a62a-f903-42c7-86d1-3355656949f9


# ╔═╡ 6636b7b2-88bf-48c4-9d3c-649b8c80ffdd


# ╔═╡ 12334db9-720e-4f4a-ac4c-2cb8bb8740c6


# ╔═╡ c8d704ae-f2a4-417e-ba44-03a84bffc704
LOAD.RBM_VOX

# ╔═╡ b17edca6-1c41-42a5-adf5-cf95abe39466
LOAD.RBM_VOXCROSSVAL

# ╔═╡ 61150cfb-8e88-40e9-9c37-f944736fc9ad
LOAD.RBM_VOXREPEAT

# ╔═╡ 584903d5-b72c-4814-99c4-d0315081cb65
LOAD.load_voxRBM("", "multivoxelized")

# ╔═╡ 7c705596-1377-4cb0-97c0-fdf4f9568644
LOAD.load_voxRBMs("Repeats", "vRBMr_multivoxelized_6fish_20.0vox_M*_l2l1*_rep")

# ╔═╡ 886c265a-cbee-4977-87d1-ac1b09e8dbed
LOAD.load_voxRBMs("CrossValidation", "M40_l2l10.1")

# ╔═╡ e4f32e01-88fd-4eb5-abb6-72f723abb374
begin
	fish="Marianne"
	v = 20.
	M = 40
	λ = 0.01
	LOAD.load_voxRBMs("CrossValidation", "$(fish)*_VOX$(v)_M$(M)_l2l1$(λ)_rep")
end

# ╔═╡ 626e5cf7-6e3b-4a68-a471-46e10c3b6566


# ╔═╡ 2cdfa8fb-452e-49eb-a185-976f241008b3


# ╔═╡ b19d2ff7-a5aa-439c-bee2-966ecd6bfe5e


# ╔═╡ 5dc0d728-d692-422f-80df-6aff1266df9a
LOAD.RBM_WBSC_biRBM

# ╔═╡ 58091b35-1f44-4a27-98c0-dbed5d8b0e7e
LOAD.RBM_WBSC_biRBMbefore

# ╔═╡ 7baf5d29-cc76-4d3c-8a52-0f81b2e7ea42
LOAD.RBM_WBSC_bRBM

# ╔═╡ 65305c90-8319-4ab1-9a34-e6d736cd6de7
LOAD.RBM_WBSC_REPEAT

# ╔═╡ 33e4c155-8166-4ce3-9377-7442eff7601a
LOAD.load_wbscRBM("bRBMs", "Marianne")

# ╔═╡ 0ea98b4a-e699-4c1d-9633-761394b87be3
LOAD.load_wbscRBM("biRBMs", "Marianne_FROM_Silvestre")

# ╔═╡ 3b2f3192-a775-4f8c-aa11-50c650b51ee0
LOAD.load_wbscRBM("biRBMs_before_training", "Marianne_FROM_Silvestre")

# ╔═╡ 3e6d5c24-21bf-4fb3-aeab-0007902c37e8
LOAD.load_wbscRBMs("Repeats", "Marianne")

# ╔═╡ fc87b35d-ff6c-48d0-95ef-2def19cb49db


# ╔═╡ 62557857-b821-4899-8740-5145591b9647


# ╔═╡ aeda2cb9-d86d-4e86-a380-905dec4da2dd


# ╔═╡ 56e84401-cba3-4f44-b5df-e108e546ca52
LOAD.MISC

# ╔═╡ ce55ab4e-73a7-4de9-bd88-4fb4ab61758c
LOAD.load_misc("WeightDist")

# ╔═╡ c2b53450-021e-4698-8361-773574db528c
LOAD.load_misc("DeepFakeFreeEnergy")

# ╔═╡ 33efb392-b824-42cc-a6fb-f442d92c4870
LOAD.load_misc("DeepFakeTransferMethods")

# ╔═╡ 1b901042-e7ac-48c9-880f-cef0dc668ee9
LOAD.load_misc("DeepFakeStats")

# ╔═╡ 34a91110-6921-4d49-8eeb-ddc264f31e09
LOAD.load_misc("DeepFakeActivityDistance")

# ╔═╡ Cell order:
# ╠═ae499992-896f-11f0-2889-d3a6f356213c
# ╠═b2c5986b-9944-4430-a1be-c32ec9aedce8
# ╠═dc7a2126-cb59-4a02-b2ad-846ef2808789
# ╠═84d7b805-0bd8-423b-869e-0305685c83c8
# ╠═abc50cf8-542c-4387-8e1f-c26c0d5d8f6e
# ╠═228f6750-5470-45b3-9fc4-f56155acf36a
# ╠═aed3e83d-3060-48fc-a109-2ed7b69d313b
# ╠═ba8db32e-6519-43b2-899d-f9dbc9f7d736
# ╠═b414ff6d-2f73-466b-abda-ae3d63e070a6
# ╠═ca7cf6c3-f38a-44ca-8dd6-d2b99412144a
# ╠═2ad87e2e-580a-43db-a8c1-087eda8a993e
# ╠═470a23ff-490b-4fa7-ae23-5b06583c16d7
# ╠═52c36b4c-7d73-4033-a6bf-0c1603fd5e5f
# ╠═6b37b423-6c3e-4c27-97b3-ee07afc701b3
# ╠═745720c7-8c7d-44f7-beda-42846229e880
# ╠═d314a62a-f903-42c7-86d1-3355656949f9
# ╠═6636b7b2-88bf-48c4-9d3c-649b8c80ffdd
# ╠═12334db9-720e-4f4a-ac4c-2cb8bb8740c6
# ╠═c8d704ae-f2a4-417e-ba44-03a84bffc704
# ╠═b17edca6-1c41-42a5-adf5-cf95abe39466
# ╠═61150cfb-8e88-40e9-9c37-f944736fc9ad
# ╠═584903d5-b72c-4814-99c4-d0315081cb65
# ╠═7c705596-1377-4cb0-97c0-fdf4f9568644
# ╠═886c265a-cbee-4977-87d1-ac1b09e8dbed
# ╠═e4f32e01-88fd-4eb5-abb6-72f723abb374
# ╠═626e5cf7-6e3b-4a68-a471-46e10c3b6566
# ╠═2cdfa8fb-452e-49eb-a185-976f241008b3
# ╠═b19d2ff7-a5aa-439c-bee2-966ecd6bfe5e
# ╠═5dc0d728-d692-422f-80df-6aff1266df9a
# ╠═58091b35-1f44-4a27-98c0-dbed5d8b0e7e
# ╠═7baf5d29-cc76-4d3c-8a52-0f81b2e7ea42
# ╠═65305c90-8319-4ab1-9a34-e6d736cd6de7
# ╠═33e4c155-8166-4ce3-9377-7442eff7601a
# ╠═0ea98b4a-e699-4c1d-9633-761394b87be3
# ╠═3b2f3192-a775-4f8c-aa11-50c650b51ee0
# ╠═3e6d5c24-21bf-4fb3-aeab-0007902c37e8
# ╠═fc87b35d-ff6c-48d0-95ef-2def19cb49db
# ╠═62557857-b821-4899-8740-5145591b9647
# ╠═aeda2cb9-d86d-4e86-a380-905dec4da2dd
# ╠═56e84401-cba3-4f44-b5df-e108e546ca52
# ╠═ce55ab4e-73a7-4de9-bd88-4fb4ab61758c
# ╠═c2b53450-021e-4698-8361-773574db528c
# ╠═33efb392-b824-42cc-a6fb-f442d92c4870
# ╠═1b901042-e7ac-48c9-880f-cef0dc668ee9
# ╠═34a91110-6921-4d49-8eeb-ddc264f31e09
