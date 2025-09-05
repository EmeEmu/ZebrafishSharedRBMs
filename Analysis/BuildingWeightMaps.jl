### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 23794d22-1470-11f0-2ebd-d7436d72feb0
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ 48f0b114-cac6-413c-ae03-0873e41b57e1
begin
	using BrainRBMjulia
	using CairoMakie
	using BrainRBMjulia: idplotter!
end

# ╔═╡ 940875ad-dcb6-46f8-8b7a-44c2997b8990
TableOfContents()

# ╔═╡ ed794151-e8c2-48e4-b845-4dae6895666d
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ 2c8c0b97-0c36-433b-872d-2b6a319781e1


# ╔═╡ aa8ca3e2-1fe9-4723-ad02-ce5b7c1fd6ac
SCALING = 2.0; # neurons coordinates will be divided by SCALING

# ╔═╡ e2527138-624f-4842-8270-c540a9188999
md"""
!!! warn "Warning"
	This notebook precomputes teacher weight maps. This computation is long (depending on your number of threads), and results in an additional disk usage of ~15GB per fish. To avoid accidental runs, the last cells have been disabled (enable them to lauch the computation).

	If you wish to free disk space once you no longer need the precomputed maps, you can run the julia script `Misc_Code/clean_weightmaps.jl`.
"""

# ╔═╡ 6b3fb164-85a8-4b1c-ba2f-c4170778a2b7
md"""
# 1. Building 3D Box around all neurons
"""

# ╔═╡ 5e9abead-3940-4710-abbc-6a6c81ee518e
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ 77d77515-6ed0-40b2-92b6-8826cf37829b
COORDS = vcat(
	[
		load_data(
			LOAD.load_dataWBSC(fish)
		).coords./SCALING for fish in FISH
	]...
)

# ╔═╡ 5c6fd864-280f-4481-95e1-ac91841b6bc4
bigBox = BoxAround(COORDS);

# ╔═╡ 9c5e578b-b30c-4490-a5dc-1fa5126ba76e
bigBox

# ╔═╡ f21a7b56-d54f-4cc0-944d-4ec26b13161b


# ╔═╡ e2481a0e-1e9d-4c35-b024-eef702331ed7
md"""
# 2. Creating Maps per fish
"""

# ╔═╡ 87e25586-cf45-4d24-8805-472d9b2594f3
md"""
## 2.1. Getting data
"""

# ╔═╡ cb704bc8-014e-453c-8769-ea68dc409d64
md"selected fish : $(@bind fish Select(FISH))"

# ╔═╡ 7243c09d-8bca-4be2-9ada-cc7e659513b3
rbm_path = LOAD.load_wbscRBM("bRBMs", fish)

# ╔═╡ a7a2a5d7-ed14-4065-a7aa-68833d8d21cb
rbm,_,_,_,_,_ = load_brainRBM(rbm_path);

# ╔═╡ a29073a7-8fb5-4db6-8e6c-531c5d08eb84
weights = rbm.w;

# ╔═╡ 6dcc91d3-6df9-4e05-91ab-babd41555b26
coords = load_data(LOAD.load_dataWBSC(fish)).coords;

# ╔═╡ 26bec058-9449-4647-a82c-76cc0411d1a0
@assert size(weights,1) == size(coords,1)

# ╔═╡ 9d312425-73ae-4ef0-8ea2-fb8839139315
md"""
## 2.2. Building maps
"""

# ╔═╡ 40880245-abc4-4f69-8503-0b6f49956261
# ╠═╡ disabled = true
#=╠═╡
maps = Maps(
	coords,
	bigBox,
	weights,
	scaling=SCALING,
	R=4,
	σ=4.,
	verbose=false,
);
  ╠═╡ =#

# ╔═╡ 59146427-6f1a-4f6a-af3f-82c10a249858
md"""
## 2.3. Checking Weight reconstruction from weights
"""

# ╔═╡ eb379d35-c061-4f3d-aa20-62ccdaa5a5e2
#=╠═╡
reconstr_weights = interpolation(maps, coords, verbose=false);
  ╠═╡ =#

# ╔═╡ 33cb8a06-cfae-4130-a8b2-5228d632b4fa
#=╠═╡
begin
	begin
		fig_reconsWeights = Figure()
		Axis(
			fig_reconsWeights[1,1], 
			xlabel=L"w^-_{T,\mu}", 
			ylabel=L"\beta_{\mu}w^+_{T,\mu}", 
			title="Reproducing weights"
		)
		idplotter!(weights, reconstr_weights)
		fig_reconsWeights
	end
end
  ╠═╡ =#

# ╔═╡ f7214e4f-bb66-4c56-b679-244d1dec0ab6
md"""
## 2.4. Dumping weight maps
"""

# ╔═╡ d0373414-6b9b-40d3-a41a-437090a6ee86
#=╠═╡
dump_maps(rbm_path, maps, "Weight Maps")
  ╠═╡ =#

# ╔═╡ 979284d7-49af-4335-b5db-c87e4080238d


# ╔═╡ Cell order:
# ╠═23794d22-1470-11f0-2ebd-d7436d72feb0
# ╠═48f0b114-cac6-413c-ae03-0873e41b57e1
# ╠═940875ad-dcb6-46f8-8b7a-44c2997b8990
# ╠═ed794151-e8c2-48e4-b845-4dae6895666d
# ╠═2c8c0b97-0c36-433b-872d-2b6a319781e1
# ╠═aa8ca3e2-1fe9-4723-ad02-ce5b7c1fd6ac
# ╟─e2527138-624f-4842-8270-c540a9188999
# ╟─6b3fb164-85a8-4b1c-ba2f-c4170778a2b7
# ╠═5e9abead-3940-4710-abbc-6a6c81ee518e
# ╠═77d77515-6ed0-40b2-92b6-8826cf37829b
# ╠═5c6fd864-280f-4481-95e1-ac91841b6bc4
# ╠═9c5e578b-b30c-4490-a5dc-1fa5126ba76e
# ╠═f21a7b56-d54f-4cc0-944d-4ec26b13161b
# ╟─e2481a0e-1e9d-4c35-b024-eef702331ed7
# ╟─87e25586-cf45-4d24-8805-472d9b2594f3
# ╟─cb704bc8-014e-453c-8769-ea68dc409d64
# ╠═7243c09d-8bca-4be2-9ada-cc7e659513b3
# ╠═a7a2a5d7-ed14-4065-a7aa-68833d8d21cb
# ╠═a29073a7-8fb5-4db6-8e6c-531c5d08eb84
# ╠═6dcc91d3-6df9-4e05-91ab-babd41555b26
# ╠═26bec058-9449-4647-a82c-76cc0411d1a0
# ╟─9d312425-73ae-4ef0-8ea2-fb8839139315
# ╠═40880245-abc4-4f69-8503-0b6f49956261
# ╟─59146427-6f1a-4f6a-af3f-82c10a249858
# ╠═eb379d35-c061-4f3d-aa20-62ccdaa5a5e2
# ╠═33cb8a06-cfae-4130-a8b2-5228d632b4fa
# ╟─f7214e4f-bb66-4c56-b679-244d1dec0ab6
# ╠═d0373414-6b9b-40d3-a41a-437090a6ee86
# ╠═979284d7-49af-4335-b5db-c87e4080238d
