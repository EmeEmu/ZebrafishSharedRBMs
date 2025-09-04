### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 8696354c-300d-11f0-2524-8dc8a1538c50
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ 44d4f416-abec-4694-93bd-9621a65759ba
begin
	# loading modules
	using BrainRBMjulia
	using CairoMakie
	using HDF5
end

# ╔═╡ d352b999-017b-49a3-8268-929d858bd23a


# ╔═╡ 566c5c99-0a90-4802-8e91-4d4b920797ff
TableOfContents()

# ╔═╡ ddd30da4-160e-44c6-b0d8-db5da0744e08
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ f567b3bd-def6-4a10-980e-9190148d63dc


# ╔═╡ 216da4e8-9f63-47f6-883b-2948818af71b


# ╔═╡ ddadd427-de26-415f-90aa-ad4de6693358
md"""
!!! warn "Requires full dataset"
	This file cannot be run without the pre-processed data, which is not included in this repository. It available from the authors on request. Therefore the following cells have been disabled.

	However, the created datasets are available and will be downloaded automatically.
"""

# ╔═╡ ff95d061-6e8b-4427-86d0-fd9a293d6ce5
md"""
# 1. Fish and data
"""

# ╔═╡ e3d5d1c2-12f5-42a2-ac35-1564d7d5127e
FISH = [
	"MarianneHawkins",
	"EglantineHawkins",
	"SilvestreHawkins",
	"CarolinneKinmont",
	"HectorKinmont",
	"MichelKinmont",
];

# ╔═╡ 54a5c44f-db63-4268-b89f-ef7fefe7ff42
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	DATASETS = []
	for fish in FISH
		# datapath = LOAD.load_data(CONV.DATAPATH, fish)
		inpath = "/DATA1/Bahamut/Individuals/$(fish)/combined.h5"
		@assert isfile(inpath)
		println("$(fish) : \n\tfrom : $(inpath)")
		
		name = h5read(inpath, "Info/name")[1]
		@assert name == fish

		coords = permutedims(h5read(inpath, "Data/Brain/coordsBahamut2"))
		dff = Float64.(h5read(inpath, "Data/Brain/dff"))
		dff[isinf.(dff)] .= 0

		dataset = Data(replace(name, " "=>"").*"_DFF", dff[:,601:end], coords)
		push!(
			DATASETS, 
			dataset
		)
	end
end
  ╠═╡ =#

# ╔═╡ 7c8b0651-cafe-4b3d-b77a-5b28b945a95d
md"""
# 2. Voxelization
"""

# ╔═╡ 29377083-3a61-4fdc-862b-d7660ca9324d
voxel_size = 50.; # in μm

# ╔═╡ 3f4676e3-928b-48c0-9331-1d21cdbcb357
ϵ = 1.e-9; # to avoid zeros

# ╔═╡ 128457f5-46a9-427d-b287-127b0b4d381e
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
out_vox_path = joinpath(
	LOAD.DATA_VOXGRID, 
	"VOXgrid_$(length(FISH))fish_vs$(voxel_size).h5"
)
  ╠═╡ =#

# ╔═╡ cfef9330-e32f-4449-82b4-31e7fb876513
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	COORDS = Vector{Matrix{Float64}}(undef, length(DATASETS))
	ACTS = Vector{Matrix{Float64}}(undef, length(DATASETS))
	for (i,dataset) in enumerate(DATASETS)
		COORDS[i] = dataset.coords
		spikes = Float64.(permutedims(dataset.spikes)).+ϵ
		if size(spikes, 1) == 3750
			ACTS[i] = spikes[601:end,:]
		else
			ACTS[i] = spikes
		end
	end
end
  ╠═╡ =#

# ╔═╡ 9e1f9b8f-9eda-4dd5-960a-20e5653003b6
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
vox = VoxelGrid(
	COORDS,
	ACTS,
	voxsize=[voxel_size, voxel_size, voxel_size],
);
  ╠═╡ =#

# ╔═╡ 17a86345-c896-4d0b-8418-66785da52c32
#=╠═╡
md"""
Created Voxel grid :
- Number of fish : $(length(FISH))
- Voxel Size : $(voxel_size) μm
- Number of voxels : $(size(vox.voxel_composition,2))

→ saving at : $(out_vox_path)
"""
  ╠═╡ =#

# ╔═╡ a5048f0d-3bbb-4afa-8f34-bcea103c6df8
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
dump_voxel(
	out_vox_path,
	vox,
	comment="Voxelisation of $(join(FISH, ","))"
)
  ╠═╡ =#

# ╔═╡ bc635449-ca05-46fe-86a2-b821e10b1e69


# ╔═╡ a1ef33a3-1cd3-4f16-b665-abbecea4fb1c
md"""
# 3. Building Voxelized datasets
"""

# ╔═╡ 55d97a9e-9803-46da-ab27-2cbc6cd0f2d0
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
# addapted from https://stackoverflow.com/a/54300691
to_coords(a::AbstractArray{CartesianIndex{L}}) where L = permutedims(reshape(reinterpret(Int, a), (L, size(a)...)))
  ╠═╡ =#

# ╔═╡ f854b425-3844-4099-857c-678ed93bed07
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
for i in 1:length(FISH)
	name = split(DATASETS[i].name, "_")[1] .* "_VOX$(voxel_size)"
	outpath = joinpath(
		LOAD.DATA_Vox,
		"VOX$(voxel_size)_" .* split(DATASETS[i].name, "_")[1] .*".h5"
	)
	
	zact = zscore(vox.voxel_activities[i]')
	vcoords = to_coords(vox.goods)

	dataset = Data(name, zact, vcoords)
	dump_data(outpath, dataset, comment="Voxelized with $(join(FISH, ","))")
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═8696354c-300d-11f0-2524-8dc8a1538c50
# ╠═44d4f416-abec-4694-93bd-9621a65759ba
# ╠═d352b999-017b-49a3-8268-929d858bd23a
# ╠═566c5c99-0a90-4802-8e91-4d4b920797ff
# ╠═ddd30da4-160e-44c6-b0d8-db5da0744e08
# ╠═f567b3bd-def6-4a10-980e-9190148d63dc
# ╠═216da4e8-9f63-47f6-883b-2948818af71b
# ╟─ddadd427-de26-415f-90aa-ad4de6693358
# ╟─ff95d061-6e8b-4427-86d0-fd9a293d6ce5
# ╠═e3d5d1c2-12f5-42a2-ac35-1564d7d5127e
# ╠═54a5c44f-db63-4268-b89f-ef7fefe7ff42
# ╟─7c8b0651-cafe-4b3d-b77a-5b28b945a95d
# ╠═29377083-3a61-4fdc-862b-d7660ca9324d
# ╠═3f4676e3-928b-48c0-9331-1d21cdbcb357
# ╠═128457f5-46a9-427d-b287-127b0b4d381e
# ╠═cfef9330-e32f-4449-82b4-31e7fb876513
# ╠═9e1f9b8f-9eda-4dd5-960a-20e5653003b6
# ╟─17a86345-c896-4d0b-8418-66785da52c32
# ╠═a5048f0d-3bbb-4afa-8f34-bcea103c6df8
# ╠═bc635449-ca05-46fe-86a2-b821e10b1e69
# ╟─a1ef33a3-1cd3-4f16-b665-abbecea4fb1c
# ╠═55d97a9e-9803-46da-ab27-2cbc6cd0f2d0
# ╠═f854b425-3844-4099-857c-678ed93bed07
