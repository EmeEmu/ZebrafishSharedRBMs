### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ cf46e4a6-0fb2-11f0-3416-d5d2b4b5f546
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ e7f1ae5a-e0f1-4052-8b33-99e8ceab7ae9
begin
	# loading modules
	using BrainRBMjulia
	using CairoMakie
	using Random
	using Clustering
	using HDF5
end

# ╔═╡ f4cdf5e7-2251-4fce-a839-16f3b5e3fac2
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ 3626deb0-86fd-49e3-ba1f-c662d677626c
TableOfContents()

# ╔═╡ a9fbd251-95fb-45ce-bb83-348c272b6a2d


# ╔═╡ 9d25b59f-26f3-41f6-8424-a489afe195cf
md"""
# 1. Building datasets

Construct dataset structure and save to HDF5 files for later use.
"""

# ╔═╡ e069dd1a-545e-420f-ae85-be875d03dbb2
md"""
!!! warn "Requires full dataset"
	This section cannot be run without the pre-processed data, which is not included in this repository. It available from the authors on request. Therefore the following cell has been disabled.

	However, the created datasets are available and will be downloded automatically.
"""

# ╔═╡ 774e307e-832f-4186-93b5-43b9cc531f0b
fish_list = [
    "EglantineHawkins",
    "MarianneHawkins",
    "SilvestreHawkins",
    "CarolinneKinmont",
    "HectorKinmont",
    "MichelKinmont",
]

# ╔═╡ 44e1831b-9da3-4084-bee2-77d0df36aafb
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	for fish in fish_list
		outpath = LOAD.DATA_WBSC .* "/DATA_$(replace(fish, " "=>"")).h5"
		if isfile(outpath)
			continue
		end
		inpath = "/DATA1/Bahamut/Individuals/$(fish)/combined.h5"
		@assert isfile(inpath)
		println("$(fish) : \n\tfrom : $(inpath)\n\tto   : $(outpath)")

		name = h5read(inpath, "Info/name")[1]
		@assert name == fish

		coords = permutedims(h5read(inpath, "Data/Brain/coordsBahamut2"))
		#spikes = Bool.(permutedims(h5read(inpath, "Data/Brain/spikes")))
		spikes = Bool.(h5read(inpath, "Data/Brain/spikes"))
		println("\tcoords : $(size(coords)) | spikes : $(size(spikes))")
		
		dataset = Data(replace(name, " "=>"").*"_WBSC", spikes[:,601:end], coords)

		dump_data(
		    outpath,
		    dataset,
		    comment="Whole-Brain Single-Cell of $(name)."
		)
	end
end
  ╠═╡ =#

# ╔═╡ 728cfaf2-2d2b-49ac-bebc-15db05964942


# ╔═╡ 62f6a352-d3a2-469b-b2b9-9c1977821edf


# ╔═╡ 74822e14-0a9e-4a99-a200-071a5bc41dc9
md"""
# 2. Verifying datasets
"""

# ╔═╡ e2fa00ea-0859-4f8e-a911-cc7736fa723b
for fish in fish_list
	dataset = load_data(LOAD.load_dataWBSC(fish))
	println(dataset.name)
	println("\tspikes : $(size(dataset.spikes))")
	println("\tcoords : $(size(dataset.coords))")
end

# ╔═╡ 71490f93-3186-47dc-a7ff-42ce28f94ed6


# ╔═╡ 1c2c3153-ee38-4dcd-a443-8d7d04804a16
md"""
# 3. Showing activity
"""

# ╔═╡ dd588b40-7397-49dd-9d7c-877e7e65eea6
fish = fish_list[1]

# ╔═╡ 7ab97bb6-27dc-496c-9008-9a9c614f6799
dataset = load_data(LOAD.load_dataWBSC(fish))

# ╔═╡ fa012b40-109c-4348-8806-3db6caa71e0f
function corr_order(x)
	C = cor(x')
	replace!(C, NaN=>0)
	HCLUST = hclust(1 .- C, linkage=:ward, branchorder=:optimal)
	return HCLUST.order
end

# ╔═╡ befe6447-d060-4823-9d79-b0f79d73cc28
function ordered_raster_sample(spikes; n=100)
	N,T = size(spikes)
	inds = randperm(N)[1:n]
	x = spikes[inds,:]
	#C = cor(x')
	#replace!(C, NaN=>0)
	#HCLUST = hclust(1 .- C, linkage=:ward, branchorder=:optimal)
	#order = HCLUST.order
	order = corr_order(x)
	return x[order,:]
end

# ╔═╡ 4287c2d7-8032-4b56-9543-24a5d73c91c8


# ╔═╡ 91263eda-87de-4a93-80be-620981da039a
begin
	fig_act = Figure()
	Axis(fig_act[1,1])
	heatmap!(ordered_raster_sample(dataset.spikes, n=1000)', colormap=:inferno, colorrange=(0,1))
	fig_act
end

# ╔═╡ 3500e07c-12d7-4616-929d-656bcd20a846


# ╔═╡ 6ae766f6-b65a-418f-88b8-2937841cfe44


# ╔═╡ dc92d6a0-1ce1-4282-8c43-657b2e8a9d70


# ╔═╡ Cell order:
# ╠═cf46e4a6-0fb2-11f0-3416-d5d2b4b5f546
# ╠═e7f1ae5a-e0f1-4052-8b33-99e8ceab7ae9
# ╠═f4cdf5e7-2251-4fce-a839-16f3b5e3fac2
# ╠═3626deb0-86fd-49e3-ba1f-c662d677626c
# ╠═a9fbd251-95fb-45ce-bb83-348c272b6a2d
# ╟─9d25b59f-26f3-41f6-8424-a489afe195cf
# ╟─e069dd1a-545e-420f-ae85-be875d03dbb2
# ╠═774e307e-832f-4186-93b5-43b9cc531f0b
# ╠═44e1831b-9da3-4084-bee2-77d0df36aafb
# ╠═728cfaf2-2d2b-49ac-bebc-15db05964942
# ╠═62f6a352-d3a2-469b-b2b9-9c1977821edf
# ╟─74822e14-0a9e-4a99-a200-071a5bc41dc9
# ╠═e2fa00ea-0859-4f8e-a911-cc7736fa723b
# ╠═71490f93-3186-47dc-a7ff-42ce28f94ed6
# ╟─1c2c3153-ee38-4dcd-a443-8d7d04804a16
# ╠═dd588b40-7397-49dd-9d7c-877e7e65eea6
# ╠═7ab97bb6-27dc-496c-9008-9a9c614f6799
# ╠═fa012b40-109c-4348-8806-3db6caa71e0f
# ╠═befe6447-d060-4823-9d79-b0f79d73cc28
# ╠═4287c2d7-8032-4b56-9543-24a5d73c91c8
# ╠═91263eda-87de-4a93-80be-620981da039a
# ╠═3500e07c-12d7-4616-929d-656bcd20a846
# ╠═6ae766f6-b65a-418f-88b8-2937841cfe44
# ╠═dc92d6a0-1ce1-4282-8c43-657b2e8a9d70
