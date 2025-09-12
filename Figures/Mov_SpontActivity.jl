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

# ╔═╡ cbd368fa-3c73-418d-8868-b9870802c152
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ 35983245-e4e4-4136-8197-2622ed3b4417
begin
	using BrainRBMjulia
	using LinearAlgebra: diagind
	using HDF5
	
	using GLMakie
	using BrainRBMjulia: dfsize, quantile_range, neuron2dscatter!, cmap_aseismic, cmap_hardseismic, cmap_dff, cmap_Gbin, cmap_ainferno
	using ColorSchemes

	CONV = @ingredients("conventions.jl")
	include(joinpath(dirname(Base.current_project()), "Misc_Code", "fig_saving.jl"))

end

# ╔═╡ 73d45550-016f-402a-be2b-ed34a44bee52
using BrainRBMjulia: colorscheme_alpha_sigmoid

# ╔═╡ e0945b60-eb8d-11ef-3139-fd7b39966a81
md"""
# 0. Imports + Notebook Preparation
"""

# ╔═╡ 9036e933-7ee4-4951-be5e-ab759c5d6f5f
# ╠═╡ disabled = true
#=╠═╡
set_theme!(CONV.style_publication)
  ╠═╡ =#

# ╔═╡ e991f0ea-79c3-4957-a113-fedef70dd933
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ 678b24c2-1853-4603-94e6-d24d6ff78546
TableOfContents()

# ╔═╡ 8ebbc3ea-0c55-4c6c-916f-97062a72595c


# ╔═╡ 7730af66-38ad-4172-822f-8dbd68511aef
begin
	pt_px = 2
	ticklabel = 8
	axelabel = 10
	title = 12
	ticklenght = 3
	rasterize = 5
	
	style_publication_activitymovie = Theme(
	  size=(480, 540),
	  font="Arial", # ? 
	  fontsize=7 * pt_px,
	  backgroundcolor = :gray,

		
	  Axis=(
    	backgroundcolor=:grey,
	    rightspinevisible=false,
	    topspinevisible=false,
	    leftspinevisible=false,
	    bottomspinevisible=false,
		xticklabelsvisible=false,
		yticklabelsvisible=false,
		xticksvisible=false,
		yticksvisible=false,
	    # spinewidth=1 * pt_px,
	    titlefont=:bold,
	    titlesize=title * pt_px,
	    subtitlesize=(title - 1) * pt_px,
	    xgridvisible=false,
	    ygridvisible=false,
	    xlabelsize=axelabel * pt_px,
	    ylabelsize=axelabel * pt_px,
	    xtrimspine=false,
	    ytrimspine=false,
	    xticksize=ticklenght * pt_px,
	    yticksize=ticklenght * pt_px,
	    xticklabelsize=ticklabel * pt_px,
	    yticklabelsize=ticklabel * pt_px,
	  ),
	  Colorbar=(
	    ticksize=ticklenght * pt_px,
	    size=12,
	    labelsize=axelabel * pt_px,
	    ticklabelsize=ticklabel * pt_px,
	  ),
	  Scatter=(
	    markersize=5,
	    # rasterize=rasterize,
	  ),
	  Text=(
	    fontsize=ticklabel * pt_px,
	  ),
	  GLMakie=(
	    px_per_unit=2,
	    type="png",
	  ),
	)
	
	set_theme!(style_publication_activitymovie)
	
	GLMakie.activate!()
end

# ╔═╡ 677aa56e-d11c-4f6d-9292-6a45f1b3f463


# ╔═╡ 78a2bbcb-abac-4c40-936d-272f0c3ac976
md"""
# 0. Datasets
"""

# ╔═╡ 5d5f82ad-392a-4cbd-960c-c19012d0f212
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ 855f26eb-87dd-484e-8b82-98840285e5d6
md"Use numbers instead of fish names $(@bind numbered CheckBox(default=true))"

# ╔═╡ 6c5568dd-77df-448c-a80e-7706e4adc8b8
begin
	if numbered
		FISH_DISP = ["F$i" for i in 1:length(FISH)]
	else
		FISH_DISP = FISH
	end
end;

# ╔═╡ 7eced897-2518-467b-b5ec-6ec44a489fcb
dataset_paths = [LOAD.load_dataWBSC(fish) for fish in FISH]

# ╔═╡ 73c61d1c-e61f-4935-9068-3b1aed89eb34
datasets = [load_data(path) for path in dataset_paths]

# ╔═╡ acfa23b8-3d04-43ad-b0bd-f84de92c1449


# ╔═╡ d74ede45-f5d3-4cdc-a1c7-bad27c27a3c6


# ╔═╡ 1a6efd6a-219e-4de3-9425-e9f75ada7627
md"""
# 1. Videos
"""

# ╔═╡ 9e3923eb-5d34-4a25-9499-c217c2650d7d
md"""
## 1.1 Parameters
"""

# ╔═╡ 25e0ce4c-1d97-4db4-8424-86b4f7f61789
begin
	cycle_time = 0.4
	img_freq = 1/cycle_time
	out_framerate = 10
	speed_up = round(Int, out_framerate / img_freq)
end

# ╔═╡ cb5d1aa6-9f4a-4737-ad27-341ff563d353
begin
	cmap = ColorSchemes.amp
	cmapa = colorscheme_alpha_sigmoid(
		cmap,
		0.05,
		0.25,
		0.975
	)
end

# ╔═╡ 76d2b5dc-d85e-459b-8e06-58b8936ab1bd
begin
	c_off = cmapa[1]
	c_on = cmapa[round(Int, 1*length(cmapa.colors)/1.4)]
	cmap_OnOff = Makie.Categorical([c_off, c_on])
end

# ╔═╡ d25a5e14-196d-44cb-85d8-003d5c254a25
neurons_params_bin = (
	edgecolor=(:white, 0.5), 
	edgewidth=0.1,
	radius=4,
	cmap=cmap_OnOff, 
	# cmap=cmap_OnOff,
	range=(0,1),
);

# ╔═╡ 311269ad-91df-41cc-ad41-b5a67ab92085
begin
	scatter_axis_args = (
	    aspect=DataAspect(),
	    # bottomspinevisible=true, leftspinevisible=true,
	    # xticklabelsvisible=true, yticklabelsvisible=true,
	    # xticksvisible=true, yticksvisible=true,
	)
	colorbar_args = (
		flipaxis=true, vertical=false,
	)
end

# ╔═╡ 496789ba-9471-4e24-961a-63973bf2b844


# ╔═╡ e9778762-430a-4c69-aa46-539e235bfb5b


# ╔═╡ 996ec001-f19a-4425-a223-d60e33803d03


# ╔═╡ b7460355-44cb-4d79-8cad-ca71e1c9e86e


# ╔═╡ 2116c647-4f2e-4b4a-9c48-98ad97d1ccfe
md"""
## 1.1 Single Fish
"""

# ╔═╡ 8a445c6a-54b6-4666-ad9d-2169bbdfc197
fish = datasets[1]

# ╔═╡ 0778c69e-a95c-46a6-b97c-81d3eb3ae17d
t_single_start, t_single_stop = 600, 1000;

# ╔═╡ 9b4db997-682c-48d0-b495-dbc555f6be0b
begin
	t_single = Observable(t_single_start)
	fig_mouv_single = Figure(size=dfsize().*(1,1))
	ax_mouv_single = Axis(fig_mouv_single[1,1]; scatter_axis_args...)

	h_single = neuron2dscatter!(
		ax_mouv_single,
		fish.coords[:,1], fish.coords[:,2],
		@lift(fish.spikes[:,$t_single]);
		neurons_params_bin...
	)

	cbar_single = Colorbar(
		fig_mouv_single[1,0], 
		h_single,
		height=Relative(0.1),
		flipaxis=false,
		label="Spikes",
		# ticks=([0,2],["off","on"]),
	)
	cbar_single.ticks[] = ([0,1], ["off","on"])
	text!(ax_mouv_single,
		150, 10, 
		text="200μm", 
		align=(:center, :bottom),
	)
	lines!(ax_mouv_single,
		[50,250], [0,0], 
		color=:black,
	)
	text!(ax_mouv_single,
		50, 1000, 
		text="$(speed_up)×", 
		align=(:center, :bottom),
	)
end

# ╔═╡ b27c0d9f-773e-496b-b277-0cb6936ba36b
display(fig_mouv_single)

# ╔═╡ 3efbb871-935f-49f9-8fe1-fb482705821a
# ╠═╡ disabled = true
#=╠═╡
record(
		fig_mouv_single, 
		replace(@figpath("SingleFish"), ".svg"=>".mp4"), 
		t_single_start:t_single_stop,
		framerate=out_framerate,
		format="mp4", profile="main", 
	) do time
		t_single[] = time
	end
  ╠═╡ =#

# ╔═╡ 3f15dc46-7a6e-45cc-b512-bf0da9a3954c


# ╔═╡ 8d4294af-e617-4891-a8d5-681d39ee51f5
h_single.colormap

# ╔═╡ 9d9ae4b0-d80f-4361-8a5c-fba800a43cbe
md"""
## 1.1 Multi Fish
"""

# ╔═╡ 2f951c88-5653-4461-84a6-7d0492bf4f29
t_multi_start, t_multi_stop = 600, 1000;

# ╔═╡ 12f0513b-c775-446d-be85-125183292eed
begin
	t_multi = Observable(t_multi_start)
	fig_mouv_multi = Figure(size=dfsize().*(3,1))
	ax_mouv_multi = Axis(fig_mouv_multi[1,1]; scatter_axis_args...)

	for (f,fish) in enumerate(datasets)
		neuron2dscatter!(
			ax_mouv_multi,
			fish.coords[:,1] .+ (f-1)*500, fish.coords[:,2],
			@lift(fish.spikes[:,$t_multi]);
			neurons_params_bin...
		)
		text!(ax_mouv_multi,
			250 + (f-1)*500, 1000, 
			text=FISH_DISP[f], 
			align=(:center, :bottom),
			fontsize=Makie.current_default_theme().Axis.titlesize.val,
			# font=Makie.current_default_theme().Axis.titlefont.val,
		)
	end

	cbar_multi = Colorbar(
		fig_mouv_multi, 
		h_single,
		alignmode = Outside(10),
		height=Relative(0.15),
		flipaxis=true,
		label="Spikes",
		halign=:right, valign=:bottom,
		bbox=ax_mouv_multi.scene.viewport,
		# ticks=([0,2],["off","on"]),
	)
	cbar_multi.ticks[] = ([0,1], ["off","on"])
	text!(ax_mouv_multi,
		100, 10, 
		text="200μm", 
		align=(:center, :bottom),
	)
	lines!(ax_mouv_multi,
		[0,200], [0,0], 
		color=:black,
	)
	text!(ax_mouv_multi,
		50, 1000, 
		text="$(speed_up)×", 
		align=(:center, :bottom),
	)

	ylims!(ax_mouv_multi, -10, 1065)
	xlims!(ax_mouv_multi, -10, 3010)
end

# ╔═╡ 9a23bc1e-f8d8-4ce3-86ba-640c6c8dede8
display(fig_mouv_multi)

# ╔═╡ 2db1b8a8-d9cd-42e3-8b64-2a5226175869
record(
		fig_mouv_multi, 
		replace(@figpath("SingleMulti"), ".svg"=>".mp4"), 
		t_single_start:t_single_stop,
		framerate=out_framerate,
		format="mp4", profile="main", 
	) do time
		t_multi[] = time
	end

# ╔═╡ 16e2b3e7-a81d-4fb1-856d-781b1a20f004


# ╔═╡ 96477e08-e353-4c9b-9b45-33403c1ccfbd


# ╔═╡ 95cd0e4c-ffb9-4858-8f97-44681805b235


# ╔═╡ c9458240-37cd-4b37-86d7-6eea0cf0e12b


# ╔═╡ 243c2627-9c72-454a-90ba-e5d229afda13


# ╔═╡ Cell order:
# ╠═e0945b60-eb8d-11ef-3139-fd7b39966a81
# ╠═cbd368fa-3c73-418d-8868-b9870802c152
# ╠═35983245-e4e4-4136-8197-2622ed3b4417
# ╠═9036e933-7ee4-4951-be5e-ab759c5d6f5f
# ╠═e991f0ea-79c3-4957-a113-fedef70dd933
# ╠═678b24c2-1853-4603-94e6-d24d6ff78546
# ╠═8ebbc3ea-0c55-4c6c-916f-97062a72595c
# ╠═7730af66-38ad-4172-822f-8dbd68511aef
# ╠═677aa56e-d11c-4f6d-9292-6a45f1b3f463
# ╟─78a2bbcb-abac-4c40-936d-272f0c3ac976
# ╠═5d5f82ad-392a-4cbd-960c-c19012d0f212
# ╟─855f26eb-87dd-484e-8b82-98840285e5d6
# ╟─6c5568dd-77df-448c-a80e-7706e4adc8b8
# ╠═7eced897-2518-467b-b5ec-6ec44a489fcb
# ╠═73c61d1c-e61f-4935-9068-3b1aed89eb34
# ╠═acfa23b8-3d04-43ad-b0bd-f84de92c1449
# ╠═d74ede45-f5d3-4cdc-a1c7-bad27c27a3c6
# ╟─1a6efd6a-219e-4de3-9425-e9f75ada7627
# ╟─9e3923eb-5d34-4a25-9499-c217c2650d7d
# ╠═25e0ce4c-1d97-4db4-8424-86b4f7f61789
# ╠═73d45550-016f-402a-be2b-ed34a44bee52
# ╠═cb5d1aa6-9f4a-4737-ad27-341ff563d353
# ╠═76d2b5dc-d85e-459b-8e06-58b8936ab1bd
# ╠═d25a5e14-196d-44cb-85d8-003d5c254a25
# ╠═311269ad-91df-41cc-ad41-b5a67ab92085
# ╠═496789ba-9471-4e24-961a-63973bf2b844
# ╠═e9778762-430a-4c69-aa46-539e235bfb5b
# ╠═996ec001-f19a-4425-a223-d60e33803d03
# ╠═b7460355-44cb-4d79-8cad-ca71e1c9e86e
# ╟─2116c647-4f2e-4b4a-9c48-98ad97d1ccfe
# ╠═8a445c6a-54b6-4666-ad9d-2169bbdfc197
# ╠═0778c69e-a95c-46a6-b97c-81d3eb3ae17d
# ╠═9b4db997-682c-48d0-b495-dbc555f6be0b
# ╠═b27c0d9f-773e-496b-b277-0cb6936ba36b
# ╠═3efbb871-935f-49f9-8fe1-fb482705821a
# ╠═3f15dc46-7a6e-45cc-b512-bf0da9a3954c
# ╠═8d4294af-e617-4891-a8d5-681d39ee51f5
# ╟─9d9ae4b0-d80f-4361-8a5c-fba800a43cbe
# ╠═2f951c88-5653-4461-84a6-7d0492bf4f29
# ╠═12f0513b-c775-446d-be85-125183292eed
# ╠═9a23bc1e-f8d8-4ce3-86ba-640c6c8dede8
# ╠═2db1b8a8-d9cd-42e3-8b64-2a5226175869
# ╠═16e2b3e7-a81d-4fb1-856d-781b1a20f004
# ╠═96477e08-e353-4c9b-9b45-33403c1ccfbd
# ╠═95cd0e4c-ffb9-4858-8f97-44681805b235
# ╠═c9458240-37cd-4b37-86d7-6eea0cf0e12b
# ╠═243c2627-9c72-454a-90ba-e5d229afda13
