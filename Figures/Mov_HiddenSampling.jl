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
	include(joinpath(CONV.UTILSPATH, "fig_saving.jl"))
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

# ╔═╡ bdde0e03-82b3-4a68-a525-e8d84e838d3c
md"example fish : $(@bind EXfish Select(FISH, default=FISH[1]))"

# ╔═╡ 7eced897-2518-467b-b5ec-6ec44a489fcb
dataset = load_data(LOAD.load_dataWBSC(EXfish))

# ╔═╡ acfa23b8-3d04-43ad-b0bd-f84de92c1449
base_mod = "*_WBSC_M100_l10.02_l2l10";

# ╔═╡ f9f8f7d0-0313-453d-9306-54ab8659b62e
rbm, _,_,_,_,_ = load_brainRBM(LOAD.load_wbscRBM("bRBMs", "bRBM*$(EXfish)$(base_mod)"))

# ╔═╡ d74ede45-f5d3-4cdc-a1c7-bad27c27a3c6


# ╔═╡ 5ff468c0-da80-40bb-b16d-c401566e1502
# t_exs = [510, 1200, 1500, 2550]
t_exs = [510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620]

# ╔═╡ e4e4ff1c-78fa-4f31-8598-abe75f49ef70
v_ex = hcat([repeat(dataset.spikes[:, t], 1,40) for t in t_exs]...);

# ╔═╡ 5007000a-a05e-4917-b6d2-04211b9d236c
m_vhv_ex = mean_v_from_h(rbm, mean_h_from_v(rbm, v_ex));

# ╔═╡ d75eb5b0-0cb7-4916-b282-c8f5b23d5bc7
vhv_ex = sample_v_from_h(rbm, mean_h_from_v(rbm, v_ex));

# ╔═╡ c18722b3-87dd-489a-9da2-7fa87de2d590


# ╔═╡ eaa56bd7-883d-4312-8e64-d7d23796e085


# ╔═╡ 44410d71-a119-4ded-b998-d4b5203dc823


# ╔═╡ 8ae3e452-b2c1-40d0-b59e-cb60321ed56e


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

# ╔═╡ b5fb1336-ec2e-4f12-927d-cb6993eb05e2
neurons_params_cont = (
	edgecolor=(:white, 0.5), 
	edgewidth=0.1,
	radius=4,
	cmap=cmapa, 
	range=(0,+1),
);

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

# ╔═╡ 9b4db997-682c-48d0-b495-dbc555f6be0b
begin
	t_single = Observable(100)
	fig_mouv_single = Figure(size=dfsize().*(2,1.1))
	ax_mouv_single = Axis(fig_mouv_single[1,1]; scatter_axis_args...)

	h_bin = neuron2dscatter!(
		ax_mouv_single,
		dataset.coords[:,1], dataset.coords[:,2],
		@lift(v_ex[:,$t_single]);
		neurons_params_bin...
	)

	neuron2dscatter!(
		ax_mouv_single,
		dataset.coords[:,1] .+ 500, dataset.coords[:,2],
		@lift(vhv_ex[:,$t_single]);
		neurons_params_bin...
	)

	h_cont = neuron2dscatter!(
		ax_mouv_single,
		dataset.coords[:,1] .+ 1000, dataset.coords[:,2],
		@lift(m_vhv_ex[:,$t_single]);
		neurons_params_cont...
	)

	cbar_bin = Colorbar(
		fig_mouv_single, 
		h_bin,
		height=Relative(0.3),
		flipaxis=false,
		label="Spikes",
		valign=:bottom, halign=:left,
		alignmode = Outside(10),
		bbox=ax_mouv_single.scene.viewport,
		# ticks=([0,2],["off","on"]),
	)
	cbar_cont = Colorbar(
		fig_mouv_single, 
		h_cont,
		height=Relative(0.3),
		flipaxis=true,
		label="𝔼[v]",
		alignmode = Outside(10),
		valign=:bottom, halign=:right,
		bbox=ax_mouv_single.scene.viewport,
		# ticks=([0,2],["off","on"]),
	)

	
	text!(ax_mouv_single,
		500, 10, 
		text="200μm", 
		align=(:center, :bottom),
	)
	lines!(ax_mouv_single,
		[400,600], [0,0], 
		color=:black,
	)

	text!(ax_mouv_single,
		250, 1000, 
		text="vₜ", 
		align=(:center, :bottom),
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		# font=Makie.current_default_theme().Axis.titlefont.val,
	)
	text!(ax_mouv_single,
		750, 1000, 
		text="v ~ P(v | P(h|vₜ))", 
		align=(:center, :bottom),
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		# font=Makie.current_default_theme().Axis.titlefont.val,
	)
	text!(ax_mouv_single,
		1250, 1000, 
		text="𝔼[v | 𝔼[h|vₜ]]", 
		align=(:center, :bottom),
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		# font=Makie.current_default_theme().Axis.titlefont.val,
	)


	xlims!(ax_mouv_single, -150, 1650)
	ylims!(ax_mouv_single, -10, 1065)
end

# ╔═╡ b27c0d9f-773e-496b-b277-0cb6936ba36b
display(fig_mouv_single)

# ╔═╡ 3efbb871-935f-49f9-8fe1-fb482705821a
record(
		fig_mouv_single, 
		replace(@figpath("HiddenSamples"), ".svg"=>".mp4"), 
		1:size(v_ex,2),
		framerate=out_framerate,
		format="mp4", profile="main", 
	) do time
		t_single[] = time
	end

# ╔═╡ fe271eb2-f481-4d5d-833a-1811b120428c


# ╔═╡ 7df2372f-8dbb-450c-9f03-54eaa285e7d8


# ╔═╡ 16e2b3e7-a81d-4fb1-856d-781b1a20f004


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
# ╟─bdde0e03-82b3-4a68-a525-e8d84e838d3c
# ╠═7eced897-2518-467b-b5ec-6ec44a489fcb
# ╠═acfa23b8-3d04-43ad-b0bd-f84de92c1449
# ╠═f9f8f7d0-0313-453d-9306-54ab8659b62e
# ╠═d74ede45-f5d3-4cdc-a1c7-bad27c27a3c6
# ╠═5ff468c0-da80-40bb-b16d-c401566e1502
# ╠═e4e4ff1c-78fa-4f31-8598-abe75f49ef70
# ╠═5007000a-a05e-4917-b6d2-04211b9d236c
# ╠═d75eb5b0-0cb7-4916-b282-c8f5b23d5bc7
# ╠═c18722b3-87dd-489a-9da2-7fa87de2d590
# ╠═eaa56bd7-883d-4312-8e64-d7d23796e085
# ╠═44410d71-a119-4ded-b998-d4b5203dc823
# ╠═8ae3e452-b2c1-40d0-b59e-cb60321ed56e
# ╟─1a6efd6a-219e-4de3-9425-e9f75ada7627
# ╟─9e3923eb-5d34-4a25-9499-c217c2650d7d
# ╠═25e0ce4c-1d97-4db4-8424-86b4f7f61789
# ╠═73d45550-016f-402a-be2b-ed34a44bee52
# ╠═cb5d1aa6-9f4a-4737-ad27-341ff563d353
# ╠═76d2b5dc-d85e-459b-8e06-58b8936ab1bd
# ╠═b5fb1336-ec2e-4f12-927d-cb6993eb05e2
# ╠═d25a5e14-196d-44cb-85d8-003d5c254a25
# ╠═311269ad-91df-41cc-ad41-b5a67ab92085
# ╠═496789ba-9471-4e24-961a-63973bf2b844
# ╠═e9778762-430a-4c69-aa46-539e235bfb5b
# ╠═996ec001-f19a-4425-a223-d60e33803d03
# ╠═b7460355-44cb-4d79-8cad-ca71e1c9e86e
# ╟─2116c647-4f2e-4b4a-9c48-98ad97d1ccfe
# ╠═9b4db997-682c-48d0-b495-dbc555f6be0b
# ╠═b27c0d9f-773e-496b-b277-0cb6936ba36b
# ╠═3efbb871-935f-49f9-8fe1-fb482705821a
# ╠═fe271eb2-f481-4d5d-833a-1811b120428c
# ╠═7df2372f-8dbb-450c-9f03-54eaa285e7d8
# ╠═16e2b3e7-a81d-4fb1-856d-781b1a20f004
