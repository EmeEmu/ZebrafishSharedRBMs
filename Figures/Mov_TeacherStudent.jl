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

# ╔═╡ d2053251-4612-4b20-b8a4-0e225825fbbf
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ 5c1e28c7-b9a3-4b50-8672-ae0202df866b
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

# ╔═╡ b4cbe70e-9413-4875-9e64-4dbc97b816f7
using BrainRBMjulia: colorscheme_alpha_sigmoid

# ╔═╡ 6188f2d4-3aea-11f0-0f7e-0fb793ee5f9f
md"""
# Imports + Notebook Preparation
"""

# ╔═╡ cfc4d607-fc48-4e4b-841e-40c5dc4e0ab6
TableOfContents()

# ╔═╡ 9f4df0ca-7366-4c3f-90c7-6a8d80f510be
# ╠═╡ disabled = true
#=╠═╡
set_theme!(CONV.style_publication)
  ╠═╡ =#

# ╔═╡ 2fef345f-1d56-4287-8afb-1c8e8e95f7dd
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ ca29d3d8-4e7c-479f-aaf8-7c37d5960063


# ╔═╡ 6688652f-925c-4eca-81ca-178abf033276
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

# ╔═╡ 13430c22-29c6-4489-ab89-6e07070bedb0


# ╔═╡ 0634840b-2470-42d1-9974-18b16a0549fb


# ╔═╡ 2e35c6aa-e6a0-4dc3-a447-c1776fdcc071
md"""
# 0. Fish and training base
"""

# ╔═╡ e9a155a7-5f98-4e72-b0b4-cfbfcc937bb3
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ 42e44427-d2c8-40b6-b9cc-38779bafa326
base_mod = "*_WBSC_M100_l10.02_l2l10";

# ╔═╡ b828f13c-2bac-4db3-b4c1-b9df3f975dfd
md"example teacher : $(@bind EXteacher Select(FISH, default=FISH[3]))"

# ╔═╡ d25cb057-402f-49c7-b5a7-ab43cb48da35
STUDENTS = [fish for fish in FISH if fish!=EXteacher];

# ╔═╡ f88c08fd-1c27-4429-8f46-40ae86b61db4
md"example student : $(@bind EXstudent Select(STUDENTS, default=STUDENTS[1]))"

# ╔═╡ d1cc31ba-876e-469a-a8e7-17759aed13e9


# ╔═╡ d7c71d33-9958-4486-abf9-71858bb16d49
dataT = load_data(LOAD.load_dataWBSC(EXteacher));

# ╔═╡ 3475cc75-dfc2-4e69-b508-197af4328335
dataS = load_data(LOAD.load_dataWBSC(EXstudent));

# ╔═╡ 42584857-a753-472a-ba72-18cfed2577de


# ╔═╡ 26748574-8755-4379-83b4-7d33cef86372
begin
	cycle_time = 0.4
	img_freq = 1/cycle_time
	out_framerate = 10
	speed_up = round(Int, out_framerate / img_freq)
end

# ╔═╡ 1f07e89f-06d8-4417-a86d-6643be5645f2
begin
	#cmap = ColorSchemes.amp
	cmap = ColorSchemes.algae
	cmapa = colorscheme_alpha_sigmoid(
		cmap,
		0.05,
		0.25,
		0.975
	)
end

# ╔═╡ f5742e81-b9f3-4ad8-89b9-091fc9b87228
neurons_params_cont = (
	edgecolor=(:white, 0.5), 
	edgewidth=0.1,
	radius=4,
	cmap=cmapa, 
	range=(0,+1),
);

# ╔═╡ 398564e4-158e-4dd4-bd3b-7e66725ddd68
begin
	c_off = neurons_params_cont.cmap[1]
	c_on = neurons_params_cont.cmap[round(Int, 1*length(neurons_params_cont.cmap.colors)/1.4)]
	cmap_OnOff = Makie.Categorical([c_off, c_on])
end

# ╔═╡ e5eebd1b-fbcb-4928-b9f7-aaf6dec8f01f
neurons_params_bin = (
	edgecolor=(:white, 0.5), 
	edgewidth=0.1,
	radius=4,
	#cmap=cmap_dff(), 
	cmap=cmap_OnOff,
	range=(0,1),
);

# ╔═╡ a5cff628-a192-4956-8497-ce7e95b59da4
md"""
# 1. T -> S
"""

# ╔═╡ 140aca9e-e3c6-4d1f-8405-7760e90567f1
# t_TS_start, t_TS_stop = 500, 1000;
t_TS_start, t_TS_stop = 1000, 1500;

# ╔═╡ 7802da5f-9386-4e31-bb9d-26202081d9e3
md"""
# 1. S -> T
"""

# ╔═╡ 1bd88559-fe15-4660-9c41-0f13374d8635
t_ST_start, t_ST_stop = 500, 1000;

# ╔═╡ bcb62d5e-b4f8-4f71-86e9-8cee2e27f2b7


# ╔═╡ 2cc9a225-aed1-49d2-8e79-3bb8b2502059


# ╔═╡ e8956202-85ce-40dd-95de-e5aaba39d05f


# ╔═╡ d8de0442-0db6-433c-a656-30eacdca4b76


# ╔═╡ a18a44ae-10ee-4f57-a72d-d2be9b39347c


# ╔═╡ 67a4354c-c00b-4abd-aa51-170a7b0ff782
md"""
# Tools
"""

# ╔═╡ fd86f5b6-49af-4e8c-9a10-6e2138b5c05d
function get_teacher_student_act(
	teacher::String, 
	student::String,
	rtype::String,
)
	@assert rtype ∈ ["TT", "SS", "TS", "ST"]
	if rtype[1] == 'T'
		# println("input = teacher")
		rbmI,_,_,_,_,_ = load_brainRBM(LOAD.load_wbscRBM("bRBMs",  "bRBM_".*teacher.*base_mod))
		dataI = load_data(LOAD.load_dataWBSC(teacher))
	else
		# println("input = student")
		rbmI,_,_,_,_,_ = load_brainRBM(LOAD.load_wbscRBM("biRBMs", "biRBM_$(student)_FROM_$(teacher)$(base_mod)"))
		dataI = load_data(LOAD.load_dataWBSC(student))
	end
	if rtype[2] == 'T'
		# println("output = teacher")
		rbmO,_,_,_,_,_ = load_brainRBM(LOAD.load_wbscRBM("bRBMs",  "bRBM_".*teacher.*base_mod))
	else
		# println("output = student")
		rbmO,_,_,_,_,_ = load_brainRBM(LOAD.load_wbscRBM("biRBMs", "biRBM_$(student)_FROM_$(teacher)$(base_mod)"))
	end

	Ih = mean_h_from_v(rbmI, dataI.spikes)
	IhO = mean_v_from_h(rbmO ,Ih)
	return IhO
end

# ╔═╡ 8dde8527-5fde-4cf2-bdaf-ce2fe7f6b0ce
begin
	TvmT = get_teacher_student_act(EXteacher, EXstudent, "TT")
	TvmS = get_teacher_student_act(EXteacher, EXstudent, "TS")
end;

# ╔═╡ fa424471-8d5a-4c13-871e-ffc371ad03fe
begin
	t_TS = Observable(t_TS_start)
	fig_mouv_TS = Figure(size=dfsize().*(2,1))
	
	ax_mouv_TS_main = Axis(fig_mouv_TS[1,1], aspect=DataAspect())

	h_TS_bin = neuron2dscatter!(ax_mouv_TS_main,
		dataT.coords[:,1], dataT.coords[:,2],
		@lift(dataT.spikes[:,$t_TS]);
		neurons_params_bin...,
	)

	neuron2dscatter!(ax_mouv_TS_main,
		dataT.coords[:,1] .+ 500, dataT.coords[:,2],
		@lift(TvmT[:,$t_TS]);
		neurons_params_cont...,
	)

	neuron2dscatter!(ax_mouv_TS_main,
		dataS.coords[:,1] .+ 1000, dataS.coords[:,2],
		@lift(TvmS[:,$t_TS]);
		neurons_params_cont...,
	)

	Colorbar(
		fig_mouv_TS[1,2], 
		colormap=neurons_params_cont.cmap, 
		colorrange=neurons_params_cont.range,
		height=Relative(0.5),
		label="𝔼[v]"
	)
	Colorbar(
		fig_mouv_TS[1,0], 
		h_TS_bin,
		# colormap=neurons_params_bin.cmap, 
		# colorrange=neurons_params_bin.range,
		height=Relative(0.1),
		flipaxis=false,
		label="Spikes",
		# ticks=([0,1],["off","on"]),
	)

	text!(ax_mouv_TS_main,
		150, 10, 
		text="200μm", 
		align=(:center, :bottom),
	)
	lines!(ax_mouv_TS_main,
		[50,250], [0,0], 
		color=:black,
	)
	text!(ax_mouv_TS_main,
		50, 1000, 
		text="$(speed_up)×", 
		align=(:center, :bottom),
	)

	text!(ax_mouv_TS_main,
		250, 1000, 
		text="vᵀ", 
		align=(:center, :bottom),
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		# font=Makie.current_default_theme().Axis.titlefont.val,
	)
	text!(ax_mouv_TS_main,
		250+500, 1000, 
		text="𝔼ᵀ[v | 𝔼ᵀ[h|vᵀ]]", 
		align=(:center, :bottom),
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		# font=Makie.current_default_theme().Axis.titlefont.val,
	)
	text!(ax_mouv_TS_main,
		250+1000, 1000, 
		text="𝔼ˢ[v | 𝔼ᵀ[h|vᵀ]]", 
		align=(:center, :bottom),
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		# font=Makie.current_default_theme().Axis.titlefont.val,
	)


	
	text!(ax_mouv_TS_main,
		250, 1100, 
		text="Data T", 
		align=(:center, :bottom),
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		# font=Makie.current_default_theme().Axis.titlefont.val,
	)
	text!(ax_mouv_TS_main,
		250+500, 1100, 
		text="T➔T", 
		align=(:center, :bottom),
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		# font=Makie.current_default_theme().Axis.titlefont.val,
	)
	text!(ax_mouv_TS_main,
		250+1000, 1100, 
		text="T➔S", 
		align=(:center, :bottom),
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		# font=Makie.current_default_theme().Axis.titlefont.val,
	)

	fig_mouv_TS
end

# ╔═╡ f43db26a-0f6d-4014-8e08-971078693b13
display(fig_mouv_TS)

# ╔═╡ 2a055b66-e1d5-4426-b4d4-6fdb86ef73d8
record(
		fig_mouv_TS, 
		replace(@figpath("DeepFake_TS_green"), ".svg"=>".mp4"), 
		t_TS_start:t_TS_stop,
		framerate=out_framerate,
		format="mp4", profile="main", 
	) do time
		t_TS[] = time
	end

# ╔═╡ 563bb535-5df2-408b-bc3f-ea7655fa9343
begin
	SvmS = get_teacher_student_act(EXteacher, EXstudent, "SS")
	SvmT = get_teacher_student_act(EXteacher, EXstudent, "ST")
end;

# ╔═╡ e8e927ff-3e53-4550-b2bf-134862ca52cb
begin
	t_ST = Observable(t_ST_start)
	fig_mouv_ST = Figure(size=dfsize().*(2,1))
	
	ax_mouv_ST_main = Axis(fig_mouv_ST[1,1], aspect=DataAspect())

	h_ST_bin = neuron2dscatter!(ax_mouv_ST_main,
		dataS.coords[:,1], dataS.coords[:,2],
		@lift(dataS.spikes[:,$t_ST]);
		neurons_params_bin...,
	)

	neuron2dscatter!(ax_mouv_ST_main,
		dataS.coords[:,1] .+ 500, dataS.coords[:,2],
		@lift(SvmS[:,$t_ST]);
		neurons_params_cont...,
	)

	neuron2dscatter!(ax_mouv_ST_main,
		dataT.coords[:,1] .+ 1000, dataT.coords[:,2],
		@lift(SvmT[:,$t_ST]);
		neurons_params_cont...,
	)

	Colorbar(
		fig_mouv_ST[1,2], 
		colormap=neurons_params_cont.cmap, 
		colorrange=neurons_params_cont.range,
		height=Relative(0.5),
		label="𝔼[v]"
	)
	Colorbar(
		fig_mouv_ST[1,0], 
		h_ST_bin,
		# colormap=neurons_params_bin.cmap, 
		# colorrange=neurons_params_bin.range,
		height=Relative(0.1),
		flipaxis=false,
		label="Spikes",
		# ticks=([0,1],["off","on"]),
	)

	text!(ax_mouv_ST_main,
		150, 10, 
		text="200μm", 
		align=(:center, :bottom),
	)
	lines!(ax_mouv_ST_main,
		[50,250], [0,0], 
		color=:black,
	)
	text!(ax_mouv_ST_main,
		50, 1000, 
		text="$(speed_up)×", 
		align=(:center, :bottom),
	)

	text!(ax_mouv_ST_main,
		250, 1000, 
		text="vˢ", 
		align=(:center, :bottom),
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		# font=Makie.current_default_theme().Axis.titlefont.val,
	)
	text!(ax_mouv_ST_main,
		250+500, 1000, 
		text="𝔼ˢ[v | 𝔼ˢ[h|vˢ]]", 
		align=(:center, :bottom),
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		# font=Makie.current_default_theme().Axis.titlefont.val,
	)
	text!(ax_mouv_ST_main,
		250+1000, 1000, 
		text="𝔼ᵀ[v | 𝔼ˢ[h|vˢ]]", 
		align=(:center, :bottom),
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		# font=Makie.current_default_theme().Axis.titlefont.val,
	)


	
	text!(ax_mouv_ST_main,
		250, 1100, 
		text="Data S", 
		align=(:center, :bottom),
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		# font=Makie.current_default_theme().Axis.titlefont.val,
	)
	text!(ax_mouv_ST_main,
		250+500, 1100, 
		text="S➔S", 
		align=(:center, :bottom),
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		# font=Makie.current_default_theme().Axis.titlefont.val,
	)
	text!(ax_mouv_ST_main,
		250+1000, 1100, 
		text="S➔T", 
		align=(:center, :bottom),
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		# font=Makie.current_default_theme().Axis.titlefont.val,
	)

	fig_mouv_ST
end

# ╔═╡ 825038e3-40da-46bb-b7f6-14f22d172f1a
record(
		fig_mouv_ST, 
		replace(@figpath("DeepFake_ST_green"), ".svg"=>".mp4"),
		t_ST_start:t_ST_stop,
		framerate=out_framerate,
		format="mp4", profile="main", 
	) do time
		t_ST[] = time
	end

# ╔═╡ Cell order:
# ╟─6188f2d4-3aea-11f0-0f7e-0fb793ee5f9f
# ╠═d2053251-4612-4b20-b8a4-0e225825fbbf
# ╠═5c1e28c7-b9a3-4b50-8672-ae0202df866b
# ╠═cfc4d607-fc48-4e4b-841e-40c5dc4e0ab6
# ╠═9f4df0ca-7366-4c3f-90c7-6a8d80f510be
# ╠═2fef345f-1d56-4287-8afb-1c8e8e95f7dd
# ╠═ca29d3d8-4e7c-479f-aaf8-7c37d5960063
# ╠═6688652f-925c-4eca-81ca-178abf033276
# ╠═13430c22-29c6-4489-ab89-6e07070bedb0
# ╠═0634840b-2470-42d1-9974-18b16a0549fb
# ╟─2e35c6aa-e6a0-4dc3-a447-c1776fdcc071
# ╠═e9a155a7-5f98-4e72-b0b4-cfbfcc937bb3
# ╠═42e44427-d2c8-40b6-b9cc-38779bafa326
# ╟─b828f13c-2bac-4db3-b4c1-b9df3f975dfd
# ╟─d25cb057-402f-49c7-b5a7-ab43cb48da35
# ╟─f88c08fd-1c27-4429-8f46-40ae86b61db4
# ╠═d1cc31ba-876e-469a-a8e7-17759aed13e9
# ╠═d7c71d33-9958-4486-abf9-71858bb16d49
# ╠═3475cc75-dfc2-4e69-b508-197af4328335
# ╠═42584857-a753-472a-ba72-18cfed2577de
# ╠═26748574-8755-4379-83b4-7d33cef86372
# ╠═b4cbe70e-9413-4875-9e64-4dbc97b816f7
# ╠═1f07e89f-06d8-4417-a86d-6643be5645f2
# ╠═f5742e81-b9f3-4ad8-89b9-091fc9b87228
# ╠═398564e4-158e-4dd4-bd3b-7e66725ddd68
# ╠═e5eebd1b-fbcb-4928-b9f7-aaf6dec8f01f
# ╟─a5cff628-a192-4956-8497-ce7e95b59da4
# ╠═8dde8527-5fde-4cf2-bdaf-ce2fe7f6b0ce
# ╠═140aca9e-e3c6-4d1f-8405-7760e90567f1
# ╠═fa424471-8d5a-4c13-871e-ffc371ad03fe
# ╠═f43db26a-0f6d-4014-8e08-971078693b13
# ╠═2a055b66-e1d5-4426-b4d4-6fdb86ef73d8
# ╟─7802da5f-9386-4e31-bb9d-26202081d9e3
# ╠═563bb535-5df2-408b-bc3f-ea7655fa9343
# ╠═1bd88559-fe15-4660-9c41-0f13374d8635
# ╠═e8e927ff-3e53-4550-b2bf-134862ca52cb
# ╠═825038e3-40da-46bb-b7f6-14f22d172f1a
# ╠═bcb62d5e-b4f8-4f71-86e9-8cee2e27f2b7
# ╠═2cc9a225-aed1-49d2-8e79-3bb8b2502059
# ╠═e8956202-85ce-40dd-95de-e5aaba39d05f
# ╠═d8de0442-0db6-433c-a656-30eacdca4b76
# ╠═a18a44ae-10ee-4f57-a72d-d2be9b39347c
# ╟─67a4354c-c00b-4abd-aa51-170a7b0ff782
# ╠═fd86f5b6-49af-4e8c-9a10-6e2138b5c05d
