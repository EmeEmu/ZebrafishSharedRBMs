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

# ╔═╡ a088a9f3-b96c-49dd-8266-be67e657a4e0
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ ce99e7e7-0a15-4afe-b1ff-7768a0d9c6e8
begin
	using BrainRBMjulia
	using LinearAlgebra: diagind
	using HDF5
	
	using CairoMakie
	using BrainRBMjulia: multipolarnrmseplotter!, idplotter!, dfsize, quantile_range, neuron2dscatter!, cmap_aseismic, polarnrmseplotter!
	using ColorSchemes: reverse, RdYlGn_9

	CONV = @ingredients("conventions.jl")
	include(joinpath(dirname(Base.current_project()), "Misc_Code", "fig_saving.jl"))
end

# ╔═╡ bf857ba1-f41d-4e0c-b0d5-9defbb3373cc
using ColorSchemes

# ╔═╡ 0ff9016e-d7a4-4ccf-806e-86a28f2daa21
begin
	using Colors, ColorSchemeTools
	function ggr_diverging(vmin::Real, vmax::Real; n::Int = 256, basemap=ColorSchemes.reverse(ColorSchemes.RdYlGn_9))
	    @assert vmin ≤ 0 ≤ 1 ≤ vmax "range must include 0 and 1"
	
	    p0 = (0 - vmin)/(vmax - vmin)            # where green → gold
	    p1 = (1 - vmin)/(vmax - vmin)            # where gold  → red
	    pm = (p0 + p1)/2                         # pure gold
		
		idxlist = [
	        (0.0, (basemap[1].r,basemap[1].g,basemap[1].b).*0.75),
		]
		for i in 1:length(basemap)
			x = (i-1)/(length(basemap)-1)
			y = x*(p1-p0) + p0
			push!(
				idxlist, 
				(y, (basemap[i].r,basemap[i].g,basemap[i].b))
			)
		end
		push!(idxlist, (1.0, (basemap[end].r,basemap[end].g,basemap[end].b).*0.1))
	    make_colorscheme(Tuple(idxlist); length = n)    # ColorSchemeTools
	end
	
end

# ╔═╡ c2a89a06-dab9-4da5-813d-21bcb6777ee3
using BrainRBMjulia: cmap_ainferno

# ╔═╡ 7b88bfed-2dfe-41c8-9859-8376e3687081
using BrainRBMjulia: cmap_Gbin

# ╔═╡ aa956757-8042-4e15-bf78-abadf74bea8c
begin
	using Statistics: median
	function free_energy_contrast(h5path::String, teacher::String, student::String, rtype::String; rmax::String="randv", rmin::String="aa")
		if teacher == student
			return NaN
		end
		teacher_i = findall(h5read(h5path, "fish_list") .== teacher)[1]
		student_j = findall(h5read(h5path, "fish_list") .== student)[1]
	
		@assert length(rtype) == 2
		inr, outr = rtype
	
		@assert rmax ∈ ["randv", "randvv"]
		@assert rmin ∈ ["a", "aa"]
	
		if outr == 'T'
			X = median(h5read(h5path, "Teacher_$teacher_i/Student_$student_j/F_ST"))
			Xmax = median(h5read(h5path, "Teacher_$teacher_i/F_T_$rmax"))
			if length(rmin) == 1
				Xmin = median(h5read(h5path, "Teacher_$teacher_i/F_T"))
			else
				Xmin = median(h5read(h5path, "Teacher_$teacher_i/F_TT"))
			end
	
		elseif outr == 'S'
			X = median(h5read(h5path, "Teacher_$teacher_i/Student_$student_j/F_TS"))
			Xmax = median(h5read(h5path, "Teacher_$teacher_i/Student_$student_j/F_S_$rmax"))
			if length(rmin) == 1
				Xmin = median(h5read(h5path, "Teacher_$teacher_i/Student_$student_j/F_S"))
			else
				Xmin = median(h5read(h5path, "Teacher_$teacher_i/Student_$student_j/F_SS"))
			end
			
		else
			return NaN
		end
	
		return (X - Xmin) / (Xmax - Xmin)
		# return (exp(Xmin-X) - 1) / (exp(Xmin-Xmax) - 1)
		# return exp(Xmin-Xmax) - 1
		# return exp(Xmin-X)-1 , exp(Xmin-Xmax)-1
	end
	
	function free_energy_contrast(h5path::String, fish::Vector{String}, rtype::String; rmax::String="randv", rmin::String="aa")
		return [free_energy_contrast(h5path, t, s, rtype; rmax, rmin) for t=fish, s=fish]
	end
end

# ╔═╡ 6f15ddb3-5eec-44c3-8056-96dd23bf0c03
using HypothesisTests

# ╔═╡ 554a0fb4-38ef-11f0-10d3-8bf0f5e7002d
md"""
# Imports + Notebook Preparation
"""

# ╔═╡ bbe5511a-ae48-4afc-8901-dd5467358e2a
TableOfContents()

# ╔═╡ fc7fc50c-9d9a-4e1d-94a7-26690ac496c6
set_theme!(CONV.style_publication)

# ╔═╡ db58397d-3121-48ad-83b9-cfd896e95b9e
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ 14d3ee1c-0343-4630-800f-6c00926c99de


# ╔═╡ cbe1e272-76ba-45aa-bc89-7109664153f2
md"""
# 0. Fish and training base
"""

# ╔═╡ 2d7880d8-84ad-45ed-ae3d-c5664f320da6
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ 9ed90752-edd7-4f4b-93c1-99d3301f2b8d
md"Use numbers instead of fish names $(@bind numbered CheckBox(default=true))"

# ╔═╡ 956c18fe-276f-44e2-a6a6-6720ea47fb4b
base_mod = "*_WBSC_M100_l10.02_l2l10";

# ╔═╡ b18a79ba-1cbc-49b6-8f13-767a28c19aee
begin
	using BrainRBMjulia: var_h_from_v
	
	function ddsigmoid(x::Real)
		A = (exp(x) - 1)*exp(x)
		B = (exp(x) + 1)^3
		return -A/B
	end
	
	function ddf(
		rbm::R, 
		h::AbstractVector,
	) where R <: Union{BrainRBMjulia.RBM, BrainRBMjulia.StandardizedRBM}
		ddsigmoid.(rbm.visible.θ .* (rbm.w * h)) .* rbm.w.^2
	end

	function transfer_joint(teacher::String, student::String; N_t::Int=100)
		rbmT,_,_,splitT,_,_ = load_brainRBM(LOAD.load_wbscRBM(
			"bRBMs",  
			"bRBM_".*teacher.*base_mod
		))
		dataT = load_data(LOAD.load_dataWBSC(teacher))
		rbmS,_,_,splitS,_,_ = load_brainRBM(LOAD.load_wbscRBM(
			"biRBMs", 
			"biRBM_$(student)_FROM_$(teacher)$(base_mod)"
		))

		v_old = mean_v_from_h(rbmS, mean_h_from_v(rbmT, dataT.spikes[:,1:N_t]));

		
		Δs = Matrix{Float64}(undef, size(v_old)...)
		for t in 1:size(v_old,2)
			v = dataT.spikes[:,t]
			h = sample_h_from_v(rbmT, v)
			f′′ = ddf(rbmS, h)
			Var = var_h_from_v(rbmT, v)
			Δs[:,t] .= 0.5 .* f′′ * Var
		end
		Δs[isnan.(Δs)] .= 0

		return v_old, Δs
	end
end

# ╔═╡ b54f03ba-5cc3-433b-b494-e5cd904e78ab
for fish in FISH
	println(LOAD.load_dataWBSC(fish))
	println(LOAD.load_wbscRBM("bRBMs", "bRBM_".*fish.*base_mod))
end

# ╔═╡ a31f926e-fc44-4f30-8f31-e92dbf487977
md"example teacher : $(@bind EXteacher Select(FISH, default=FISH[3]))"

# ╔═╡ 37e5c5b3-352b-4bbb-9831-aa02aa923d63
STUDENTS = [fish for fish in FISH if fish!=EXteacher];

# ╔═╡ 90145bf6-353c-4c83-8627-31f59f7cb338
md"example student : $(@bind EXstudent Select(STUDENTS, default=STUDENTS[1]))"

# ╔═╡ 8aef390a-9c77-44f9-a966-e4488253a72b
begin
	if numbered
		FISH_DISP = ["F$i" for i in 1:length(FISH)]
		STUDENT_DISP = ["Student $i" for i in 1:length(STUDENTS)]
	else
		FISH_DISP = FISH
		STUDENT_DISP = STUDENTS
	end
end;

# ╔═╡ ec1bd161-3868-4505-84e7-e1572a38da46


# ╔═╡ 02b71d98-90a7-401c-933d-83f0c8f1183e
md"""
# 1. Main Figure
"""

# ╔═╡ 9a9ad7d5-80ed-4de5-8c72-d1870954980c
begin
	fig_main = Figure(size=(53, 49).*(4,6).*(4/3/0.35)) #[ud, lr]
	
	g_a = fig_main[1, 1:6] = GridLayout()

	g_bcdefg = fig_main[2:6, 1:6] = GridLayout()

	g_bdf = g_bcdefg[1:5, 1:2] = GridLayout()
	g_ceg = g_bcdefg[1:5, 4:5] = GridLayout()
	g_CBL =  g_bcdefg[1:5, 3] = GridLayout()
	g_CBR =  g_bcdefg[1:5, 6] = GridLayout()

	g_b = g_bdf[1:3, 1] = GridLayout()
	g_d = g_bdf[4, 1] = GridLayout()
	g_f = g_bdf[5, 1] = GridLayout()

	g_c = g_ceg[1:3, 1] = GridLayout()
	g_e = g_ceg[4, 1] = GridLayout()
	g_g = g_ceg[5, 1] = GridLayout()

	g_bCB = g_CBL[1:3,1] = GridLayout()
	g_empty = g_CBL[4:5,1] = GridLayout()
	g_cCB = g_CBR[1:3,1] = GridLayout()
	g_eCB = g_CBR[4,1] = GridLayout()
	g_gCB = g_CBR[5,1] = GridLayout()

	for (label, layout) in zip(
		["A", "B", "C", "D", "E", "F", "G"],#, "H", "I", "J", "K"], 
		[g_a, g_b, g_c, g_d, g_e, g_f, g_g],#, g_h, g_i, g_j, g_k]
	)
	    Label(layout[1, 1, TopLeft()], label,
	        fontsize = Makie.current_default_theme().Axis.titlesize.val,
	        font = :bold,
	        padding = (0, 5, 5, 0),
	        halign = :right)
	end
end

# ╔═╡ 0d6a15d8-2cc1-4f1e-a13a-57b9644c4ac1
md"""
## 1.B. Transfer Statistics - Example
"""

# ╔═╡ 07059f14-2432-45cf-9b0e-88847942f0f1


# ╔═╡ 44013910-35c4-4614-862d-87361a10b47d
md"""
## 1.C. Transfer Statistics - All pairs
"""

# ╔═╡ 8c4ed0eb-d052-4bb6-a2a0-12e463c600c2


# ╔═╡ 53a203c5-313d-4b53-ba77-ec238bc22ad5
md"""
## 1.D. Free Energy - Example
"""

# ╔═╡ a198bb85-d365-4b36-9975-84c94e8bd7c8
inpathfreenergy = LOAD.load_misc("DeepFakeFreeEnergy_$(length(FISH))fish_$(base_mod[3:end])")

# ╔═╡ 79dc3ae7-2fbd-4718-ac40-7442389c70c6
begin
	teacher_i = findall(h5read(inpathfreenergy, "fish_list") .== EXteacher)[1]
	student_j = findall(h5read(inpathfreenergy, "fish_list") .== EXstudent)[1]

	F_T = h5read(inpathfreenergy, "Teacher_$teacher_i/F_T")
	F_T_randvv = h5read(inpathfreenergy, "Teacher_$teacher_i/F_T_randvv")
	F_T_randv = h5read(inpathfreenergy, "Teacher_$teacher_i/F_T_randv")
	F_TT = h5read(inpathfreenergy, "Teacher_$teacher_i/F_TT")

	F_S = h5read(inpathfreenergy, "Teacher_$teacher_i/Student_$student_j/F_S")
	F_S_randvv = h5read(inpathfreenergy, "Teacher_$teacher_i/Student_$student_j/F_S_randvv")
	F_S_randv = h5read(inpathfreenergy, "Teacher_$teacher_i/Student_$student_j/F_S_randv")
	F_Sb = h5read(inpathfreenergy, "Teacher_$teacher_i/Student_$student_j/F_Sb")
	F_SS = h5read(inpathfreenergy, "Teacher_$teacher_i/Student_$student_j/F_SS")
	F_ST = h5read(inpathfreenergy, "Teacher_$teacher_i/Student_$student_j/F_ST")
	F_TS = h5read(inpathfreenergy, "Teacher_$teacher_i/Student_$student_j/F_TS")
	F_S_ind = h5read(inpathfreenergy, "Teacher_$student_j/F_T")

	
	F_list_T = [F_T, F_TT, F_ST];
	F_list_lab_T = ["T", "T→T", "S→T"];
	
	F_list_S = [F_S, F_SS, F_TS];
	F_list_lab_S = ["S", "S→S", "T→S"];

	F_cat_T = vcat(F_list_T...)
	F_lab_T = vcat([fill(i, size(F_list_T[i])) for i in 1:length(F_list_T)]...)

	F_cat_S = vcat(F_list_S...)
	F_lab_S = vcat([fill(i, size(F_list_S[i])) for i in 1:length(F_list_S)]...)
end;

# ╔═╡ 71c3cc8c-656f-467f-9522-7bb3bf5a2977
begin
	#fig_freeenergy_ex = Figure(size=dfsize().*(1.5,1))
	
	Qs1 = [0.25,0.75];
	Qs2 = [0.005,0.995];

	
	ax_freeenergy_ex_y = Axis(
		g_d[1,1],
		# xticks = (1:length(F_list_lab_T), F_list_lab_T),
		ylabel = "Free Energy , F(v)",
		ytickformat = ys -> ["$(round(Int, y/1.e3))" for y in ys],
		bottomspinevisible=false,
		xticksvisible=false, xticklabelsvisible=false
	)
	Label(g_d[1,1,Top()], halign=:left, "×10³")
	
	ax_freeenergy_ex_F_T = Axis(
		g_d[1,2],
		xticks = (1:length(F_list_lab_T), F_list_lab_T),
		# ylabel = "F(v)",
		ytickformat = ys -> ["$(round(Int, y/1.e3))" for y in ys],
		leftspinevisible=false, yticksvisible=false, yticklabelsvisible=false,
	)
	hspan!(quantile(F_T_randv, Qs1)..., color=(:red, 0.2))
	hspan!(quantile(F_T_randvv, Qs1)..., color=(:orange, 0.2))
	hspan!(quantile(F_T_randv, Qs2)..., color=(:red, 0.1))
	hspan!(quantile(F_T_randvv, Qs2)..., color=(:orange, 0.1))
	violin!(F_lab_T, F_cat_T, color=CONV.COLOR_TEACHER, scale=:area, width=1.2, show_median=true)
	
	ax_freeenergy_ex_F_S = Axis(
		g_d[1,3],
		xticks = (1:length(F_list_lab_S), F_list_lab_S),
		# ylabel = "F(v)",
		# ytickformat = ys -> ["$(round(Int, y/1.e3))" for y in ys],
		leftspinevisible=false, yticksvisible=false, yticklabelsvisible=false,
	)
	hspan!(quantile(F_S_randv, Qs1)..., color=(:red, 0.2))
	hspan!(quantile(F_S_randvv, Qs1)..., color=(:orange, 0.2))
	hspan!(quantile(F_S_randv, Qs2)..., color=(:red, 0.1))
	hspan!(quantile(F_S_randvv, Qs2)..., color=(:orange, 0.1))
	violin!(F_lab_S, F_cat_S, color=CONV.COLOR_STUDENT, scale=:area, width=1.2, show_median=true)


	ylims!(-1.5e3, max(maximum(F_T_randv), maximum(F_S_randv)))
	linkyaxes!(ax_freeenergy_ex_y, ax_freeenergy_ex_F_T, ax_freeenergy_ex_F_S)

	colsize!(g_d, 1, Relative(0.0))
	
	# fig_freeenergy_ex
end

# ╔═╡ 75921dee-499c-4930-bff4-c71cf5d63e89


# ╔═╡ bef64e9f-067d-4676-99c4-30b8fdc3fad2
md"""
## 1.E. Free Energy - All Pairs
"""

# ╔═╡ bf1f4985-3062-4ee9-84db-46153c491b82
ColorSchemes.reverse(ColorSchemes.RdYlGn_9)

# ╔═╡ bb96c64c-e83c-47f2-a4d6-6667e946b36a
function gold_diverging(floor, ceil)
    @assert floor ≤ 0 ≤ 1 ≤ ceil "range must cover 0 and 1"

    # relative positions of the two break‑points within [0,1]
    s0 = (0   - floor)/(ceil - floor)       # where the green half turns to gold
    s1 = (1   - floor)/(ceil - floor)       # where the gold half turns to red
    sm = (s0 + s1)/2                        # gold itself (mid‑point)

    ColorScheme(
        [
            colorant"#004400",              # dark‑green  (xmin)
            colorant"#00ff00",              # green       (0)
            colorant"#ffd700",              # gold        (½)
            colorant"#ff0000",              # red         (1)
            colorant"#440000"               # dark‑red    (xmax)
        ],
        stops = [0.0, s0, sm, s1, 1.0],
        name  = "Green‑Gold‑Red"
    )
end

# ╔═╡ 1e9f1588-fa0c-4886-bf2d-81cf65daf121
begin
	fe_constrastTS = free_energy_contrast(inpathfreenergy, FISH, "TS", rmin="aa", rmax="randvv")
	fe_constrastST = free_energy_contrast(inpathfreenergy, FISH, "ST", rmin="aa", rmax="randvv")
end;

# ╔═╡ de4c7c6a-b09b-4805-a139-4f05b6334347
begin
	fe_cont_min = min(minimum(fe_constrastTS[isfinite.(fe_constrastTS)]), minimum(fe_constrastST[isfinite.(fe_constrastST)]))
	fe_cont_max = max(maximum(fe_constrastTS[isfinite.(fe_constrastTS)]), maximum(fe_constrastST[isfinite.(fe_constrastST)]))
end

# ╔═╡ fd54108d-99b3-49b5-a3c3-dfdb2aad83eb
begin
	# fig_freeenergy_all = Figure(size=dfsize().*(2.3,1.3))
	
	ax_fe_allTS = Axis(
		g_e[1,1], aspect=1,
	    xticks=((1:length(FISH)), FISH_DISP),
	    yticks=((1:length(FISH)), FISH_DISP),
		xticklabelrotation = π/4,
		xlabel = "Teacher", ylabel="Student",
		title="T→S"
	)
	heatmap!(
		ax_fe_allTS,
		free_energy_contrast(inpathfreenergy, FISH, "TS", rmin="aa", rmax="randvv"),
		colormap=ggr_diverging(fe_cont_min, fe_cont_max),
		colorrange=(fe_cont_min, fe_cont_max),
	)

	
	
	ax_fe_allST = Axis(
		g_e[1,2], aspect=1,
	    xticks=((1:length(FISH)), FISH_DISP),
	    yticks=((1:length(FISH)), FISH_DISP),
		xticklabelrotation = π/4,
		# xlabel = "Teacher", 
		# ylabel="Student",
		yticklabelsvisible=false,
		xticklabelsvisible=false,
		title="S→T"
	)
	heatmap!(
		ax_fe_allST,
		free_energy_contrast(inpathfreenergy, FISH, "ST", rmin="aa", rmax="randvv"),
		colormap=ggr_diverging(fe_cont_min, fe_cont_max),
		colorrange=(fe_cont_min, fe_cont_max),
	)

	Colorbar(
		g_eCB[1,1], 
		colormap=ggr_diverging(fe_cont_min, fe_cont_max),
		colorrange=(fe_cont_min, fe_cont_max),
		label=L"\frac{F(\mathbf{v}^{f_1 \to f_2}) - F(\mathbf{v}^{f_2 \to f_2})}{F(\mathbf{v}^{f_1}_{i, \pi_i(t)}) - F(\mathbf{v}^{f_2 \to f_2})}"
	)
	
	# fig_freeenergy_all
end

# ╔═╡ 5ac7a0e0-a796-415c-a0ca-4626138d1795


# ╔═╡ b0db6f7e-c489-4e96-bbee-3659d6da5fe2
md"""
## 1.G. Distance between transfered frames - Matrix
"""

# ╔═╡ 7a71fccc-3fd7-4a4b-82b3-ea4b87158b03


# ╔═╡ 90f0dcd4-85c8-4962-999e-1a39eab3f26a


# ╔═╡ 210298d8-879d-4067-8954-f8d361306894
md"""
## 1.F. Distance between transfered frames - Example
"""

# ╔═╡ f2623330-f6ac-445a-9314-81c0eaa512c5


# ╔═╡ 435737df-a681-49b9-8d0d-3b0585c23a61
md"""
## 1.END Adjustments
"""

# ╔═╡ 89e0109a-8ca7-42ff-8a20-bc3ccb8bf871
all_axes = [ax for ax in fig_main.content if typeof(ax)==Axis];

# ╔═╡ dcdde4e6-7599-43b3-ab0e-9637e0454d6e
# ╠═╡ disabled = true
#=╠═╡
for ax in all_axes
	ax.alignmode = Mixed(left=0)
end
  ╠═╡ =#

# ╔═╡ e4e39a30-bb6c-4589-b602-101be8917032
fig_main

# ╔═╡ 2106652f-5e59-41d1-8400-1be570bfed6c
# ╠═╡ disabled = true
#=╠═╡
save(@figpath("main"), fig_main)
  ╠═╡ =#

# ╔═╡ 695fb21c-035c-46a9-8596-ac02bee19fa6


# ╔═╡ 417e1a28-3407-4347-b8ae-8f0eb0752ea6
md"""
# 2. Supplementaries
"""

# ╔═╡ 2377b684-9eec-4d88-8e41-62d44baafd53
md"""
## 2.1. Bad Residuals
"""

# ╔═╡ a1e06e54-1202-4fc5-9620-a84ad7bae950
begin
	fig_S_bad_residuals = Figure(size=(53, 49).*(3.2,3).*(4/3/0.35))
	
	g_a_S_bad_residuals = fig_S_bad_residuals[1,1] = GridLayout()
	g_b_S_bad_residuals = fig_S_bad_residuals[1,2] = GridLayout()
	g_c_S_bad_residuals = fig_S_bad_residuals[2:3,1:3] = GridLayout()
	g_d_S_bad_residuals = fig_S_bad_residuals[1,3] = GridLayout()
	
	for (label, layout) in zip(
		["A", "B", "C", "D"], 
		[g_a_S_bad_residuals, g_b_S_bad_residuals, g_c_S_bad_residuals, g_d_S_bad_residuals]
	)
		Label(layout[1, 1, TopLeft()], label,
			fontsize = Makie.current_default_theme().Axis.titlesize.val,
			font = :bold,
			padding = (0, 5, 5, 0),
			halign = :right)
	end
end

# ╔═╡ f857d4e9-725c-4dd4-82a1-86524774e7c0


# ╔═╡ bd71917e-ca95-4245-a9a2-a71244a7167a
md"""
### 2.1.A. percent bad residuals example
"""

# ╔═╡ 4a21f8b2-10ef-4df0-bb5a-1e42d1b618f9
md"""
### 2.1.B. percent bad residuals matrix
"""

# ╔═╡ 30a648ec-2dd8-45b4-98cf-9fd5de916dcd
md"""
### 2.1.C. Maps
"""

# ╔═╡ 126a45a9-fe66-4bb7-bcfb-655492c04780


# ╔═╡ ec2cf804-ec40-4f8b-bb56-f1ca68940c0e
md"""
### 2.1.D. Hidden Participation
"""

# ╔═╡ 617245a3-dfcf-47a4-9660-b2bccc5b7238


# ╔═╡ bdff6eef-3213-4743-a37a-32222c9831f0


# ╔═╡ 54e8a938-ffd1-45c6-9bc2-7d04439ee80d
md"""
### 2.1.END Adjustments
"""

# ╔═╡ 2a7960bb-0b21-4270-bfe5-f5766cabafee
all_axes_S_bad_residuals = [ax for ax in fig_S_bad_residuals.content if typeof(ax)==Axis];

# ╔═╡ ade4796e-e7cc-46b1-b520-b85abb37295a
for ax in all_axes_S_bad_residuals
	ax.alignmode = Mixed(left=0)
end

# ╔═╡ 6a01c7ec-d1e9-4b7b-9af4-37f3821cbd82
fig_S_bad_residuals

# ╔═╡ 57e4fbeb-4c50-45de-a07c-ee5c3cbb852c
# ╠═╡ disabled = true
#=╠═╡
save(@figpath("SUPP_residuals"), fig_S_bad_residuals)
  ╠═╡ =#

# ╔═╡ 8f4cb83c-e9e3-4e51-bac7-ffff3ff7f9e0


# ╔═╡ 59e629bf-99f5-49b2-bb0b-5d0d052bcc8a


# ╔═╡ e2d36e5b-41dc-43cf-881b-5ba948345662


# ╔═╡ 776c1dc0-b096-469c-b44b-f88a53c5dd26


# ╔═╡ 70cc2097-7a3f-436f-891d-436fb4effdde
md"""
## 2.2. Activity transfer method v→h→v
"""

# ╔═╡ 872932aa-5f15-423b-a511-0dd9b1332035
inpathtransfer = LOAD.load_misc("DeepFakeTransferMethods_$(length(FISH))fish_$(base_mod[3:end])")

# ╔═╡ aa63fa66-7946-4e66-8bd9-089a4e6de405
n_samples_vhv = h5read(inpathtransfer, "n_samples")

# ╔═╡ f4a246ab-60bd-4db9-828a-5eca0149aabd
begin
	fig_S_transfer_method = Figure(size=(53, 49).*(2,4).*(4/3/0.35))
	
	g_a_S_transfer_method = fig_S_transfer_method[1,1] = GridLayout()
	g_b_S_transfer_method = fig_S_transfer_method[1,2] = GridLayout()
	g_c_S_transfer_method = fig_S_transfer_method[2,1] = GridLayout()
	g_d_S_transfer_method = fig_S_transfer_method[2,2] = GridLayout()
	g_e_S_transfer_method = fig_S_transfer_method[3:4,1:2] = GridLayout()
	
	for (label, layout) in zip(
		["A", "B", "C", "D", "E"], 
		[g_a_S_transfer_method, g_b_S_transfer_method, g_c_S_transfer_method, g_d_S_transfer_method, g_e_S_transfer_method]
	)
		Label(layout[1, 1, TopLeft()], label,
			fontsize = Makie.current_default_theme().Axis.titlesize.val,
			font = :bold,
			padding = (0, 5, 5, 0),
			halign = :right)
	end
end

# ╔═╡ acd47348-5df8-4a04-a518-e23e4315f5e8
md"""
### 2.2.AB Id plot
"""

# ╔═╡ 75fe17c1-4f45-4d76-a403-c87c268250c4
begin
	ax_a_S_transfer_method = Axis(
		g_a_S_transfer_method[1,1],
		aspect=1,
		xlabel=L"\mathbb{E}_{\mathbf{h}_T|\mathbf{v}_T} \mathbb{E}_{\mathbf{v}_S|\mathbf{h}_T} \mathbf{v}_S",
		ylabel=L"\mathbb{E}_{\mathbf{v}_S|\mathbb{E}_{\mathbf{h}_T|\mathbf{v}_T}} \mathbf{v}_S",
	)

	idplotter!(
		ax_a_S_transfer_method,
		h5read(inpathtransfer, "Teacher_$(findall(h5read(inpathfreenergy, "fish_list") .== EXteacher)[1])/Student_$(findall(h5read(inpathfreenergy, "fish_list") .== EXstudent)[1])/v_new"),
		h5read(inpathtransfer, "Teacher_$(findall(h5read(inpathfreenergy, "fish_list") .== EXteacher)[1])/Student_$(findall(h5read(inpathfreenergy, "fish_list") .== EXstudent)[1])/v_old"),
	)
end

# ╔═╡ 9a336639-20fe-4c9f-85ca-52f412be7199
begin
	ax_b_S_transfer_method = Axis(
		g_b_S_transfer_method[1,1],
		aspect=1,
		xlabel=L"\mathbb{E}_{\mathbf{h}_T|\mathbf{v}_T} \mathbb{E}_{\mathbf{v}_S|\mathbf{h}_T} \mathbf{v}_S",
		ylabel=L"\mathbb{E}_{\mathbf{v}_S|\mathbb{E}_{\mathbf{h}_T|\mathbf{v}_T}} \mathbf{v}_S",
	)

	idplotter!(
		ax_b_S_transfer_method,
		reduce(vcat, vec([h5read(inpathtransfer, "Teacher_$i/Student_$j/v_new") for i=1:length(FISH) for j=1:length(FISH) if i != j ])),
		reduce(vcat, vec([h5read(inpathtransfer, "Teacher_$i/Student_$j/v_old") for i=1:length(FISH) for j=1:length(FISH) if i != j ]))
	)
end

# ╔═╡ 86c4e793-c7d2-4284-97f4-7d87366dd75d
md"""
### 2.2.C nRMSE convergence
"""

# ╔═╡ 5bd976b8-1984-4c74-aa65-bd22f6113c5f
begin
	ax_c_S_transfer_method = Axis(
		g_c_S_transfer_method[1,1],
		aspect=1,
		xscale=log10,
		xlabel="N samples", ylabel="nRMSE",
	)
	NRMSES = []
	for (i,teacher) in enumerate(FISH)
		for (j,student) in enumerate(FISH)
			if teacher == student
				nrmse = h5read(inpathtransfer, "Teacher_$i/nrmse")
			else
				nrmse = h5read(inpathtransfer, "Teacher_$i/Student_$j/nrmse")
			end
			push!(NRMSES, nrmse)
		end
	end

	for nrmse in NRMSES
		lines!(
			ax_c_S_transfer_method,
			n_samples_vhv, nrmse, 
			color=(:black, 0.1),
		)
	end
	ylims!(ax_c_S_transfer_method, 0, 1)
end

# ╔═╡ d768b71b-07ad-4881-bbea-aae1491217b3


# ╔═╡ 1a70076c-4915-4a13-92c1-d9a8e535ed34
md"""
### 2.2.D nRMSE matrix
"""

# ╔═╡ b77e8998-695d-4a1e-8f5d-1f7844f03ebc
begin
	ax_d_S_transfer_method = Axis(
		g_d_S_transfer_method[1,1],aspect=1,
	    xticks=((1:length(FISH)), FISH_DISP),
	    yticks=((1:length(FISH)), FISH_DISP),
		xticklabelrotation = π/4,
		xlabel = "Teacher", ylabel="Student"
	)
	
	nrmses_best_mat = Matrix{Float64}(undef, length(FISH), length(FISH))
	for (i,teacher) in enumerate(FISH)
		for (j,student) in enumerate(FISH)
			if teacher == student
				nrmse = h5read(inpathtransfer, "Teacher_$i/nrmse")
			else
				nrmse = h5read(inpathtransfer, "Teacher_$i/Student_$j/nrmse")
			end
			nrmses_best_mat[i,j] = nrmse[end]
		end
	end

	heatmap!(
		ax_d_S_transfer_method,
		nrmses_best_mat, 
		colormap=CONV.CMAP_GOODNESS, colorrange=(0,1),
	)

	Colorbar(
		g_d_S_transfer_method[1,2], 
		colormap=CONV.CMAP_GOODNESS, colorrange=(0,1),
	)
end

# ╔═╡ c8385977-8348-4627-922b-88c7e53a9489


# ╔═╡ 7841fa7e-8d31-47ee-bcda-d3a635617a06
md"""
### 2.2.E Delta joint plot
"""

# ╔═╡ a78d008f-c6c3-4ff6-b36c-e5009f0e6ce4
begin
	vs_trans, Δs_trans = transfer_joint(EXteacher, EXstudent, N_t=10)
	
	ax_e_S_transfer_method_joint = Axis(
		g_e_S_transfer_method[2,1],
		ylabel=L"\mathbb{E}_{\mathbf{v}_S|\mathbb{E}_{\mathbf{h}_T|\mathbf{v}_T}} \mathbf{v}_S",
		xlabel=L"\Delta",
	)
	ax_e_S_transfer_method_Δ = Axis(
		g_e_S_transfer_method[1,1], 
		yscale=log10, 
		ylabel="Density", 
		xticklabelsvisible=false
	)
	ax_e_S_transfer_method_vs = Axis(
		g_e_S_transfer_method[2,2], 
		xscale=log10, 
		xlabel="Density", 
		yticklabelsvisible=false
	)

	linkyaxes!(ax_e_S_transfer_method_joint, ax_e_S_transfer_method_vs)
	linkxaxes!(ax_e_S_transfer_method_joint, ax_e_S_transfer_method_Δ)
	colsize!(g_e_S_transfer_method, 1, Relative(3/4))
	rowsize!(g_e_S_transfer_method, 2, Relative(3/4))

	hexbin!(
		ax_e_S_transfer_method_joint,
		vec(Δs_trans), vec(vs_trans), 
		bins=100, 
		colorscale=log10, colormap=:inferno,
	)

	hist!(
		ax_e_S_transfer_method_Δ,
		vec(Δs_trans), 
		color=:black, 
		bins=100, 
		normalization=:density
	)

	hist!(
		ax_e_S_transfer_method_vs,
		vec(vs_trans), 
		color=:black, 
		bins=100, 
		normalization=:density, 
		direction=:x
	)
	
end

# ╔═╡ 7f4af0e2-91bb-4c69-b93f-44c194d5b9c1


# ╔═╡ 76d508a2-1b7f-416a-961d-6e3fe5a9c1d6
md"""
### 2.2.END Adjustments
"""

# ╔═╡ dac69c70-13d2-4e4e-a388-2b7ebf02af09
all_axes_S_transfer_method = [ax for ax in fig_S_transfer_method.content if typeof(ax)==Axis];

# ╔═╡ 22b71877-1c74-4b7c-ad0b-b21b9321de90
for ax in all_axes_S_transfer_method
	ax.alignmode = Mixed(left=0)
end

# ╔═╡ a962118d-6a38-4a5d-9315-fb2e31e8d9f3
fig_S_transfer_method

# ╔═╡ dd90e2ec-2fd9-47b4-9904-c994c02c9174
# ╠═╡ disabled = true
#=╠═╡
save(@figpath("SUPP_transfer_method"), fig_S_transfer_method)
  ╠═╡ =#

# ╔═╡ 7b47957e-471a-4c6a-b6de-77128a46b5c4


# ╔═╡ 1030de36-8c3f-42b9-829f-a4f799962a3c
md"""
## 2.3. Frame Distances
"""

# ╔═╡ fcfe9c88-993d-4bda-9111-a2617c295970
# ╠═╡ disabled = true
#=╠═╡
save(@figpath("SUPP_all_rhos"), fig_all_rhos)
  ╠═╡ =#

# ╔═╡ 0d3bcc75-b014-41c1-8385-178005a917d3


# ╔═╡ e3cc5a3c-6eac-4f81-9790-f8453ab4ed6e
md"""
# 3. Tools
"""

# ╔═╡ 461d1707-91ca-4881-b0b1-862d30049fd5
md"""
## 3.1. Statistics
"""

# ╔═╡ 4e860cbc-f97c-4696-bedf-4e0807b2f83b
stats_in_path = LOAD.load_misc("DeepFakeStats_$(length(FISH))fish_$(base_mod[3:end])")

# ╔═╡ 422520f0-4866-4cf1-aa1f-e7cbc6843cf2
function stats_loader_nrmse(moment::String, rtype::String)
	@assert moment in ["<v>", "<vv>"]
	@assert rtype in ["samp", "ST", "TS", "STS", "TST"]
	return h5read(stats_in_path, "$moment/nrmse_$rtype")
end

# ╔═╡ ad769b6e-139a-4913-84d1-58ef0c1f0e54
begin
	for (j,r) in enumerate(["samp", "TS", "ST"])
		for (i,m) in enumerate(["<v>", "<vv>"])
			ax = Axis(
				g_c[j,i],
				aspect=1,
				xticks=1:length(FISH), yticks=1:length(FISH),
				xticklabelsvisible=false, yticklabelsvisible=false, 
			)
			heatmap!(ax,
				stats_loader_nrmse(m, r), 
				colormap=CONV.CMAP_GOODNESS, colorrange=(0,1),
			)
		end
	end
	
	ax_ticks = g_c.content[6].content
	ax_ticks.xlabel = "Teacher"
	ax_ticks.ylabel = "Student"
	ax_ticks.xticks = ((1:length(FISH)), FISH_DISP)
	ax_ticks.yticks = ((1:length(FISH)), FISH_DISP)
	ax_ticks.xticklabelsvisible = true
	ax_ticks.yticklabelsvisible = true
	ax_ticks.xticklabelrotation = π/4

	Label(g_c[1,1, Top()],
		"⟨v⟩",
		padding = (0, 0, 0, 0), 
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		font=Makie.current_default_theme().Axis.titlefont.val,
	)
	Label(g_c[1,2, Top()],
		"⟨vv⟩ - ⟨v⟩⟨v⟩",
		padding = (0, 0, 0, 0), 
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		font=Makie.current_default_theme().Axis.titlefont.val,
	)
	Colorbar(g_cCB[1,1], 
			 colormap=CONV.CMAP_GOODNESS, colorrange=(0,1),
			 label="nRMSE",
			 height=Relative(1/3),
	)
end

# ╔═╡ b3157753-aa9d-473f-a948-0d3c9f94d705
begin
	function get_teacher_student(
		f::HDF5.File, 
		teacher_i::Int, 
		student_j::Int,
		moment::String,
		rtype::String,
	)
		@assert moment ∈ ["<v>", "<vv>"]
		@assert rtype ∈ ["samp", "TS", "ST", "TST", "STS"]
	
		if (teacher_i == student_j)
			if (rtype == "samp")
				# looking at self-teacher
				m1 = f["$moment/Teacher_$teacher_i/data"]
				m2 = f["$moment/Teacher_$teacher_i/samp"]
			else
				# impossible
				@warn "impossible request i=$teacher_i j=$student_j type=$rtype"
				return nothing, nothing
			end
			
		elseif rtype == "samp"
			m1 = f["$moment/Teacher_$teacher_i/Student_$student_j/data"]
			m2 = f["$moment/Teacher_$teacher_i/Student_$student_j/samp"]
			
		elseif rtype ∈ ["TS", "ST", "TST", "STS"]
			end_ = rtype[end]
			if end_ == 'T'
				m1 = f["$moment/Teacher_$teacher_i/data"]
			else
				m1 = f["$moment/Teacher_$teacher_i/Student_$student_j/data"]
			end
			m2 = f["$moment/Teacher_$teacher_i/Student_$student_j/$rtype"]
				
		else
			@warn "impossible request i=$teacher_i j=$student_j type=$rtype"
			return nothing, nothing
		end
		return read(m1), read(m2)
	end

	function get_teacher_student(
			fpath::String, 
			teacher::String, 
			student::String,
			moment::String,
			rtype::String,
		)
		flist = h5read(fpath, "fish_list")
		teacher_i = findall(flist.==teacher)[1]
		student_j = findall(flist.==student)[1]
		f = h5open(fpath, "r")
		m1, m2 = get_teacher_student(
			f,
			teacher_i,
			student_j,
			moment,
			rtype,
		)
		close(f)
		return m1, m2
	end
end

# ╔═╡ 2b3f6116-131f-4335-882c-fbc6dab96006
begin
	for (j,r) in enumerate(["samp", "TS", "ST"])
		for (i,m) in enumerate(["<v>", "<vv>"])
			ax = Axis(
				g_b[j,i],
				aspect=1,
				xticklabelsvisible=false, yticklabelsvisible=false, 
			)
			idplotter!(
				ax,
				get_teacher_student(
					stats_in_path, 
					EXteacher, 
					EXstudent, 
					m, 
					r
				)...,
				
			)
		end
	end
	
	g_b.content[6].content.xlabel = "Data"
	g_b.content[6].content.ylabel = "Sampled"
	g_b.content[7].content.xticklabelsvisible=true
	g_b.content[7].content.yticklabelsvisible=true
	g_b.content[6].content.xticklabelsvisible=true
	g_b.content[6].content.yticklabelsvisible=true

	linkaxes!([g_b.content[i].content for i in [2,4,6]]...)
	linkaxes!([g_b.content[i].content for i in [3,5,7]]...)

	Label(g_b[1,1, Top()],
		"⟨v⟩",
		padding = (0, 0, 0, 0), 
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		font=Makie.current_default_theme().Axis.titlefont.val,
	)
	Label(g_b[1,2, Top()],
		"⟨vv⟩ - ⟨v⟩⟨v⟩",
		padding = (0, 0, 0, 0), 
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		font=Makie.current_default_theme().Axis.titlefont.val,
	)
	
	Label(g_b[1,1, Left()],
		"S → S",
		padding = (0, 50, 0, 0), 
		rotation = pi/2,
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		font=Makie.current_default_theme().Axis.titlefont.val,
	)
	Label(g_b[2,1, Left()],
		"T → S",
		padding = (0, 50, 0, 0), 
		rotation = pi/2,
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		font=Makie.current_default_theme().Axis.titlefont.val,
	)
	Label(g_b[3,1, Left()],
		"S → T",
		padding = (0, 50, 0, 0), 
		rotation = pi/2,
		fontsize=Makie.current_default_theme().Axis.titlesize.val,
		font=Makie.current_default_theme().Axis.titlefont.val,
	)

	Colorbar(g_bCB[1,1], 
		colormap=:inferno, colorrange=(0,1),
		label="PLACE HOLDER",
		height=Relative(1/3),
	)
end

# ╔═╡ bf76e545-d6ec-4562-a0f2-d8fcd5b34fee
ex_mT, ex_mS = get_teacher_student(
	stats_in_path, 
	FISH[6], 
	FISH[3], 
	"<v>", 
	"TS"
);

# ╔═╡ 8b368e58-20f7-4a10-bae0-185968181afc
begin
	# figtest = Figure()
	ax_a_S_bad_residuals = Axis(
		g_a_S_bad_residuals[1,1], 
		aspect=1, 
		xlabel="⟨vᵀ⟩", ylabel="⟨vˢ→T⟩",
	)
	h_id = idplotter!(
		ax_a_S_bad_residuals,
		ex_mT, ex_mS
	)
	Colorbar(g_a_S_bad_residuals[1,2], colormap=h_id.colormap.val, colorrange=h_id.colorrange.val, scale=log10, height=Relative(0.7), label="Density")
	colgap!(g_a_S_bad_residuals, 5)
	# figtest
end

# ╔═╡ b3b7ad5d-fb01-4855-bd43-fcb93cd0cb6a
function get_teacher_student_residual(
		fpath::String, 
		teacher::String, 
		student::String,
		moment::String,
		rtype::String,
	)
	flist = h5read(fpath, "fish_list")
	teacher_i = findall(flist.==teacher)[1]
	student_j = findall(flist.==student)[1]
	f = h5open(fpath, "r")
	m1, m2 = get_teacher_student(
		f,
		teacher_i,
		student_j,
		moment,
		rtype,
	)
	close(f)

	
	return abs.(m1 .- m2)
	end

# ╔═╡ 436156e1-6756-41f2-aacb-60e74ce251c6
begin
	frac_high_residuals = Matrix{Float64}(undef, length(FISH), length(FISH))
	for (i,teacher) in enumerate(FISH)
		for (j,student) in enumerate(FISH)
			if student == teacher
				frac_high_residuals[i,j] = NaN
			else
				frac_high_residuals[i,j] = mean(get_teacher_student_residual(
					stats_in_path, 
					teacher, 
					student, 
					"<v>", 
					"TS"
				)[:,1] .> 0.1 )
			end
		end
	end
end

# ╔═╡ 52c2d641-a082-4786-95a6-0e1cc9230d57
frac_high_residuals .> 0.05

# ╔═╡ 9db4ebe0-9c1c-4e7f-a352-f62792e7a19d
begin
	ax_b_S_bad_residuals = Axis(g_b_S_bad_residuals[1,1], aspect=1,
	    xticks=((1:length(FISH)), FISH_DISP),
	    yticks=((1:length(FISH)), FISH_DISP),
		xticklabelrotation = π/4,
		xlabel = "Teacher", ylabel="Student",)
	h_high_residuals = heatmap!(
		ax_b_S_bad_residuals,
		frac_high_residuals.*100, 
		colormap=:cividis, 
		colorrange=(0,maximum(frac_high_residuals[isfinite.(frac_high_residuals)])).*100
	)
	Colorbar(
		g_b_S_bad_residuals[1,2], 
		h_high_residuals, 
		label="% |⟨v⟩ᵀ - ⟨v⟩ˢ→T| > 0.1",
		# vertical=false, flipaxis=false,
		# width=Relative(0.7),
		height=Relative(0.7)
	)
	colgap!(g_b_S_bad_residuals, 5)
end

# ╔═╡ 75bb5736-c7de-493e-87b4-a0a9759b3d8f
"""
    topk_cartesianindices(A::AbstractMatrix, k::Integer)
Return a `Vector{CartesianIndex{2}}` containing the indices of the `k`
largest finite entries of `A`, ordered from largest to smallest.
"""
function topk_cartesianindices(A::AbstractMatrix, k::Integer)
    @assert ndims(A) == 2 && size(A,1) == size(A,2) "A must be square"
    @assert k ≥ 1 "k must be positive"

    # linear indices of finite (non-NaN) entries
    finite_mask = .!isnan.(A)
    lin_idxs    = findall(finite_mask)
    vals        = @view A[lin_idxs]

    k   = min(k, length(vals))
    sel = partialsortperm(vals, 1:k; rev = true)

    # return CartesianIndex objects
    return CartesianIndices(A)[lin_idxs[sel]]
end

# ╔═╡ 622c7000-a631-4f01-9cb4-066b774dd60c
top_inds = topk_cartesianindices(frac_high_residuals, 12)

# ╔═╡ 9110ba04-10c5-41de-95bc-53fc05e4b136
begin	
	axs = []
	for i in 1:2
		for j in 1:6
			ax = Axis(
				g_c_S_bad_residuals[i,j],
				aspect=DataAspect(),
		        bottomspinevisible=false, leftspinevisible=false,
		        topspinevisible=false, rightspinevisible=false,
		        xticklabelsvisible=false, yticklabelsvisible=false,
		        xticksvisible=false, yticksvisible=false,
			)
			push!(axs, ax)
		end
	end
	linkaxes!(axs...)

	for i in 1:length(axs)
		teacher_i, student_j = Tuple(top_inds[i])
		teacher, student = FISH[teacher_i], FISH[student_j]
		
		coords = load_data(LOAD.load_dataWBSC(student)).coords
		R = get_teacher_student_residual(
			stats_in_path, 
			teacher, 
			student, 
			"<v>", 
			"TS"
		)[:,1]

		neuron2dscatter!(
			axs[i],
			coords[:,1], coords[:,2],
			R,
			cmap=cmap_ainferno(),
			range=(0,1),
			edgewidth=0.03, radius=1.0,
			rasterize=5,
			edgecolor=(:black, 0.3)
		)

		text!(
			axs[i],
			450, 950, 
			text="$(round(mean(R.>0.1).*100, sigdigits=2))% > 0.1",
			align=(:right, :bottom),
			fontsize=Makie.current_default_theme().Axis.xticklabelsize.val
		)
	end

	Colorbar(
		g_c_S_bad_residuals[2,7], 
		colormap=cmap_ainferno(), 
		colorrange=(0,1), 
		label="|⟨v⟩ᵀ - ⟨v⟩ˢ→T|",
		height=Relative(0.5)
	)

	text!(axs[7], 150, 10, text="200μm", align=(:center, :bottom))
	lines!(axs[7], [50,250], [0,0], color=:black)
end

# ╔═╡ fbee7b14-f1ba-40f7-8fcc-5e11282250ad
begin
	function load_residual_cause(teacher::String, student::String)
		r = get_teacher_student_residual(
			stats_in_path, 
			teacher, 
			student, 
			"<v>", 
			"TS"
		)[:,1]
		w = load_brainRBM(LOAD.load_wbscRBM(
			"biRBMs", 
			"biRBM_$(student)_FROM_$(teacher)$(base_mod)"
		))[1].w
		return sort(abs.(r'*w)[1,:], rev=true)
		# a = abs.(r'*w)[1,:]
		# println(size(r))
		# b = abs.((ones(size(r)) ./ length(r))'*w)[1,:]
		# return sort(a./b, rev=true)
	end
	function load_residual_cause(teacher_i::Int, student_j::Int)
		teacher, student = FISH[teacher_i], FISH[student_j]
		return load_residual_cause(teacher, student)
	end
end

# ╔═╡ 908db432-d1f1-45e2-8dea-d515d82fc03c
begin
	ax_d_S_bad_residuals = Axis(
		g_d_S_bad_residuals[1,1], 
		yscale=log10,
		xlabel="Ranked hidden unit μ",
		ylabel="∑ᵢ |⟨v⟩ᵀ - ⟨v⟩ˢ→T|ᵢ wᵢ_μ"
	)
	for i in 1:length(top_inds)
		lines!(ax_d_S_bad_residuals, load_residual_cause(Tuple(top_inds[i])...), color=(:black, 0.25))
	end
	ylims!(ax_d_S_bad_residuals, 1.e-2, 1.e+4)
end

# ╔═╡ d4a1e72c-b82d-457c-b569-dfb84b8d7474


# ╔═╡ aa8e41b8-b94d-4238-a140-701051dcd347


# ╔═╡ 04df0c51-de25-4e06-b497-638bd85db5a2
md"""
## 3.2. Visible activity
"""

# ╔═╡ 4a1d3dff-2c11-4af5-82e0-446016730b53
begin
	function get_teacher_student_act(
			fpath::String, 
			teacher_i::Int, 
			student_j::Int,
			rtype::String,
		)
		@assert rtype ∈ ["TT", "SS", "TS", "ST"]
		if rtype == "TT"
			return h5read(fpath, "Teacher_$teacher_i/TT")
		else
			if teacher_i == student_j
				throw("teacher == student but $rtype was requested")
			end
			return  h5read(fpath, "Teacher_$teacher_i/Student_$student_j/$rtype")
		end
	end
	
	function get_teacher_student_act(
				fpath::String, 
				teacher::String, 
				student::String,
				rtype::String,
			)
			flist = h5read(fpath, "fish_list")
			teacher_i = findall(flist.==teacher)[1]
			student_j = findall(flist.==student)[1]
			a = get_teacher_student_act(
				fpath,
				teacher_i,
				student_j,
				rtype,
			)
			return a
		end
end

# ╔═╡ 26de244c-8d03-4fde-b427-fe6cb83e15a0
function get_teacher_student_correction(
		fpath::String, 
		teacher::String, 
		student::String,
		rtype::String,
	)
	@assert rtype ∈ ["TT", "SS", "TS", "ST"]
	flist = h5read(fpath, "fish_list")
	teacher_i = findall(flist.==teacher)[1]
	student_j = findall(flist.==student)[1]
	f = h5open(fpath, "r")

	if rtype == "TT"
		stype = "samp"
		student_j = teacher_i
	elseif rtype == "SS"
		stype = "samp"
	else
		stype = rtype
	end

	
	m1, m2 = get_teacher_student(
		f,
		teacher_i,
		student_j,
		"<v>",
		stype,
	)
	close(f)
	return (m1 .- m2)[:,1]
	end

# ╔═╡ 63d870b5-5bbc-4566-99d9-c4dddf95d397


# ╔═╡ 442b9e95-edd4-4cf7-bdf6-c7fb97a52d9d
md"""
## 3.3. Free Energy
"""

# ╔═╡ 55e4ed24-77aa-43b3-a71f-875671050737


# ╔═╡ e5ff5ed2-f45d-4f9a-905d-00aba7f51ff3
md"""
## 3.4. Transfer Method
"""

# ╔═╡ 2d2dc1e6-3598-4374-830f-4ecbb997180b


# ╔═╡ 7ceee459-625d-429f-a2bc-fedf9030c4e9


# ╔═╡ d89d4b23-4b51-4923-9f25-93c7145450ed
md"""
## 3.5. Maps distance
"""

# ╔═╡ ab4519f4-6798-49cb-a593-7d118a048b18
inpathactdist = LOAD.load_misc("DeepFakeActivityDistance_$(length(FISH))fish_$(base_mod[3:end])")

# ╔═╡ aeb8f7c4-feba-4afa-ad57-19887379af1c
begin
	function load_map_rho(h5path::String, teacher::String, student::String, rtype::String)
		@assert rtype ∈ ["rhoT_", "rho_Tshuff", "rho_S", "rho_Sshuff"]
		
		teacher_i = findall(h5read(h5path, "fish_list") .== teacher)[1]
		student_j = findall(h5read(h5path, "fish_list") .== student)[1]
	
		return h5read(h5path, "Teacher_$teacher_i/Student_$student_j/$rtype")
	end
	function load_map_rho(h5path::String, fish::Vector{String}, rtype::String)
		@assert rtype ∈ ["rhoT_", "rho_Tshuff", "rho_S", "rho_Sshuff"]
		a = Matrix{Float64}(undef, length(fish), length(fish))
		for (i, t) in enumerate(fish)
			for (j, s) in enumerate(fish)
				if i==j
					a[i,j] = NaN
				else
					a[i,j] = median(load_map_rho(h5path, t, s, rtype))
				end
			end
		end
		return a
	end
end

# ╔═╡ e5aceebc-8fa5-45dd-b623-c2980790b181
begin
	# fig_dist_mat = Figure(size=dfsize().*(2.3,1.3))
	
	ax_rhodTS = Axis(
		g_g[1,1], aspect=1,
	    xticks=((1:length(FISH)), FISH_DISP),
	    yticks=((1:length(FISH)), FISH_DISP),
		xticklabelrotation = π/4,
		xlabel = "Teacher", ylabel="Student",
		title="T→S"
	)
	heatmap!(
		ax_rhodTS,
		load_map_rho(inpathactdist, FISH, "rhoT_"), 
		colormap=:seismic, 
		colorrange=(-1,+1)
	)
	
	
	ax_rhodST = Axis(
		g_g[1,2], aspect=1,
	    xticks=((1:length(FISH)), FISH_DISP),
	    yticks=((1:length(FISH)), FISH_DISP),
		xticklabelrotation = π/4,
		# xlabel = "Teacher", 
		# ylabel="Student",
		yticklabelsvisible=false,
		xticklabelsvisible=false,
		title="S→T"
	)
	heatmap!(
		ax_rhodST,
		load_map_rho(inpathactdist, FISH, "rho_S"), 
		colormap=:seismic, 
		colorrange=(-1,+1)
	)

	Colorbar(
		g_gCB[1,1], 
		colormap=:seismic, 
		colorrange=(-1,1),
		label=L"\text{median}_t(\rho)"
	)
	
	# fig_dist_mat
end

# ╔═╡ 137aa2ca-0557-467d-a86c-bc804a3716da
begin
	# fig_dist_ex = Figure(size=dfsize())
	ax_dist_ex = Axis(
		g_f[1,1],
		xlabel="ρ",
		ylabel="Density"
	)
	ρshuff = vcat(load_map_rho(inpathactdist, EXteacher, EXstudent, "rho_Sshuff"), load_map_rho(inpathactdist, EXteacher, EXstudent, "rho_Tshuff"))
	ρT = load_map_rho(inpathactdist, EXteacher, EXstudent, "rhoT_")
	ρS = load_map_rho(inpathactdist, EXteacher, EXstudent, "rho_S")
	# ρshuff = vcat(load_map_rho(inpathactdist, FISH[4], FISH[3], "rho_Sshuff"), load_map_rho(inpathactdist, FISH[4], FISH[3], "rho_Tshuff"))
	# ρT = load_map_rho(inpathactdist, FISH[4], FISH[3], "rhoT_")
	# ρS = load_map_rho(inpathactdist, FISH[4], FISH[3], "rho_S")
	
	# ρSshuff = vcat([load_map_rho(inpathactdist, t, s, "rho_Sshuff") for t∈FISH, s∈FISH if t!=s]...)	
	# ρTshuff = vcat([load_map_rho(inpathactdist, t, s, "rho_Tshuff") for t∈FISH, s∈FISH if t!=s]...)
	# ρshuff = vcat(ρSshuff, ρTshuff)
	# ρT = vcat([load_map_rho(inpathactdist, t, s, "rhoT_") for t∈FISH, s∈FISH if t!=s]...)
	# ρS = vcat([load_map_rho(inpathactdist, t, s, "rho_S") for t∈FISH, s∈FISH if t!=s]...)
	

	density!(ax_dist_ex, ρshuff, color=(:grey, 0.5))
	density!(ax_dist_ex, ρT, color=(:white, 0), strokecolor=CONV.COLOR_TEACHER, strokewidth=3)
	density!(ax_dist_ex, ρS, color=(:white, 0), strokecolor=CONV.COLOR_STUDENT, strokewidth=3)

	xlims!(ax_dist_ex, 0,1)

	# fig_dist_ex
end

# ╔═╡ 3416de92-6c8e-423e-95d9-adf8d2be8b0d
begin
	fig_all_rhos = Figure(size=dfsize().*3)
	Axis(
		fig_all_rhos[1,1],
		bottomspinevisible=false, xticksvisible=false, xticklabelsvisible=false,
		leftspinevisible=false, yticksvisible=false, yticklabelsvisible=false,
	)
	for (j, teacher) in enumerate(FISH)
		for (i, student) in enumerate(FISH)
			if teacher == student
				continue
			end
			ρshuff = vcat(load_map_rho(inpathactdist, teacher, student, "rho_Sshuff"), load_map_rho(inpathactdist, teacher, student, "rho_Tshuff"))
			ρT = load_map_rho(inpathactdist, teacher, student, "rhoT_")
			ρS = load_map_rho(inpathactdist, teacher, student, "rho_S")
	
			oi = i*6
			oj = j*(1.1)
			density!(ρshuff .+ oj, color=(:grey, 0.5), offset=oi)
			density!(ρT.+ oj, color=(:white, 0), strokecolor=CONV.COLOR_TEACHER, strokewidth=3, offset=oi)
			density!(ρS .+ oj, color=(:white, 0), strokecolor=CONV.COLOR_STUDENT, strokewidth=3, offset=oi)

			lines!([0,1].+oj, [0,0].+oi, color=:black)
			lines!([0,0].+oj, [0.05,-0.25].+oi, color=:black)
			lines!([1,1].+oj, [0.05,-0.25].+oi, color=:black)
			lines!([0.5,0.5].+oj, [0.05,-0.25].+oi, color=:black)
		end
	end
	text!(0.0+(1*(1.1)), -0.25+(2*6), text="0",align=(:center, :top),fontsize=Makie.current_default_theme().Axis.xticklabelsize.val)
	text!(0.5+(1*(1.1)), -0.25+(2*6), text="0.5",align=(:center, :top), fontsize=Makie.current_default_theme().Axis.xticklabelsize.val)
	text!(1.0+(1*(1.1)), -0.25+(2*6), text="1",align=(:center, :top), fontsize=Makie.current_default_theme().Axis.xticklabelsize.val)
	text!(0.5+(1*(1.1)), -1+(2*6), text="ρ",align=(:center, :top), fontsize=Makie.current_default_theme().Axis.xlabelsize.val)

	
	for (i,fish) in enumerate(FISH_DISP)
		text!(
			1,i*6+2,
			text=fish, 
			font=Makie.current_default_theme().Axis.titlefont.val,
			fontsize=Makie.current_default_theme().Axis.titlesize.val,
			align=(:right, :center)
		)
		text!(
			i*(1.1)+0.5,5,
			text=fish, 
			font=Makie.current_default_theme().Axis.titlefont.val,
			fontsize=Makie.current_default_theme().Axis.titlesize.val,
			align=(:center, :top)
		)
	end
	
	fig_all_rhos
end

# ╔═╡ 695a67fd-7bdf-4c24-8762-0904304b4124
function load_map_Signif(h5path::String, fish::Vector{String}, rtype::String)
	@assert rtype ∈ ["T", "S"]
	if rtype == "T"
		rtype1 = "rhoT_"
		rtype2 = "rho_Tshuff"
	else
		rtype1 = "rho_S"
		rtype2 = "rho_Sshuff"
	end
	a = Matrix{Float64}(undef, length(fish), length(fish))
	for (i, t) in enumerate(fish)
		for (j, s) in enumerate(fish)
			if i==j
				a[i,j] = NaN
			else
				ρ1 = load_map_rho(h5path, t, s, rtype1)
				ρ2 = load_map_rho(h5path, t, s, rtype2)
				a[i,j] = pvalue(MannWhitneyUTest(ρ1, ρ2), tail=:right)
			end
		end
	end
	return a
end

# ╔═╡ 12c079b0-74f1-4958-83c9-edb626d3a0dc


# ╔═╡ Cell order:
# ╟─554a0fb4-38ef-11f0-10d3-8bf0f5e7002d
# ╠═a088a9f3-b96c-49dd-8266-be67e657a4e0
# ╠═ce99e7e7-0a15-4afe-b1ff-7768a0d9c6e8
# ╠═bbe5511a-ae48-4afc-8901-dd5467358e2a
# ╠═fc7fc50c-9d9a-4e1d-94a7-26690ac496c6
# ╠═db58397d-3121-48ad-83b9-cfd896e95b9e
# ╠═14d3ee1c-0343-4630-800f-6c00926c99de
# ╟─cbe1e272-76ba-45aa-bc89-7109664153f2
# ╠═2d7880d8-84ad-45ed-ae3d-c5664f320da6
# ╟─9ed90752-edd7-4f4b-93c1-99d3301f2b8d
# ╠═956c18fe-276f-44e2-a6a6-6720ea47fb4b
# ╠═b54f03ba-5cc3-433b-b494-e5cd904e78ab
# ╟─a31f926e-fc44-4f30-8f31-e92dbf487977
# ╟─37e5c5b3-352b-4bbb-9831-aa02aa923d63
# ╟─90145bf6-353c-4c83-8627-31f59f7cb338
# ╟─8aef390a-9c77-44f9-a966-e4488253a72b
# ╠═ec1bd161-3868-4505-84e7-e1572a38da46
# ╟─02b71d98-90a7-401c-933d-83f0c8f1183e
# ╠═9a9ad7d5-80ed-4de5-8c72-d1870954980c
# ╟─0d6a15d8-2cc1-4f1e-a13a-57b9644c4ac1
# ╠═2b3f6116-131f-4335-882c-fbc6dab96006
# ╠═07059f14-2432-45cf-9b0e-88847942f0f1
# ╟─44013910-35c4-4614-862d-87361a10b47d
# ╠═ad769b6e-139a-4913-84d1-58ef0c1f0e54
# ╠═8c4ed0eb-d052-4bb6-a2a0-12e463c600c2
# ╟─53a203c5-313d-4b53-ba77-ec238bc22ad5
# ╠═a198bb85-d365-4b36-9975-84c94e8bd7c8
# ╠═79dc3ae7-2fbd-4718-ac40-7442389c70c6
# ╠═71c3cc8c-656f-467f-9522-7bb3bf5a2977
# ╠═75921dee-499c-4930-bff4-c71cf5d63e89
# ╟─bef64e9f-067d-4676-99c4-30b8fdc3fad2
# ╠═bf857ba1-f41d-4e0c-b0d5-9defbb3373cc
# ╠═bf1f4985-3062-4ee9-84db-46153c491b82
# ╠═bb96c64c-e83c-47f2-a4d6-6667e946b36a
# ╠═0ff9016e-d7a4-4ccf-806e-86a28f2daa21
# ╠═1e9f1588-fa0c-4886-bf2d-81cf65daf121
# ╠═de4c7c6a-b09b-4805-a139-4f05b6334347
# ╠═fd54108d-99b3-49b5-a3c3-dfdb2aad83eb
# ╠═5ac7a0e0-a796-415c-a0ca-4626138d1795
# ╟─b0db6f7e-c489-4e96-bbee-3659d6da5fe2
# ╠═e5aceebc-8fa5-45dd-b623-c2980790b181
# ╠═7a71fccc-3fd7-4a4b-82b3-ea4b87158b03
# ╠═90f0dcd4-85c8-4962-999e-1a39eab3f26a
# ╟─210298d8-879d-4067-8954-f8d361306894
# ╠═137aa2ca-0557-467d-a86c-bc804a3716da
# ╠═f2623330-f6ac-445a-9314-81c0eaa512c5
# ╟─435737df-a681-49b9-8d0d-3b0585c23a61
# ╠═89e0109a-8ca7-42ff-8a20-bc3ccb8bf871
# ╠═dcdde4e6-7599-43b3-ab0e-9637e0454d6e
# ╠═e4e39a30-bb6c-4589-b602-101be8917032
# ╠═2106652f-5e59-41d1-8400-1be570bfed6c
# ╠═695fb21c-035c-46a9-8596-ac02bee19fa6
# ╟─417e1a28-3407-4347-b8ae-8f0eb0752ea6
# ╟─2377b684-9eec-4d88-8e41-62d44baafd53
# ╠═a1e06e54-1202-4fc5-9620-a84ad7bae950
# ╠═f857d4e9-725c-4dd4-82a1-86524774e7c0
# ╟─bd71917e-ca95-4245-a9a2-a71244a7167a
# ╠═52c2d641-a082-4786-95a6-0e1cc9230d57
# ╠═bf76e545-d6ec-4562-a0f2-d8fcd5b34fee
# ╠═8b368e58-20f7-4a10-bae0-185968181afc
# ╟─4a21f8b2-10ef-4df0-bb5a-1e42d1b618f9
# ╠═436156e1-6756-41f2-aacb-60e74ce251c6
# ╠═9db4ebe0-9c1c-4e7f-a352-f62792e7a19d
# ╟─30a648ec-2dd8-45b4-98cf-9fd5de916dcd
# ╠═c2a89a06-dab9-4da5-813d-21bcb6777ee3
# ╠═622c7000-a631-4f01-9cb4-066b774dd60c
# ╠═9110ba04-10c5-41de-95bc-53fc05e4b136
# ╠═126a45a9-fe66-4bb7-bcfb-655492c04780
# ╟─ec2cf804-ec40-4f8b-bb56-f1ca68940c0e
# ╠═617245a3-dfcf-47a4-9660-b2bccc5b7238
# ╠═908db432-d1f1-45e2-8dea-d515d82fc03c
# ╠═bdff6eef-3213-4743-a37a-32222c9831f0
# ╟─54e8a938-ffd1-45c6-9bc2-7d04439ee80d
# ╠═2a7960bb-0b21-4270-bfe5-f5766cabafee
# ╠═ade4796e-e7cc-46b1-b520-b85abb37295a
# ╠═6a01c7ec-d1e9-4b7b-9af4-37f3821cbd82
# ╠═57e4fbeb-4c50-45de-a07c-ee5c3cbb852c
# ╠═8f4cb83c-e9e3-4e51-bac7-ffff3ff7f9e0
# ╠═59e629bf-99f5-49b2-bb0b-5d0d052bcc8a
# ╠═e2d36e5b-41dc-43cf-881b-5ba948345662
# ╠═776c1dc0-b096-469c-b44b-f88a53c5dd26
# ╟─70cc2097-7a3f-436f-891d-436fb4effdde
# ╟─872932aa-5f15-423b-a511-0dd9b1332035
# ╟─aa63fa66-7946-4e66-8bd9-089a4e6de405
# ╠═f4a246ab-60bd-4db9-828a-5eca0149aabd
# ╟─acd47348-5df8-4a04-a518-e23e4315f5e8
# ╠═75fe17c1-4f45-4d76-a403-c87c268250c4
# ╠═9a336639-20fe-4c9f-85ca-52f412be7199
# ╟─86c4e793-c7d2-4284-97f4-7d87366dd75d
# ╠═5bd976b8-1984-4c74-aa65-bd22f6113c5f
# ╠═d768b71b-07ad-4881-bbea-aae1491217b3
# ╟─1a70076c-4915-4a13-92c1-d9a8e535ed34
# ╠═b77e8998-695d-4a1e-8f5d-1f7844f03ebc
# ╠═c8385977-8348-4627-922b-88c7e53a9489
# ╟─7841fa7e-8d31-47ee-bcda-d3a635617a06
# ╠═a78d008f-c6c3-4ff6-b36c-e5009f0e6ce4
# ╠═7f4af0e2-91bb-4c69-b93f-44c194d5b9c1
# ╟─76d508a2-1b7f-416a-961d-6e3fe5a9c1d6
# ╠═dac69c70-13d2-4e4e-a388-2b7ebf02af09
# ╠═22b71877-1c74-4b7c-ad0b-b21b9321de90
# ╠═a962118d-6a38-4a5d-9315-fb2e31e8d9f3
# ╠═dd90e2ec-2fd9-47b4-9904-c994c02c9174
# ╠═7b47957e-471a-4c6a-b6de-77128a46b5c4
# ╟─1030de36-8c3f-42b9-829f-a4f799962a3c
# ╠═3416de92-6c8e-423e-95d9-adf8d2be8b0d
# ╠═fcfe9c88-993d-4bda-9111-a2617c295970
# ╠═0d3bcc75-b014-41c1-8385-178005a917d3
# ╟─e3cc5a3c-6eac-4f81-9790-f8453ab4ed6e
# ╟─461d1707-91ca-4881-b0b1-862d30049fd5
# ╠═4e860cbc-f97c-4696-bedf-4e0807b2f83b
# ╠═422520f0-4866-4cf1-aa1f-e7cbc6843cf2
# ╠═b3157753-aa9d-473f-a948-0d3c9f94d705
# ╠═b3b7ad5d-fb01-4855-bd43-fcb93cd0cb6a
# ╠═75bb5736-c7de-493e-87b4-a0a9759b3d8f
# ╠═fbee7b14-f1ba-40f7-8fcc-5e11282250ad
# ╠═d4a1e72c-b82d-457c-b569-dfb84b8d7474
# ╠═aa8e41b8-b94d-4238-a140-701051dcd347
# ╟─04df0c51-de25-4e06-b497-638bd85db5a2
# ╠═7b88bfed-2dfe-41c8-9859-8376e3687081
# ╠═4a1d3dff-2c11-4af5-82e0-446016730b53
# ╠═26de244c-8d03-4fde-b427-fe6cb83e15a0
# ╠═63d870b5-5bbc-4566-99d9-c4dddf95d397
# ╟─442b9e95-edd4-4cf7-bdf6-c7fb97a52d9d
# ╠═aa956757-8042-4e15-bf78-abadf74bea8c
# ╠═55e4ed24-77aa-43b3-a71f-875671050737
# ╟─e5ff5ed2-f45d-4f9a-905d-00aba7f51ff3
# ╠═b18a79ba-1cbc-49b6-8f13-767a28c19aee
# ╠═2d2dc1e6-3598-4374-830f-4ecbb997180b
# ╠═7ceee459-625d-429f-a2bc-fedf9030c4e9
# ╟─d89d4b23-4b51-4923-9f25-93c7145450ed
# ╠═ab4519f4-6798-49cb-a593-7d118a048b18
# ╠═aeb8f7c4-feba-4afa-ad57-19887379af1c
# ╠═6f15ddb3-5eec-44c3-8056-96dd23bf0c03
# ╠═695a67fd-7bdf-4c24-8762-0904304b4124
# ╠═12c079b0-74f1-4958-83c9-edb626d3a0dc
