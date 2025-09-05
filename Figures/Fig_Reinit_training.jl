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

# ╔═╡ 3ef22527-b023-4942-b76e-4ee055b1af6b
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ ed7a57fe-b0b3-11ef-2c97-b71673d90a65
begin
	using BrainRBMjulia
	using LinearAlgebra: diagind
	using HDF5
	using Statistics
	
	using CairoMakie
	using BrainRBMjulia: multipolarnrmseplotter!, idplotter!, dfsize, quantile_range, neuron2dscatter!, cmap_aseismic, polarnrmseplotter!
	using ColorSchemes: reverse, RdYlGn_9

	CONV = @ingredients("conventions.jl")
	include(joinpath(dirname(Base.current_project()), "Misc_Code", "fig_saving.jl"))
end

# ╔═╡ ccba55e1-a141-40a3-b8f9-44eac06e5ff2
begin
	using StatsBase: Weights
	using Makie.GeometryBasics
end

# ╔═╡ 62fe2c43-075e-49f4-82f2-7d8e6c923377
using HypothesisTests

# ╔═╡ 8bd05deb-f7bc-4bb8-bfbd-c1f90bd43dda
begin
	using Random
	function eye(n::Int; ϵ=1.e9)
		a = zeros(n,n)
		a[diagind(a)] .= ϵ
		return a
	end
	function nanmax(a::AbstractMatrix; ϵ=-1.e9)
		b = copy(a)
		b[isnan.(b)] .= ϵ
		mb = maximum(b, dims=1)
		mb[mb.==ϵ] .= NaN
		return mb[1,:]
	end
	function nansum(a::AbstractVector)
		return sum(a[isfinite.(a)])
	end
	function nansum(A::AbstractArray; dims=2)
		a = copy(A)
		a[isnan.(a)] .= 0
		return sum(a; dims)
	end
	function greater_than_diag(A::AbstractMatrix)
		dinds = diagind(A)
		EYE = eye(size(A,1))
		return A[dinds] .- nanmax(A .- EYE)
	end
	function greater_than_diag_bootstrap(A::AbstractMatrix; N::Int=5000)
		b = Matrix{Float64}(undef, size(A, 2), N)
		for i in 1:N
			perm = randperm(size(A, 1))
			b[:,i] .= greater_than_diag(A[perm,:])
		end
		return b
	end
	function greater_than_diag_pval(A::AbstractMatrix; N::Int=5000)
		COUNT = greater_than_diag_bootstrap(A;N) .> greater_than_diag(A)
		n_nan_cols = sum([all(isnan.(A[i,:])) for i in 1:size(A,1)])
		return sum(COUNT) / (N*(size(A,1)-n_nan_cols))
	end
	# function greater_than_diag_pval(A::AbstractMatrix; N::Int=5000)
	# 	to = nansum(greater_than_diag(A)) # observed
	# 	tb = nansum(greater_than_diag_bootstrap(A;N), dims=1)[1,:] #bootstrap
	# 	return (1+ sum(tb .>= to)) / (N+1), (1/(N+1))
	# end
end

# ╔═╡ 1f824ece-d1f2-4795-8bd3-fe5f9d98234b
begin
	using StatsBase: median
	nanmean(X) = [mean(X[i,j,:][isfinite.(X[i,j,:])]) for i=1:size(X,1), j=1:size(X,2)]
	nanmedian(X) = [median(X[i,j,:][isfinite.(X[i,j,:])]) for i=1:size(X,1), j=1:size(X,2)]
end

# ╔═╡ 736cb1dd-abe6-401e-86cc-5c97526d4444
md"""
# Imports + Notebook Preparation
"""

# ╔═╡ 2d4bcf99-236e-4291-939d-b3a8f9e6e037
TableOfContents()

# ╔═╡ 64381840-77f7-4af3-9c5d-703d8f4cf468
set_theme!(CONV.style_publication)

# ╔═╡ 6d9eb4c9-66a9-4114-87ec-6d8638a97e4e
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ cd719af5-c4a6-4822-8aee-3a497e55ac24
md"""
# 0. Fish and training base
"""

# ╔═╡ 47be63c0-05a2-4295-938b-33c4d7350058
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ cae5892f-c7a4-4347-9a04-b16acc2531ac
md"Use numbers instead of fish names $(@bind numbered CheckBox(default=true))"

# ╔═╡ 34d666bf-acc9-4433-a38d-fd131116f84a
base_mod = "*_WBSC_M100_l10.02_l2l10";

# ╔═╡ ec439d7a-9a6a-4757-aaee-2c488c0e9d17
for fish in FISH
	println(LOAD.load_dataWBSC(fish))
	println(LOAD.load_wbscRBM("bRBMs", "bRBM_".*fish.*base_mod))
end

# ╔═╡ d2817416-6e1d-49ac-8a81-cc20118f90b5
md"example teacher : $(@bind EXteacher Select(FISH, default=FISH[3]))"

# ╔═╡ 60774e68-5d35-407c-8900-40eed1d04b1d
STUDENTS = [fish for fish in FISH if fish!=EXteacher];

# ╔═╡ bff9fbfc-9ee8-40cd-b6d2-46399951ed00
md"example student : $(@bind EXstudent Select(STUDENTS, default=STUDENTS[1]))"

# ╔═╡ 3d8b112a-d8cc-4618-ba69-33858019fb53
md"example hidden unit : $(@bind EXhu NumberField(0:200, default=65))"

# ╔═╡ 70ed6e1f-4c61-4e1f-a5d7-b775487b564e
begin
	if numbered
		FISH_DISP = ["F$i" for i in 1:length(FISH)]
		STUDENT_DISP = ["Student $i" for i in 1:length(STUDENTS)]
	else
		FISH_DISP = FISH
		STUDENT_DISP = STUDENTS
	end
end;

# ╔═╡ f21f6595-e447-4bad-b796-ccc657cfb298
md"""
# 1. Main Figure
"""

# ╔═╡ fba1912a-9abc-474c-a9e0-3b603555e0d7
begin
	fig_main = Figure(size=(53, 49).*(3.5,4.5).*(4/3/0.35))
	
	g_a   = fig_main[1,1:3] = GridLayout()

	g_bcdef = fig_main[2:4,1:2] = GridLayout()
	g_b = g_bcdef[1,1:2] = GridLayout()
	g_cdef = g_bcdef[2:3,1:2] = GridLayout()
	g_c = g_cdef[1,1] = GridLayout()
	g_d = g_cdef[2,1] = GridLayout()
	g_e = g_cdef[1,2] = GridLayout()
	g_f = g_cdef[2,2] = GridLayout()

	g_ghi = fig_main[2:4, 3] = GridLayout()
	g_g = g_ghi[1,1] = GridLayout()
	g_h = g_ghi[2,1] = GridLayout()
	g_i = g_ghi[3,1] = GridLayout()
	# g_b   = fig_main[2,1] = GridLayout()
	# g_e   = fig_main[2,2:3] = GridLayout()
	# g_cfh = fig_main[3,1:3] = GridLayout()
	# g_dgi = fig_main[4,1:3] = GridLayout()

	# g_c = g_cfh[1,1] = GridLayout()
	# g_f = g_cfh[1,2] = GridLayout()
	# g_h = g_cfh[1,3] = GridLayout()
	
	# g_d = g_dgi[1,1] = GridLayout()
	# g_g = g_dgi[1,2] = GridLayout()
	# g_i = g_dgi[1,3] = GridLayout()

	for (label, layout) in zip(
		["A", "B", "C", "D", "E", "F", "G", "H", "I"], 
		[g_a, g_b, g_c, g_d, g_e, g_f, g_g, g_h, g_i]
	)
	    Label(layout[1, 1, TopLeft()], label,
	        fontsize = Makie.current_default_theme().Axis.titlesize.val,
	        font = :bold,
	        padding = (0, 5, 5, 0),
	        halign = :right)
	end
end

# ╔═╡ b4ac06f4-8d7c-4a77-a6a6-270ade2d7b71


# ╔═╡ 6f7a244a-0d19-4e58-ad68-ee0c41042b0f
md"""
## 1.A. Interpolation diagram example
"""

# ╔═╡ 0f7241eb-27f5-44b3-8538-a414ab0239b2
begin
	coord_scale = 40
	T_coords = [ 0.8 0.2 0.5 ;  0.5 0.3 0.5 ; 0.75 0.5 0.5 ; 0.6 0.75 0.5 ; 0.2 0.8 0.5 ;] .*coord_scale
	T_w = [0.01, 0.6, 0.9, -0.01, -0.8]

	S_coords = [ 0.7 0.3 0.5 ;  0.2 0.2 0.5 ; 0.25 0.6 0.5 ; 0.75 0.85 0.5].*coord_scale
end;

# ╔═╡ ff03d756-9634-45cb-aea3-c9ea4726e5a0
begin
	using BrainRBMjulia: find_optimal_bias
	R_map = 4
	σ_map = 5
	padding_map = 1
	lims = [coord_scale coord_scale coord_scale; 0. 0. 0.]
	lscale = lims[1,:] - lims[2,:]
	lims .+= hcat(+lscale.*padding_map, -lscale.*padding_map)'
	bbox = BoxAround(
		lims,
		round.(Int, lims[1,:] - lims[2,:]),
		lims[2,:],
	)
	map_x = LinRange(bbox.lims[2,1], bbox.lims[1,1], bbox.size[1]).+0.5
	map_y = LinRange(bbox.lims[2,2], bbox.lims[1,2], bbox.size[2]).+0.5
	map_w = create_map(T_coords, T_w, bbox, R=R_map);
	map_finite!(map_w)
	T_w_interp = interpolation(map_w, T_coords, σ_map, R_map, bbox)
	T_w_bias = find_optimal_bias(T_w, T_w_interp, minbias=0, maxbias=1, stepbias=0.005)
	S_w = interpolation(
		map_w,
		S_coords,
		σ_map,
		R_map,
		bbox
	).*T_w_bias
	s_map_w = permutedims(reshape(
		interpolation(
			map_w,
			permutedims(reduce(hcat,[[x,y,Float64(coord_scale//2)] for x in 1:coord_scale for y in 1:coord_scale])),
			σ_map,
			R_map,
			bbox
		),
		(coord_scale, coord_scale)
	)).*T_w_bias
end;

# ╔═╡ 5887776c-5983-4a3a-97be-fa31ae21412d


# ╔═╡ 4ec9ee87-cda2-41be-960f-3508dbb8ed79
begin	
	ax_a_diag_left = Axis(g_a[1,1], 
		 topspinevisible=true, rightspinevisible=true, 
		 spinewidth=3, aspect=1,
		 rightspinecolor=CONV.COLOR_TEACHER,leftspinecolor=CONV.COLOR_TEACHER,
		 topspinecolor=CONV.COLOR_TEACHER,bottomspinecolor=CONV.COLOR_TEACHER,
		)
	scatter!(ax_a_diag_left,
		T_coords[:,1], T_coords[:,2], 
		color=T_w, colormap=CONV.CMAP_WEIGHTS, colorrange=(-1,+1),
		markersize=10, markerspace=:data,
		strokewidth=3, strokecolor=CONV.COLOR_TEACHER,
	)
	xlims!(ax_a_diag_left,0,coord_scale)
	ylims!(ax_a_diag_left,0,coord_scale)

	
	ax_a_diag_right = Axis(g_a[1,2], 
		 topspinevisible=true, rightspinevisible=true, 
		 spinewidth=3, aspect=1,
		 rightspinecolor=CONV.COLOR_STUDENT,leftspinecolor=CONV.COLOR_STUDENT,
		 topspinecolor=CONV.COLOR_STUDENT,bottomspinecolor=CONV.COLOR_STUDENT,
		)
	heatmap!(ax_a_diag_right,
		s_map_w,
		colormap=CONV.CMAP_WEIGHTS,
		colorrange=(-1,+1)
	)
	scatter!(ax_a_diag_right,
		S_coords[:,1], S_coords[:,2], 
		color=S_w, colormap=CONV.CMAP_WEIGHTS, colorrange=(-1,+1),
		markersize=10, markerspace=:data,
		strokewidth=3, strokecolor=CONV.COLOR_STUDENT,
	)
	xlims!(ax_a_diag_right,0,coord_scale)
	ylims!(ax_a_diag_right,0,coord_scale)

	
	hidedecorations!(ax_a_diag_left)
	hidedecorations!(ax_a_diag_right)
end

# ╔═╡ 4151ae3c-2e21-46ce-8ae8-52f59b2c9e1f


# ╔═╡ e1d68d9e-0e84-4b22-8eff-094e45c47690
md"""
## 1.B. Example maps T, before, after
"""

# ╔═╡ d5b7ade0-5a0a-487b-9a01-146aecae4ced
Texcoords = load_data(LOAD.load_dataWBSC(EXteacher)).coords;

# ╔═╡ cb745154-ec84-47a9-a6e0-2f627fb3443d
Sexcoords = load_data(LOAD.load_dataWBSC(EXstudent)).coords;

# ╔═╡ f2f916cf-afca-41e1-b9f2-486e7172bae2
begin
	Texw = load_brainRBM(LOAD.load_wbscRBM("bRBMs", "bRBM_".*EXteacher.*base_mod))[1].w[:,EXhu];
	lex = quantile_range(Texw[abs.(Texw) .> 1.e-2], 0.95)
end

# ╔═╡ afea2042-1b56-40c4-82e0-08667b9e8732
begin
	stud_path = LOAD.load_wbscRBM(
		"biRBMs", 
		"biRBM_$(EXstudent)_FROM_$(EXteacher)$(base_mod)"
	)
	stud_before_path = LOAD.load_wbscRBM(
		"biRBMs_before_training", 
		"biRBM_$(EXstudent)_FROM_$(EXteacher)$(base_mod)"
	)
	Sexw = load_brainRBM(stud_path)[1].w[:,EXhu]
	SBexw = load_brainRBM(stud_before_path)[1].w[:,EXhu]
end;

# ╔═╡ 46f7e65d-7b8c-48f6-844a-1e1eda2c03cb
begin
	inset_center = mean(Texcoords, Weights(Texw), dims=1)[1,:]
	inset_size = 200
end

# ╔═╡ 64ee4253-5e3a-4155-97ba-4fecf17977a3
begin
	g_b_left = g_b[1,1] = GridLayout()
	g_b_right = g_b[1,2:4] = GridLayout()
	
	ax_teacher_large = Axis(g_b_left[1,1], aspect=DataAspect(),
        bottomspinevisible=false, leftspinevisible=false,
        topspinevisible=false, rightspinevisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false,
        xticksvisible=false, yticksvisible=false,
	)
	
	h_ex_scatter = neuron2dscatter!(ax_teacher_large,
		Texcoords[:,1], Texcoords[:,2],
		Texw;
		cmap=cmap_aseismic(), range=(-lex, +lex),
		edgewidth=0.02, radius=1.0,
		rasterize=5,
		edgecolor=(:black, 0.3),
	)
	poly!(ax_teacher_large,
		Rect(inset_center[1]-inset_size/2, inset_center[2]-inset_size/2, inset_size, inset_size),
		color=(:white, 0),
		strokecolor=:black, strokewidth=1,
	)

	text!(ax_teacher_large,150, 10, text="200μm", align=(:center, :bottom))
	lines!(ax_teacher_large,[50,250], [0,0], color=:black)

	scat_params = (
		cmap=cmap_aseismic(), range=(-lex, +lex),
		edgewidth=0.05, radius=2.0,
		rasterize=5,
		edgecolor=(:black, 0.3)
	)

	ax_teacher_small = Axis(g_b_right[1:2,1], aspect=DataAspect(),
        bottomspinevisible=true, leftspinevisible=true,
        topspinevisible=true, rightspinevisible=true,
        xticklabelsvisible=false, yticklabelsvisible=false,
        xticksvisible=false, yticksvisible=false,
		title="Teacher", titlesize=Makie.current_default_theme().Axis.xlabelsize.val, titlecolor=CONV.COLOR_TEACHER
	)
	neuron2dscatter!(ax_teacher_small,
		Texcoords[:,1], Texcoords[:,2],
		Texw;
		scat_params...
	)
	xlims!(ax_teacher_small, inset_center[1]-inset_size/2, inset_center[1]+inset_size/2)
	ylims!(ax_teacher_small, inset_center[2]-inset_size/2, inset_center[2]+inset_size/2)
	
	ax_Sbefore_small = Axis(g_b_right[1:2,2], aspect=DataAspect(),
        bottomspinevisible=true, leftspinevisible=true,
        topspinevisible=true, rightspinevisible=true,
        xticklabelsvisible=false, yticklabelsvisible=false,
        xticksvisible=false, yticksvisible=false,
		title="Student", subtitle="before training", titlesize=Makie.current_default_theme().Axis.xlabelsize.val, titlecolor=CONV.COLOR_STUDENT,
		subtitlesize=Makie.current_default_theme().Axis.xticklabelsize.val, subtitlecolor=CONV.COLOR_STUDENT,
	)
	neuron2dscatter!(ax_Sbefore_small,
		Sexcoords[:,1], Sexcoords[:,2],
		SBexw;
		scat_params...
	)
	xlims!(ax_Sbefore_small, inset_center[1]-inset_size/2, inset_center[1]+inset_size/2)
	ylims!(ax_Sbefore_small, inset_center[2]-inset_size/2, inset_center[2]+inset_size/2)
	
	ax_Safter_small = Axis(g_b_right[1:2,3], aspect=DataAspect(),
        bottomspinevisible=true, leftspinevisible=true,
        topspinevisible=true, rightspinevisible=true,
        xticklabelsvisible=false, yticklabelsvisible=false,
        xticksvisible=false, yticksvisible=false,
		title="Student", subtitle="after training", titlesize=Makie.current_default_theme().Axis.xlabelsize.val, titlecolor=CONV.COLOR_STUDENT,
		subtitlesize=Makie.current_default_theme().Axis.xticklabelsize.val, subtitlecolor=CONV.COLOR_STUDENT,
	)
	neuron2dscatter!(ax_Safter_small,
		Sexcoords[:,1], Sexcoords[:,2],
		Sexw;
		scat_params...
	)
	xlims!(ax_Safter_small, inset_center[1]-inset_size/2, inset_center[1]+inset_size/2)
	ylims!(ax_Safter_small, inset_center[2]-inset_size/2, inset_center[2]+inset_size/2)

	Colorbar(g_b_right[2,1:3], h_ex_scatter, vertical=false, width=Relative(1/3), flipaxis=false,)
end

# ╔═╡ d7f2fc1f-9192-4523-8b0e-5654e32b3473


# ╔═╡ 09f265c6-c8ce-48a8-a988-13080e8dad86
md"""
## 1.G. Example Hidden Distrib
"""

# ╔═╡ b02ce499-3f39-4587-9831-c4536708cd5a
begin
	_, _, _, _, Tgen, _ = load_brainRBM(LOAD.load_wbscRBM(
		"bRBMs", 
		"bRBM_".*EXteacher.*base_mod
	))
	_, _, _, _, Sgen, _ = load_brainRBM(LOAD.load_wbscRBM(
		"biRBMs", 
		"biRBM_$(EXstudent)_FROM_$(EXteacher)$(base_mod)"
	))
end

# ╔═╡ 26a38e63-179e-4a49-9997-965ed7c87c8c
begin
	ax_ex_Ph = Axis(
		g_g[1,1], 
		xlabel="h", ylabel="P(h)", 
		yscale=log10,
		aspect=1,
	)
	
	density!(ax_ex_Ph,
		Sgen.h[EXhu,:], 
		offset=1.e-1,
		strokewidth=2, 
		strokecolor=(CONV.COLOR_STUDENT, 0.5), 
		color=(:white, 0),
		label="Students"
	)

	density!(ax_ex_Ph,
		Tgen.h[EXhu,:], 
		offset=1.e-1,
		strokewidth=2, 
		strokecolor=CONV.COLOR_TEACHER, 
		color=(:white, 0),
		label="Teacher",
	)
end

# ╔═╡ a40ef7dc-98a8-4dba-b313-f9aad0514a92


# ╔═╡ 61099ccd-488a-4092-947b-20892c1735a3
md"""
## 1.H. QQ- all pairs
"""

# ╔═╡ f77a5e0b-306e-40bc-8265-2e1fa8765677
qs = 0:0.001:1;

# ╔═╡ 83aa7d23-1667-426c-996c-512f1ad2ac7c
begin
	TQQs = []
	SQQs = []
	NRMSES = Matrix{Float64}(undef, length(FISH), length(FISH))
	for (i,teacher) in enumerate(FISH)
		_, _, _, _, Tgen, _ = load_brainRBM(LOAD.load_wbscRBM(
			"bRBMs", 
			"bRBM_".*teacher.*base_mod
		))
		Tqqs = quantile_2d(Tgen.h,qs)
		for (j,student) in enumerate(FISH)
			if teacher == student
				continue
			end
			_, _, _, _, Sgen, _ = load_brainRBM(LOAD.load_wbscRBM(
				"biRBMs", 
				"biRBM_$(student)_FROM_$(teacher)$(base_mod)"
			))
			Sqqs = quantile_2d(Sgen.h, qs)
			NRMSES[i,j] = nRMSE(Tqqs, Sqqs)

			if (teacher == EXteacher) && (student == EXstudent)
				push!(TQQs, vec(Tqqs))
				push!(SQQs, vec(Sqqs))
			end
		end
	end
	NRMSES[diagind(NRMSES)] .= NaN
end

# ╔═╡ 30b10da5-9d9e-4036-b765-5860cdecee11
TQQs

# ╔═╡ 37c7aab8-f7f9-4d89-b252-bb0a58928045
begin
	ax_qq_all = Axis(
		g_h[1,1], 
		xlabel="quantile(Pᵀ(h))", 
		ylabel="quantile(Pˢ(h))", 
		aspect=1,
	)
	h_qq_all = idplotter!(ax_qq_all, reduce(vcat,TQQs), reduce(vcat, SQQs))
	
	Colorbar(g_h[1,2], colormap=h_qq_all.colormap.val, colorrange=h_qq_all.colorrange, scale=h_qq_all.scale, label="Density", height=Relative(0.7))
	colgap!(g_h, 1)
end

# ╔═╡ e3fa92d0-53dd-4652-9f7d-ffeaaca7f287
md"""
## 1.I. QQ matrix
"""

# ╔═╡ 17413920-d32e-41af-803a-aaf4d74de92a
begin
	ax_qq_mat = Axis(g_i[1,1], aspect=1,
	    xticks=((1:length(FISH)), FISH_DISP),
	    yticks=((1:length(FISH)), FISH_DISP),
		xticklabelrotation = π/4,
		xlabel = "Teacher", ylabel="Student",
	)
	h_qq_mat = heatmap!(ax_qq_mat,
		NRMSES, 
		colormap=reverse(RdYlGn_9), colorrange=(0,1)
	)
	
	Colorbar(g_i[1,2], h_qq_mat, label="nRMSE QQ", height=Relative(0.7))
	colgap!(g_i, 1)
	#Colorbar(fig_qq_mat[1,2], hhhhhhh, label="nRMSE(QQ(Teacher), QQ(Student))")
end

# ╔═╡ 45be7047-b1d2-4636-a78f-44a8b9bf48b3


# ╔═╡ 2f0ba514-f49a-4ed9-9c80-590e2a262826
md"""
## 1.C. Weight map distance matrix single fish
"""

# ╔═╡ 6657ac9c-a51e-4322-8074-894752845ac4
begin
	ϵ = 1.e-5; # absolute weight cutoff for correlations
	σ = 4; # kernel width in μm
end

# ╔═╡ 38df5279-98f0-4177-8654-4a20981d4d29
map_dist_path = LOAD.load_misc("WeightDist_$(length(FISH))fish_$(base_mod[3:end])_sigma$(σ)_epsilon$(ϵ)")

# ╔═╡ 3ef5e86b-b257-4426-98f4-8a3a26e3aab9
begin
	flist = h5read(map_dist_path, "fish_list")
	exmap_dist_mat = h5read(map_dist_path, "rho_afterTraining_pairwise", (findall(flist.==EXteacher)[1],findall(flist.==EXstudent)[1],:,:))
end;

# ╔═╡ db1cf899-ff94-44cc-8f1e-2d0a9b5fa446
begin
	ax_mapdist_single = Axis(g_c[1,1], xlabel="Teacher HU μ", ylabel="Student HU ν", aspect=1)
	
	h_mapdist_single = heatmap!(ax_mapdist_single, 
		exmap_dist_mat, 
		colormap=:seismic, colorrange=(-1,+1)
	)

	text!(ax_mapdist_single,
		97,3,
		text="p-value = $(round(greater_than_diag_pval(exmap_dist_mat), sigdigits=2))",
		align=(:right, :bottom),
	)
	
	Colorbar(g_c[1,2], h_mapdist_single, label="ρˢᵗμν", height=Relative(0.7))
	colgap!(g_c, 1)

	# Colorbar(
	# 	fig_main,#[1,2], 
	# 	hhhhhh, 
	# 	label="ρˢᵗμν", 
	# 	alignmode=Outside(-40),
	# 	bbox=ax_mapdist_single.scene.viewport,
	# 	halign=:right,
	# 	height=Relative(0.3),
	# )
end

# ╔═╡ 6bfc8039-60f6-4628-8afe-706322df6b52
md"""
## 1.D. Weight maps p-vals matrix
"""

# ╔═╡ 3e44bf89-341a-4342-a861-1cddb7caa690
rho_pvals_pairs = [greater_than_diag_pval(h5read(map_dist_path, "rho_afterTraining_pairwise", (i,j,:,:))) for i=1:length(FISH), j=1:length(FISH)];

# ╔═╡ bd1a5abd-9583-43fd-adcd-a8d52fa40735
begin
	ax_mapdist_pval_mat = Axis(g_d[1,1], aspect=1,
	    xticks=((1:length(FISH)), FISH_DISP),
	    yticks=((1:length(FISH)), FISH_DISP),
		xticklabelrotation = π/4,
		xlabel = "Teacher", ylabel="Student",
	)
	h_mapdist_pval_mat = heatmap!(ax_mapdist_pval_mat,
		rho_pvals_pairs, 
		colormap=:cividis
	)
	
	Colorbar(g_d[1,2], h_mapdist_pval_mat, label="p-value", height=Relative(0.7))
	colgap!(g_d, 1)
end

# ╔═╡ ac41688b-627a-4193-9f82-8461cd5b6c99
md"""
## 1.E. Weight maps distance all TS
"""

# ╔═╡ 4ea97308-bf2f-4968-8d27-30b5b39c2c0b
begin
	Ρ_beforeTraining = h5read(map_dist_path, "rho_beforeTraining")
	Ρ_afterTraining = h5read(map_dist_path, "rho_afterTraining")
	Ρ_shuff = h5read(map_dist_path, "rho_shuff")
	F_on = h5read(map_dist_path, "f_on")
end;

# ╔═╡ 0ab611b9-96ba-4c47-b67e-f09fb855a2dc


# ╔═╡ e08485b7-06b8-422d-ac15-fb0161714562
md"""
## 1.F. Weight maps distance matrix all TS
"""

# ╔═╡ 38dbc04d-3419-44a3-a42a-77c8e9193676
begin	
	ax_dist_mat = Axis(g_f[1,1], aspect=1,
		    xticks=((1:length(FISH)), FISH_DISP),
		    yticks=((1:length(FISH)), FISH_DISP),
			xticklabelrotation = π/4,
			xlabel = "Teacher", 
		    ylabel="Student",
		)
	h_dist_mat = heatmap!(ax_dist_mat,
		nanmedian(Ρ_afterTraining), 
		colormap=:seismic, colorrange=(-1,+1),
	)
	
	Colorbar(g_f[1,2], h_dist_mat, label="median(ρˢᵗ)", height=Relative(0.7))
	colgap!(g_f, 1)

	# Colorbar(fig_mapdist_mat[1,4], hhhhh, label="median(ρˢᵗ)", height=Relative(0.6))
end

# ╔═╡ 4d924012-ff6e-4147-9b94-d770035f60eb


# ╔═╡ 426f5b60-62ad-4365-94cd-0aa40779ee4e
md"""
## 1.END Adjustments
"""

# ╔═╡ a49c445f-797c-4337-b04c-fc78ba6171e5
all_axes = [ax for ax in fig_main.content if typeof(ax)==Axis];

# ╔═╡ 4474b085-3cad-4759-b075-7027e22cd399
# ╠═╡ disabled = true
#=╠═╡
begin
	# adjusting axis label spacing
	yspace = maximum(tight_yticklabel_spacing!, all_axes)
	xspace = maximum(tight_xticklabel_spacing!, all_axes)
	for ax in all_axes
		ax.yticklabelspace = yspace
		ax.yticklabelspace = yspace
	end
end
  ╠═╡ =#

# ╔═╡ 65ed213f-4fd9-4e54-aaeb-8cb0598f090f
colsize!(fig_main.layout,3,Relative(0.9/3) )

# ╔═╡ 3ef7ff2d-a1f2-480e-b537-fc9e0d363f59
for ax in all_axes
	ax.alignmode = Mixed(left=0)
end

# ╔═╡ 6019467c-1e9a-406f-b8f8-7c584329ff7b


# ╔═╡ 3f67cae9-c427-4481-9514-612be1605801
fig_main

# ╔═╡ 40a716f4-3e26-4311-8bae-48ec38f0185c
# ╠═╡ disabled = true
#=╠═╡
save(@figpath("main"), fig_main)
  ╠═╡ =#

# ╔═╡ fab3d8a9-0e7b-4395-b295-d3b571eddc81


# ╔═╡ 931b25f4-d638-4e08-9b04-a3ca114d6d6d
md"""
# 2. Supplementaries
"""

# ╔═╡ b73d7db7-9c21-49b3-a7a7-1fd3b9e34393
md"""
## 2.1. Stats all pairs
"""

# ╔═╡ 6651a85e-ce16-4487-a961-80f1dfa1521d
begin
	fig_S_stats_all_fish = Figure(size=(53, 49).*(3,3).*(4/3/0.35))
	
	g_ab_S_stats_all_fish   = fig_S_stats_all_fish[1:2,1:2] = GridLayout()
	g_a_S_stats_all_fish    = g_ab_S_stats_all_fish[1,1] = GridLayout()
	g_b_S_stats_all_fish    = g_ab_S_stats_all_fish[1,2] = GridLayout()
	g_cd_S_stats_all_fish   = fig_S_stats_all_fish[3,1:2] = GridLayout()
	g_c_S_stats_all_fish    = g_cd_S_stats_all_fish[1,1] = GridLayout()
	g_d_S_stats_all_fish    = g_cd_S_stats_all_fish[1,2] = GridLayout()

	for (label, layout) in zip(
		["A", "B", "C", "D"], 
		[g_a_S_stats_all_fish, g_b_S_stats_all_fish, g_c_S_stats_all_fish, g_d_S_stats_all_fish]
	)
	    Label(layout[1, 1, TopLeft()], label,
	        fontsize = Makie.current_default_theme().Axis.titlesize.val,
	        font = :bold,
	        padding = (0, 5, 5, 0),
	        halign = :right)
	end
end

# ╔═╡ 45b9003b-89e2-47a0-a89a-bbb75e6ea53d
md"""
### 2.1.A. All pairs all stats
"""

# ╔═╡ 73d47734-3ac9-4b9d-9068-decce1e43704
begin
	# paths
	model_paths = []
	for teacher in FISH
		t_paths = []
		for student in FISH
			ts_paths = []
			if teacher == student
				push!(
					t_paths, 
					LOAD.load_wbscRBMs(
						"Repeats",
						"bRBM_".*teacher.*base_mod
					)
				)
			else
				push!(
					t_paths, 
					LOAD.load_wbscRBM(
						"biRBMs",
						"biRBM_$(student)_FROM_$(teacher)$(base_mod)"
					)
				)				
			end
			#push!(t_paths, ts_paths)
		end
		push!(model_paths, t_paths)
	end
end

# ╔═╡ fa6658b2-1257-4a4e-b68a-4404344cf849
begin
	scale = 3
	ax_fit_1 = Axis(
	    g_a_S_stats_all_fish[1,1],
	    xticks=((1:length(FISH)).*scale, FISH_DISP),
	    yticks=((1:length(FISH)).*scale, FISH_DISP),
	    aspect=DataAspect(),
		xticklabelrotation = π/4,
		xlabel = "Teacher", ylabel="Student",
	)
	cmap_max = nRMSEs_L4(
		load_brainRBM_eval(model_paths[1][1], ignore="1-nLLH"), 
		max=true
	)

	for i in 1:length(FISH)
		for j in 1:length(FISH)
			evals = load_brainRBM_eval(model_paths[j][i], ignore="1-nLLH")
			if i != j
				evals = [evals]
			end
			if (i==length(FISH)) & (j==length(FISH))
				ax_fontsize = 8
			else
				ax_fontsize = 0
			end
			multipolarnrmseplotter!(ax_fit_1, 
				evals, 
				nRMSEs_L4(evals), 
				cmap_max=cmap_max,
				origin=[j,i].*scale, 
				ax_fontsize=ax_fontsize,
				# cmap=reverse(CONV.CMAP_GOODNESS),
			)
		end
	end

	Colorbar(
		g_a_S_stats_all_fish[1,2], 
		colormap=CONV.CMAP_GOODNESS, colorrange=(0,cmap_max),
		label="L4 norm of statistics' nRMSE",
		height=Relative(0.6),
	)
end

# ╔═╡ 287ef1c1-498e-4acf-ad9e-53f169a6743f


# ╔═╡ 5178f5bc-1fdd-495d-a1ce-83941ce75770
md"""
### 2.1.A' Teachers vs Students
"""

# ╔═╡ 7bc16911-3343-4cd4-afc7-5c1efd1aaa4d
begin
	norms_teachers, norms_students = Float64[], Float64[]
	for i in 1:length(FISH)
		for j in 1:length(FISH)
			p = model_paths[i][j]
			evals = load_brainRBM_eval(p, ignore="1-nLLH")
			norm = nRMSEs_L4(evals)
			if typeof(norm) == Float64
				push!(norms_students, norm)
			else
				for n in norm
					push!(norms_teachers, n)
				end
			end
		end
	end
end

# ╔═╡ 6674da27-b796-4d34-a26c-0ed71e3dacdf
begin
	norms = vcat(norms_teachers, norms_students)
	norm_labs = vcat(zeros(Int,size(norms_teachers)), ones(Int,size(norms_students)))
	norm_colors = vcat(fill(CONV.COLOR_TEACHER,size(norms_teachers)), fill(CONV.COLOR_STUDENT,size(norms_students)))
end;

# ╔═╡ 28351a7b-0d24-4024-93c4-5a0895c332e6
pval = pvalue(MannWhitneyUTest(norms_teachers, norms_students), tail=:right)

# ╔═╡ 4d2d64d5-7f8d-461d-923b-9de01099ae51
begin
	# fig_trainingnorms = Figure(size=dfsize().*(1,2))
	ax_trainingnorms = Axis(
		g_b_S_stats_all_fish[1,1],
		xticks=([0,1], ["Teachers", "Students"]),
		xticksvisible=false, bottomspinevisible=false, xticklabelrotation=π/4,
		ylabel="L4 norm of statistics' nRMSE",
	)
	boxplot!(
		ax_trainingnorms, 
		norm_labs, norms, 
		color=norm_colors,
		show_notch=true, whiskerwidth=0.2,
	)
	bracket!(
		ax_trainingnorms,
		0,1.3,1,1.3, 
		style = :square, 
		text="pval < $(ceil(pval, sigdigits=1))", 
		width=3
	)
	ylims!(ax_trainingnorms, 0, cmap_max)
	# fig_trainingnorms
end

# ╔═╡ f694effb-eed2-4a87-bbec-8f72c4a86fd1


# ╔═╡ 73b4ef64-54e6-4f41-809e-683cab4b1960


# ╔═╡ c5800a80-6066-4dcf-947f-6958404cbbdb
md"""
### 2.1.B. hT-hS combined
"""

# ╔═╡ 4c5f25cf-c288-4b0f-a76e-19073521e9a5
begin
	MMM = parse(Int, split(split(base_mod,"M")[2], "_")[1])
	Xs = Array{Float32}(undef, length(FISH), length(FISH), MMM^2)
	Ys = Array{Float32}(undef, length(FISH), length(FISH), MMM^2)
	Ws = Array{Float32}(undef, length(FISH), length(FISH), MMM^2)
	Zs = Array{Float32}(undef, length(FISH), length(FISH), MMM^2)
	
	for (i,teacher) in enumerate(FISH)
		teacher_path = LOAD.load_wbscRBM(
			"bRBMs",
			"bRBM_".*teacher.*base_mod
		)
		Trbm3, _, _, Tsplit3, Tgen3, _ = load_brainRBM(teacher_path)
		#Thcov = cov(Tgen.h, dims=2, corrected=false)
		momT = compute_all_moments(Trbm3, Tsplit3, Tgen3, max_vv=2)
		
		for (j,student) in enumerate(FISH)
			if teacher == student
				continue
			end
			student_path = LOAD.load_wbscRBM(
				"biRBMs",
				"biRBM_$(student)_FROM_$(teacher)$(base_mod)"
			)
			Srbm3, _, _, Ssplit3, Sgen3, _ = load_brainRBM(student_path)
			#Shcov = cov(Sgen.h, dims=2, corrected=false)
			momS = compute_all_moments(Srbm3, Ssplit3, Sgen3, max_vv=2)
	
			Xs[i,j,:] .= vec(momT.gen["<hh> - <h><h>"])
			Ys[i,j,:] .= vec(momS.gen["<hh> - <h><h>"])
			Ws[i,j,:] .= vec(momT.train["<hh> - <h><h>"])
			Zs[i,j,:] .= vec(momT.valid["<hh> - <h><h>"])
		end
	end
	Xs[.~isfinite.(Xs)] .= 0
	Ys[.~isfinite.(Ys)] .= 0
end

# ╔═╡ 5a44547b-7b68-46f6-9284-48af68e9c414


# ╔═╡ b42ba603-0ee2-4a5e-9949-2ea18841686c
md"""
### 2.1.C. hT-hS matrix
"""

# ╔═╡ dd46b267-c4ce-41a7-a279-4ee0288c6cf6
begin
	# fig_hh_mat = Figure(size=dfsize().*(1.4,1))
	ax_hh_matrix = Axis(g_d_S_stats_all_fish[1,1], aspect=1,
	    xticks=((1:length(FISH)), FISH_DISP),
	    yticks=((1:length(FISH)), FISH_DISP),
		xticklabelrotation = π/4,
		xlabel = "Teacher", ylabel="Student",
	)
	hh_mat = [nRMSE(Xs[i,j,:], Ys[i,j,:],X_opt=Ws[i,j,:], Y_opt=Zs[i,j,:]) for i=1:length(FISH), j=1:length(FISH)]
	hh_mat[diagind(hh_mat)] .= NaN
	h_hh_matrix = heatmap!(ax_hh_matrix,
		hh_mat, colormap=CONV.CMAP_GOODNESS, colorrange=(0,1))
	Colorbar(g_d_S_stats_all_fish[1,2], h_hh_matrix, label="nRMSE", height=Relative(0.7))
	colgap!(g_d_S_stats_all_fish, 1)
end

# ╔═╡ 85f6a9b3-d459-4b8f-9f83-6c5f321ca775


# ╔═╡ e06a5b08-2e1f-408c-8826-5673735127d4
md"""
### 2.1.END Adjustments
"""

# ╔═╡ daf9b955-82dc-42da-9e9c-f9720126cf57
begin
	colsize!(g_ab_S_stats_all_fish, 2, Relative(1/4))
	all_axes_S_stats_all_fish = [ax for ax in fig_S_stats_all_fish.content if typeof(ax)==Axis];
	for ax in all_axes_S_stats_all_fish
		ax.alignmode = Mixed(left=0)
	end
end

# ╔═╡ a2486aa6-e4d6-485a-ab21-0d9b1ecd45b8
fig_S_stats_all_fish

# ╔═╡ 0cdc3e62-ca2f-4a14-8ecb-6c894b78be7c
# ╠═╡ disabled = true
#=╠═╡
save(@figpath("SUPP_stats_all_pairs"), fig_S_stats_all_fish)
  ╠═╡ =#

# ╔═╡ 78746d5c-e09d-4236-9f05-8d5685c921ad


# ╔═╡ 41db29b5-c8ac-485f-9c2f-ed414adb51cb
md"""
## 2.2. Stats and Hidden Distrib Example teacher
"""

# ╔═╡ 1b024dbe-000e-4be8-b8b8-dbeacb9cb1e7


# ╔═╡ 2a9dbc89-6726-46d7-97fb-2ad6740940b7
begin
	fig_S_ex_teacher = Figure(size=(53, 49).*(6,6).*(4/3/0.35))

	ax_S_ex_T_params = (
		aspect=1,
		xticklabelsvisible=false,
		yticklabelsvisible=false,
	)
	axs_v  = [Axis(fig_S_ex_teacher[1,i]; ax_S_ex_T_params...) for i in 1:length(FISH)]
	axs_vv = [Axis(fig_S_ex_teacher[2,i]; ax_S_ex_T_params...) for i in 1:length(FISH)]
	axs_vh = [Axis(fig_S_ex_teacher[3,i]; ax_S_ex_T_params...) for i in 1:length(FISH)]
	axs_h  = [Axis(fig_S_ex_teacher[4,i]; ax_S_ex_T_params...) for i in 1:length(FISH)]
	axs_hh = [Axis(fig_S_ex_teacher[5,i]; ax_S_ex_T_params...) for i in 1:length(FISH)]
	axs_dh = [Axis(fig_S_ex_teacher[6,i]; ax_S_ex_T_params...) for i in 2:length(FISH)]
	axs_S_ex_T = Dict(
		"<v>" => axs_v,
		"<vv> - <v><v>" => axs_vv,
		"<vh>" => axs_vh,
		"<h>" => axs_h,
		"<hh> - <h><h>" => axs_hh,
		"dh" => axs_dh,
	)

	for s in keys(axs_S_ex_T)
		axs = axs_S_ex_T[s]
		if s == "<hh> - <h><h>"
			for ax in axs
				xlims!(ax, -1, +1.5)
				ylims!(ax, -1, +1.5)
			end
		else
			linkaxes!(axs)
		end
		axs[1].xticklabelsvisible = true
		axs[1].yticklabelsvisible = true
	end

	for (i, title) in zip(
		1:length(FISH),
		["Teacher", "Student 1", "Student 2", "Student 3", "Student 4", "Student 5"]
	)
		Label(
			fig_S_ex_teacher[0, i, Bottom()], 
			title, 
			padding = (0, 0, 0, 0), 
			fontsize=Makie.current_default_theme().Axis.titlesize.val,
			font=Makie.current_default_theme().Axis.titlefont.val,
		)
	end
	for (i,title) in zip(
		1:6,
		["⟨v⟩","⟨vv⟩ - ⟨v⟩⟨v⟩","⟨vh⟩","⟨h⟩","⟨hh⟩ - ⟨h⟩⟨h⟩","Q(h)"]
	)
		Label(
			fig_S_ex_teacher[i, 0, Right()], 
			title, 
			padding = (0, 0, 0, 0), 
			rotation = pi/2,
			fontsize=Makie.current_default_theme().Axis.titlesize.val,
			font=Makie.current_default_theme().Axis.titlefont.val,
		)
	end
	rowsize!(fig_S_ex_teacher.layout,0,10)
	colsize!(fig_S_ex_teacher.layout,0,10)


	axs_hh[1].xlabel = "data"
	axs_hh[1].ylabel = "generated"

	axs_dh[1].xlabel = "teacher"
	axs_dh[1].ylabel = "student"
end

# ╔═╡ a6c609b2-7629-4d0d-a537-741156090da7


# ╔═╡ 072ea2b0-eefb-44f5-aee0-7f4a3dee2f01
md"""
### 2.2.1. Training Stats
"""

# ╔═╡ e6930201-68d4-46d8-9e58-f13808d712d4
max_vv = 10000;

# ╔═╡ 94ff8a91-239a-46a8-aeee-c4c9b05d6153
begin
	# teacher
	rbm,_,evaluationsT,dsplit,gen,_ = load_brainRBM(LOAD.load_wbscRBM(
		"bRBMs", 
		"bRBM_".*EXteacher.*base_mod
	))
	momentsT = compute_all_moments(rbm, dsplit, gen; max_vv)
	for s in keys(momentsT.gen)
		idplotter!(axs_S_ex_T[s][1], 
				   momentsT.valid[s], 
				   momentsT.gen[s], 
				   nrmse=evaluationsT[s]
				  )
	end
end

# ╔═╡ 95d18772-46ab-4146-9238-f97919460c23
begin
	# students
	for (i,student) in enumerate(STUDENTS)	
		rbm,_,evaluationsS,dsplit,gen,_ = load_brainRBM(LOAD.load_wbscRBM(
			"biRBMs", 
			"biRBM_$(student)_FROM_$(EXteacher)$(base_mod)"
		))
		momentsS = compute_all_moments(rbm, dsplit, gen; max_vv)
		for s in keys(momentsS.gen)
			idplotter!(axs_S_ex_T[s][i+1], 
					   momentsS.valid[s], 
					   momentsS.gen[s], 
					   nrmse=evaluationsS[s]
					  )
		end
	end
end

# ╔═╡ 437c1dae-d477-47bc-b992-f5765f06d593


# ╔═╡ b20769a5-99e2-43b7-8f35-01daa7f99dc1


# ╔═╡ 10806fac-1aeb-4c6a-bf34-be6a96b92a10
md"""
### 2.2.2. Hidden QQs
"""

# ╔═╡ 4d89d806-6c77-4101-a48d-828020072f58
begin
	Trbm, _, _, _, _, _ = load_brainRBM(LOAD.load_wbscRBM(
		"bRBMs",
		"bRBM_".*EXteacher.*base_mod
	))
	Tdata = load_data(LOAD.load_dataWBSC(EXteacher))
	Th_samp = reshape(
			sample_h_from_v(Trbm, repeat(Tdata.spikes, 1,1,3)), 
			(size(Trbm.hidden,1), 3*size(Tdata.spikes,2))
		)
	Sh_samp = []
	for student in FISH
		if student == EXteacher
			continue
		end
		rbm, _, _, _, _, _ = load_brainRBM(LOAD.load_wbscRBM(
			"biRBMs", 
			"biRBM_$(student)_FROM_$(EXteacher)$(base_mod)"
		))
		data = load_data(LOAD.load_dataWBSC(student))
		push!(
			Sh_samp,
			reshape(
				sample_h_from_v(rbm, repeat(data.spikes, 1,1,3)), 
				(size(rbm.hidden,1), 3*size(data.spikes,2))
			)
		)
	end
end

# ╔═╡ 50a6ba23-7aa5-4d3c-a3ad-93c5a2a8ecef
begin
	Tqqs = quantile_2d(Th_samp,qs)
	Sqqs = []
	for i in 1:length(Sh_samp)
		push!(Sqqs, quantile_2d(Sh_samp[i], qs))
	end
end

# ╔═╡ d9dcdea6-af3a-4f13-83a2-83cb17c72495
begin
	# students
	for (i,student) in enumerate(STUDENTS)
		idplotter!(axs_dh[i], 
			Tqqs,Sqqs[i]
		)
	end
end

# ╔═╡ b8c1dd36-bdf8-42cf-bdbb-b29865b61ce8


# ╔═╡ 8d8e4362-e295-441f-ac66-625421ed4690
fig_S_ex_teacher

# ╔═╡ a4cf615d-032a-4dea-80b3-d5484039f89f
# ╠═╡ disabled = true
#=╠═╡
save(@figpath("SUPP_stats_exteacher"), fig_S_ex_teacher)
  ╠═╡ =#

# ╔═╡ 63d0a9d3-9c5e-4f1e-a130-c31b396bbda7


# ╔═╡ 85b8c734-f733-4cf9-9b38-1d8f2c8da584


# ╔═╡ f340e442-a8b8-452f-9065-e74008a4277e
md"""
## 2.3. Spatial distance
"""

# ╔═╡ b4c57845-d272-4936-b8a1-3f4e4dfe892e
begin
	fig_S_wmaps = Figure(size=(53, 49).*(3,4).*(4/3/0.35))
	
	g_acd_S_wmaps    = fig_S_wmaps[1:4,1] = GridLayout()
	g_b_S_wmaps    = fig_S_wmaps[1:4,2] = GridLayout()

	g_a_S_wmaps    = g_acd_S_wmaps[1,1:2] = GridLayout()
	g_c_S_wmaps    = g_acd_S_wmaps[2,1:2] = GridLayout()
	g_d_S_wmaps    = g_acd_S_wmaps[3,1:2] = GridLayout()

	for (label, layout) in zip(
		["A", "B", "C", "D"], 
		[g_a_S_wmaps, g_b_S_wmaps, g_c_S_wmaps, g_d_S_wmaps]
	)
	    Label(layout[1, 1, TopLeft()], label,
	        fontsize = Makie.current_default_theme().Axis.titlesize.val,
	        font = :bold,
	        padding = (0, 5, 5, 0),
	        halign = :right)
	end

	colsize!(fig_S_wmaps.layout,2,Relative(2/3))
end

# ╔═╡ d783a19f-4e70-402f-9df7-f7b961af4921


# ╔═╡ a9837ac3-2a16-4aa6-8685-e18bc817dc3a
md"""
### 2.3.A. Joint rho frac
"""

# ╔═╡ 480da86e-76b5-4f8a-ab00-03e4929b2819


# ╔═╡ 900ff03a-926b-412b-9609-c3e1bb961eea


# ╔═╡ f58f236a-8a65-43ef-8367-87be4e8770eb
md"""
### 2.3.B. Example maps
"""

# ╔═╡ fb2c3dab-210d-4036-b0a0-f9577074709f


# ╔═╡ 48ca783b-4e34-48dc-acc9-8c7c98f59c4a
md"""
### 2.3.C. Hist p-val
"""

# ╔═╡ b2f945bc-6f0e-47f0-b58c-5c326f1ceb16
begin
	begin
		ax_hist_pval = Axis(g_c_S_wmaps[1,1],
			xlabel="ρᵢⱼ - maxⱼ≠ᵢ ρᵢⱼ",
			ylabel="PDF",
		)
		hist!(ax_hist_pval,
			  reduce(vcat, greater_than_diag_bootstrap(exmap_dist_mat)), bins=-1.5:0.1:+1.5, normalization=:pdf, color=(:grey, 0.75))
		hist!(ax_hist_pval,
			  greater_than_diag(exmap_dist_mat), bins=-1.5:0.1:+1.5, color=(:orange, 0.75), normalization=:pdf)
	end
end

# ╔═╡ 0c4db1ec-b131-4b2d-b347-bb0850183bad
fig_S_wmaps

# ╔═╡ 355ddbc3-6818-42a1-a262-b3ade162481d
# ╠═╡ disabled = true
#=╠═╡
save(@figpath("SUPP_mapd_dist"), fig_S_wmaps)
  ╠═╡ =#

# ╔═╡ b17900f4-2551-49eb-8b32-4d23e4aac727


# ╔═╡ 7d52635d-6256-43bd-ae54-44fb50cbca6b


# ╔═╡ d6c6c5ea-e7b2-48a4-bbb8-83d4dc2ed997


# ╔═╡ d25d26f8-19ed-43c0-a25d-1d2de9098457


# ╔═╡ ca5b5f3e-eca8-4a3f-a016-9a1d8e6ee586
function off_diag(A::AbstractArray)
	@assert size(A,1) == size(A,2)
	a = [A[i,j,:] for i in 1:size(A,1) for j in 1:size(A,1) if i!=j]
	return reduce(vcat, a)
end

# ╔═╡ 87166a10-b8f4-4d5b-b2c9-ad9e1fdf6448
begin
	Ρ_beforeTraining_OD = off_diag(Ρ_beforeTraining)
	Ρ_afterTraining_OD = off_diag(Ρ_afterTraining)
	F_on_OD = off_diag(F_on)
end

# ╔═╡ b7068b37-5206-43ec-b168-150d9cc48642
begin
	ax_dist_dens = Axis(g_e[1,1], xlabel="ρˢᵗ", ylabel="Density", aspect=1)
	
	density!(ax_dist_dens,
			 Ρ_shuff[isfinite.(Ρ_shuff)], color=:grey, offset=1.e-5
			)
	density!(ax_dist_dens,
			 Ρ_beforeTraining_OD[isfinite.(Ρ_beforeTraining_OD)], 
			 color=CONV.COLOR_STUDENTb, offset=1.e-5
			)
	density!(ax_dist_dens,
			 Ρ_afterTraining_OD[isfinite.(Ρ_afterTraining_OD)], 
			 color=CONV.COLOR_STUDENT, offset=1.e-5
			)
end

# ╔═╡ eb1201e0-ae5c-4c32-a207-a63a4f24f331
begin
	ax_hh_combined = Axis(g_c_S_stats_all_fish[1,1], xlabel="Teachers", ylabel="Students", aspect=1)
	x = off_diag(Xs)
	h_hh_combined = idplotter!(ax_hh_combined,
		x, 
		off_diag(clamp.(Ys, minimum(x)*1.5, maximum(x)*1.5)),
		nrmse=nRMSE(
			off_diag(Xs), 
			off_diag(Ys), 
			X_opt=off_diag(Ws), 
			Y_opt=off_diag(Zs)
		)
	)
	
	Colorbar(g_c_S_stats_all_fish[1,2], colormap=h_hh_combined.colormap.val, colorrange=h_hh_combined.colorrange, scale=h_hh_combined.scale, label="Density", height=Relative(0.7))
	colgap!(g_c_S_stats_all_fish, 1)
end

# ╔═╡ 9e3a2f11-a474-451d-a6a0-6e89f3a9345d
begin
	function find_in_joint(X::AbstractArray,Y::AbstractArray,x::Real,y::Real)
		Dx = abs.(X .- x)
		Dx[isnan.(Dx)] .= 1.e5
		
		Dy = abs.(Y .- y)
		Dy[isnan.(Dy)] .= 1.e5
		return argmin(Dx .+ Dy)
	end
	function find_in_joint(X::AbstractArray,Y::AbstractArray,x::AbstractVector,y::AbstractVector)
		return [find_in_joint(X,Y,x[i],y[i]) for i in 1:length(x)]
	end
end

# ╔═╡ aa563bc3-09d6-48fb-916f-1da02db28e3b
begin
	inds = find_in_joint(
		Ρ_afterTraining, F_on, 
		[0.8, 0.7, 0.6, 0.75, 0.4, 0.9, 0.5, 0.0], 
		[0.5, 0.3, 0.4, 1.00, 0.3, 3.5, 2.0, 0.2]
	)
	ind_labs = ["abcdefghijk"[i] for i in 1:length(inds)]
end

# ╔═╡ 54018004-8652-44c6-be50-2d17b6c78a44
begin
	atop_rho_frac_joint = Axis(g_a_S_wmaps[1,1], xticklabelsvisible=false, ylabel="Density", yticks=[0.0, 1.0, 2.0])
	density!(atop_rho_frac_joint,
			 Ρ_afterTraining_OD[isfinite.(Ρ_afterTraining_OD)], color=CONV.COLOR_STUDENT)
	
	ajoin_rho_frac_joint = Axis(g_a_S_wmaps[2,1], xlabel="ρˢᵗ_μ", ylabel="Nˢ_μ / Nᵗ_μ")
	#scatter!(Ρ_afterTraining_OD, F_on_OD, color=(:black, 0.1))
	hexbin!(ajoin_rho_frac_joint,
			Ρ_afterTraining_OD[isfinite.(Ρ_afterTraining_OD)], F_on_OD[isfinite.(Ρ_afterTraining_OD)], colormap=:greens, bins=75)
	
	scatter!(ajoin_rho_frac_joint,
			 Ρ_afterTraining[inds], F_on[inds], color=:red, marker=:xcross, markersize=10)
	for i in 1:length(inds)
		text!(ajoin_rho_frac_joint,
			Ρ_afterTraining[inds[i]]+0.04, F_on[inds][i], 
			text="$(ind_labs[i])", font=:bold,
			align=(:left, :top),
			
		)
	end
	
	aright_rho_frac_joint = Axis(g_a_S_wmaps[2,2], xlabel="Density", yticklabelsvisible=false, xticks=[0.0, 1.5])
	density!(aright_rho_frac_joint,
			 F_on_OD[isfinite.(F_on_OD)], color=CONV.COLOR_STUDENT, direction=:y)

	linkyaxes!(ajoin_rho_frac_joint, aright_rho_frac_joint)
	linkxaxes!(ajoin_rho_frac_joint, atop_rho_frac_joint)

	colsize!(g_a_S_wmaps, 1, Relative(0.7))
	rowsize!(g_a_S_wmaps, 2, Relative(0.7))
	colgap!(g_a_S_wmaps, 1, Relative(0.01))
	rowgap!(g_a_S_wmaps, 1, Relative(0.01))
end

# ╔═╡ 831d8e8e-56e2-49f5-8732-57b4fe5ffad3
begin
	l = 1.3
	nnn = length(ind_labs)
	nnn//2
	
	axs_exmaps = []
	for i in 1:Int(nnn//2)
		for j in 1:2
			push!(
				axs_exmaps, 
				Axis(
					g_b_S_wmaps[i,j],
					aspect=DataAspect(),
			        bottomspinevisible=false, leftspinevisible=false,
			        topspinevisible=false, rightspinevisible=false,
			        xticklabelsvisible=false, yticklabelsvisible=false,
			        xticksvisible=false, yticksvisible=false,
				)
			)
		end
	end
	linkaxes!(axs_exmaps...)

	for i in 1:nnn
		text!(axs_exmaps[i],
			0, 1, space=:relative,
			text="$(ind_labs[i])", font=:bold, 
			fontsize=Makie.current_default_theme().Axis.titlesize.val,
			align=(:left, :top),
		)
		tcoords = load_data(LOAD.load_dataWBSC(FISH[inds[i][1]])).coords
		scoords = load_data(LOAD.load_dataWBSC(FISH[inds[i][2]])).coords
		tweights = load_brainRBM(
			LOAD.load_wbscRBM(
				"bRBMs",  
				"bRBM_".*FISH[inds[i][1]].*base_mod
			)
		)[1].w[:,inds[i][3]]
		sweights = load_brainRBM(
			LOAD.load_wbscRBM(
				"biRBMs", 
				"biRBM_$(FISH[inds[i][2]])_FROM_$(FISH[inds[i][1]])$(base_mod)"
			)
		)[1].w[:,inds[i][3]]
		neuron2dscatter!(axs_exmaps[i],
			tcoords[:,1], tcoords[:,2],
			tweights,
			cmap=cmap_aseismic(), range=(-l, +l),
			edgewidth=0.05, radius=1.0,
			rasterize=5,
		)
		neuron2dscatter!(axs_exmaps[i],
			scoords[:,1].+500, scoords[:,2],
			sweights,
			cmap=cmap_aseismic(), range=(-l, +l),
			edgewidth=0.05, radius=1.0,
			rasterize=5,
		)
	end

	text!(axs_exmaps[end], 600, 10, text="200μm", align=(:center, :bottom))
	lines!(axs_exmaps[end], [500,700], [0,0], color=:black)
	text!(axs_exmaps[end], 250, 950, text="Teacher", align=(:center, :bottom))
	text!(axs_exmaps[end], 750, 950, text="Student", align=(:center, :bottom))

	Colorbar(g_b_S_wmaps[Int(nnn//2)+1, 1:2], 
			 colormap=cmap_aseismic(), 
			 colorrange=(-l,+l), 
			 label="wᵢ_μ", 
			 # height=Relative(0.6), 
			 width=Relative(0.4),
			 flipaxis=false, 
			 vertical=false
			)
end

# ╔═╡ e4276472-5c23-464a-9db8-aec4ce85baf4


# ╔═╡ Cell order:
# ╟─736cb1dd-abe6-401e-86cc-5c97526d4444
# ╠═3ef22527-b023-4942-b76e-4ee055b1af6b
# ╠═ed7a57fe-b0b3-11ef-2c97-b71673d90a65
# ╠═2d4bcf99-236e-4291-939d-b3a8f9e6e037
# ╠═64381840-77f7-4af3-9c5d-703d8f4cf468
# ╠═6d9eb4c9-66a9-4114-87ec-6d8638a97e4e
# ╟─cd719af5-c4a6-4822-8aee-3a497e55ac24
# ╠═47be63c0-05a2-4295-938b-33c4d7350058
# ╟─cae5892f-c7a4-4347-9a04-b16acc2531ac
# ╠═34d666bf-acc9-4433-a38d-fd131116f84a
# ╠═ec439d7a-9a6a-4757-aaee-2c488c0e9d17
# ╟─d2817416-6e1d-49ac-8a81-cc20118f90b5
# ╟─60774e68-5d35-407c-8900-40eed1d04b1d
# ╟─bff9fbfc-9ee8-40cd-b6d2-46399951ed00
# ╟─3d8b112a-d8cc-4618-ba69-33858019fb53
# ╟─70ed6e1f-4c61-4e1f-a5d7-b775487b564e
# ╟─f21f6595-e447-4bad-b796-ccc657cfb298
# ╠═fba1912a-9abc-474c-a9e0-3b603555e0d7
# ╠═b4ac06f4-8d7c-4a77-a6a6-270ade2d7b71
# ╟─6f7a244a-0d19-4e58-ad68-ee0c41042b0f
# ╠═0f7241eb-27f5-44b3-8538-a414ab0239b2
# ╠═5887776c-5983-4a3a-97be-fa31ae21412d
# ╠═ff03d756-9634-45cb-aea3-c9ea4726e5a0
# ╠═4ec9ee87-cda2-41be-960f-3508dbb8ed79
# ╠═4151ae3c-2e21-46ce-8ae8-52f59b2c9e1f
# ╟─e1d68d9e-0e84-4b22-8eff-094e45c47690
# ╠═ccba55e1-a141-40a3-b8f9-44eac06e5ff2
# ╠═d5b7ade0-5a0a-487b-9a01-146aecae4ced
# ╠═cb745154-ec84-47a9-a6e0-2f627fb3443d
# ╠═f2f916cf-afca-41e1-b9f2-486e7172bae2
# ╠═afea2042-1b56-40c4-82e0-08667b9e8732
# ╠═46f7e65d-7b8c-48f6-844a-1e1eda2c03cb
# ╠═64ee4253-5e3a-4155-97ba-4fecf17977a3
# ╠═d7f2fc1f-9192-4523-8b0e-5654e32b3473
# ╟─09f265c6-c8ce-48a8-a988-13080e8dad86
# ╠═b02ce499-3f39-4587-9831-c4536708cd5a
# ╠═26a38e63-179e-4a49-9997-965ed7c87c8c
# ╠═a40ef7dc-98a8-4dba-b313-f9aad0514a92
# ╟─61099ccd-488a-4092-947b-20892c1735a3
# ╠═f77a5e0b-306e-40bc-8265-2e1fa8765677
# ╠═83aa7d23-1667-426c-996c-512f1ad2ac7c
# ╠═30b10da5-9d9e-4036-b765-5860cdecee11
# ╠═37c7aab8-f7f9-4d89-b252-bb0a58928045
# ╟─e3fa92d0-53dd-4652-9f7d-ffeaaca7f287
# ╠═17413920-d32e-41af-803a-aaf4d74de92a
# ╠═45be7047-b1d2-4636-a78f-44a8b9bf48b3
# ╟─2f0ba514-f49a-4ed9-9c80-590e2a262826
# ╠═6657ac9c-a51e-4322-8074-894752845ac4
# ╠═38df5279-98f0-4177-8654-4a20981d4d29
# ╠═3ef5e86b-b257-4426-98f4-8a3a26e3aab9
# ╠═db1cf899-ff94-44cc-8f1e-2d0a9b5fa446
# ╟─6bfc8039-60f6-4628-8afe-706322df6b52
# ╠═3e44bf89-341a-4342-a861-1cddb7caa690
# ╠═bd1a5abd-9583-43fd-adcd-a8d52fa40735
# ╟─ac41688b-627a-4193-9f82-8461cd5b6c99
# ╠═4ea97308-bf2f-4968-8d27-30b5b39c2c0b
# ╠═87166a10-b8f4-4d5b-b2c9-ad9e1fdf6448
# ╠═b7068b37-5206-43ec-b168-150d9cc48642
# ╠═0ab611b9-96ba-4c47-b67e-f09fb855a2dc
# ╟─e08485b7-06b8-422d-ac15-fb0161714562
# ╠═38dbc04d-3419-44a3-a42a-77c8e9193676
# ╠═4d924012-ff6e-4147-9b94-d770035f60eb
# ╟─426f5b60-62ad-4365-94cd-0aa40779ee4e
# ╠═a49c445f-797c-4337-b04c-fc78ba6171e5
# ╠═4474b085-3cad-4759-b075-7027e22cd399
# ╠═65ed213f-4fd9-4e54-aaeb-8cb0598f090f
# ╠═3ef7ff2d-a1f2-480e-b537-fc9e0d363f59
# ╠═6019467c-1e9a-406f-b8f8-7c584329ff7b
# ╠═3f67cae9-c427-4481-9514-612be1605801
# ╠═40a716f4-3e26-4311-8bae-48ec38f0185c
# ╠═fab3d8a9-0e7b-4395-b295-d3b571eddc81
# ╟─931b25f4-d638-4e08-9b04-a3ca114d6d6d
# ╟─b73d7db7-9c21-49b3-a7a7-1fd3b9e34393
# ╠═6651a85e-ce16-4487-a961-80f1dfa1521d
# ╟─45b9003b-89e2-47a0-a89a-bbb75e6ea53d
# ╠═73d47734-3ac9-4b9d-9068-decce1e43704
# ╠═fa6658b2-1257-4a4e-b68a-4404344cf849
# ╠═287ef1c1-498e-4acf-ad9e-53f169a6743f
# ╟─5178f5bc-1fdd-495d-a1ce-83941ce75770
# ╠═7bc16911-3343-4cd4-afc7-5c1efd1aaa4d
# ╠═6674da27-b796-4d34-a26c-0ed71e3dacdf
# ╠═62fe2c43-075e-49f4-82f2-7d8e6c923377
# ╠═28351a7b-0d24-4024-93c4-5a0895c332e6
# ╠═4d2d64d5-7f8d-461d-923b-9de01099ae51
# ╠═f694effb-eed2-4a87-bbec-8f72c4a86fd1
# ╠═73b4ef64-54e6-4f41-809e-683cab4b1960
# ╟─c5800a80-6066-4dcf-947f-6958404cbbdb
# ╠═4c5f25cf-c288-4b0f-a76e-19073521e9a5
# ╠═eb1201e0-ae5c-4c32-a207-a63a4f24f331
# ╠═5a44547b-7b68-46f6-9284-48af68e9c414
# ╟─b42ba603-0ee2-4a5e-9949-2ea18841686c
# ╠═dd46b267-c4ce-41a7-a279-4ee0288c6cf6
# ╠═85f6a9b3-d459-4b8f-9f83-6c5f321ca775
# ╟─e06a5b08-2e1f-408c-8826-5673735127d4
# ╠═daf9b955-82dc-42da-9e9c-f9720126cf57
# ╠═a2486aa6-e4d6-485a-ab21-0d9b1ecd45b8
# ╠═0cdc3e62-ca2f-4a14-8ecb-6c894b78be7c
# ╠═78746d5c-e09d-4236-9f05-8d5685c921ad
# ╟─41db29b5-c8ac-485f-9c2f-ed414adb51cb
# ╠═1b024dbe-000e-4be8-b8b8-dbeacb9cb1e7
# ╠═2a9dbc89-6726-46d7-97fb-2ad6740940b7
# ╠═a6c609b2-7629-4d0d-a537-741156090da7
# ╟─072ea2b0-eefb-44f5-aee0-7f4a3dee2f01
# ╠═e6930201-68d4-46d8-9e58-f13808d712d4
# ╠═94ff8a91-239a-46a8-aeee-c4c9b05d6153
# ╠═95d18772-46ab-4146-9238-f97919460c23
# ╠═437c1dae-d477-47bc-b992-f5765f06d593
# ╠═b20769a5-99e2-43b7-8f35-01daa7f99dc1
# ╟─10806fac-1aeb-4c6a-bf34-be6a96b92a10
# ╠═4d89d806-6c77-4101-a48d-828020072f58
# ╠═50a6ba23-7aa5-4d3c-a3ad-93c5a2a8ecef
# ╠═d9dcdea6-af3a-4f13-83a2-83cb17c72495
# ╠═b8c1dd36-bdf8-42cf-bdbb-b29865b61ce8
# ╠═8d8e4362-e295-441f-ac66-625421ed4690
# ╠═a4cf615d-032a-4dea-80b3-d5484039f89f
# ╠═63d0a9d3-9c5e-4f1e-a130-c31b396bbda7
# ╠═85b8c734-f733-4cf9-9b38-1d8f2c8da584
# ╟─f340e442-a8b8-452f-9065-e74008a4277e
# ╠═b4c57845-d272-4936-b8a1-3f4e4dfe892e
# ╠═d783a19f-4e70-402f-9df7-f7b961af4921
# ╟─a9837ac3-2a16-4aa6-8685-e18bc817dc3a
# ╠═aa563bc3-09d6-48fb-916f-1da02db28e3b
# ╠═54018004-8652-44c6-be50-2d17b6c78a44
# ╠═480da86e-76b5-4f8a-ab00-03e4929b2819
# ╠═900ff03a-926b-412b-9609-c3e1bb961eea
# ╟─f58f236a-8a65-43ef-8367-87be4e8770eb
# ╠═831d8e8e-56e2-49f5-8732-57b4fe5ffad3
# ╠═fb2c3dab-210d-4036-b0a0-f9577074709f
# ╟─48ca783b-4e34-48dc-acc9-8c7c98f59c4a
# ╠═b2f945bc-6f0e-47f0-b58c-5c326f1ceb16
# ╠═0c4db1ec-b131-4b2d-b347-bb0850183bad
# ╠═355ddbc3-6818-42a1-a262-b3ade162481d
# ╠═b17900f4-2551-49eb-8b32-4d23e4aac727
# ╠═7d52635d-6256-43bd-ae54-44fb50cbca6b
# ╠═d6c6c5ea-e7b2-48a4-bbb8-83d4dc2ed997
# ╠═d25d26f8-19ed-43c0-a25d-1d2de9098457
# ╠═ca5b5f3e-eca8-4a3f-a016-9a1d8e6ee586
# ╠═8bd05deb-f7bc-4bb8-bfbd-c1f90bd43dda
# ╠═1f824ece-d1f2-4795-8bd3-fe5f9d98234b
# ╠═9e3a2f11-a474-451d-a6a0-6e89f3a9345d
# ╠═e4276472-5c23-464a-9db8-aec4ce85baf4
