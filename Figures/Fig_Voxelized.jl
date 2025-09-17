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

# ╔═╡ 1e0f31c2-4a8e-11f0-1d64-6b53ace7d4fe
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ a874ccf6-77aa-4d6c-b483-b4860943f131
begin
	using BrainRBMjulia
	using LinearAlgebra: diagind
	using HDF5
	
	using CairoMakie
	using BrainRBMjulia: multipolarnrmseplotter!, idplotter!, dfsize, quantile_range, neuron2dscatter!, cmap_aseismic, polarnrmseplotter!
	using ColorSchemes: reverse, RdYlGn_9, cool

	CONV = @ingredients("conventions.jl")
	include(joinpath(dirname(Base.current_project()), "Misc_Code", "fig_saving.jl"))
end

# ╔═╡ fd2bf186-4f5c-48c7-bd05-03fcb25c068d
using Statistics

# ╔═╡ b3c0d22b-f925-4d42-9a1f-893ecee98099
using Clustering

# ╔═╡ d8514df1-f200-411e-b4dc-259395debb5f
using BrainRBMjulia:corrplotter!

# ╔═╡ ecf7149d-9a42-47f9-9f87-b41c80e6b0cc
begin
	using LinearAlgebra
	############################
	# 1. Extract a triangle as a vector
	############################
	"""
	    triangle_part(A; part = :upper) -> Vector
	
	Return a vector of the **strict** upper- or lower-triangular elements of a
	square matrix `A`, *excluding* the main diagonal.
	
	`part` may be `:upper` (default) or `:lower`.
	
	The elements are returned in Julia’s natural (column-major) order. If you
	prefer row-major ordering, replace the mask trick below with an explicit
	comprehension such as `[A[i, j] for i in 1:n, j in 1:n if i < j]`.
	"""
	function triangle_part(A::AbstractMatrix; part::Symbol = :upper)
	    @assert size(A,1) == size(A,2) "Matrix must be square"
	    n = size(A,1)
	
	    if part === :upper
	        mask = triu(trues(n, n), 1)   # true only above the diagonal
	    elseif part === :lower
	        mask = tril(trues(n, n), -1)  # true only below the diagonal
	    else
	        error("`part` must be :upper or :lower")
	    end
	
	    return A[mask]           # logical indexing → Vector{eltype(A)}
	end
	
	
	############################
	# 2. Mix two triangles (zero diagonal)
	############################
	"""
	    mix_triangles(A, B) -> Matrix
	
	Create matrix `C` whose strict upper triangle comes from `A`,
	strict lower triangle comes from `B`, and whose diagonal is zero.
	`A` and `B` must be square and of the same size.
	"""
	function mix_triangles(A::AbstractMatrix, B::AbstractMatrix)
	    @assert size(A) == size(B) "Matrices must have the same dimensions"
	    C = triu(A, 1) + tril(B, -1)      # neither call includes the diagonal
	    return C
	end

end

# ╔═╡ ddf1132e-3f96-443b-89b8-36a883a8e088
using Random: shuffle, bitrand

# ╔═╡ 7908ce1e-df3f-4b20-808a-382b7c7171e0
TableOfContents()

# ╔═╡ ec4c0f9f-20db-43a5-8647-7a2c0b259de0
set_theme!(CONV.style_publication)

# ╔═╡ ffe1ad1a-04c0-4f4a-bcc2-306bd8d6ffa6
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ 1a46c70c-3e25-49f4-bf18-784fe16f55fb
begin
	using Glob
	function crossval_eval_loader(v, M, λ, fish="MarianneHawkins")
		# basedir = joinpath(
		# 	CONV.MODELPATH,
		# 	"Voxelized/CrossValidation",
		# )
		# files = glob(
		# 	"$(fish)_VOX$(v)_M$(M)_l2l1$(λ)_rep*.h5",
		# 	basedir
		# )
		files = LOAD.load_voxRBMs("CrossValidation", "$(fish)*_VOX$(v)_M$(M)_l2l1$(λ)_rep")
		return load_brainRBM_eval(files; ignore="<v>")
	end
end

# ╔═╡ d0624ba1-c1aa-4b9d-9ae1-d1d3e079376d


# ╔═╡ c44354cd-28aa-4df8-91b7-f7eadfa24e39
md"""
# 0. Fish and training base
"""

# ╔═╡ bc79c953-18cc-43d7-9dd9-6a6a5a32dcdf
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ 06dd898e-3c97-456b-93b0-9a2e606db2a2
md"Use numbers instead of fish names $(@bind numbered CheckBox(default=true))"

# ╔═╡ 594d27b0-b8a0-4aef-aa78-b0642c4f2f7d
vox_size = 20.

# ╔═╡ 05d603b3-f8e0-4c33-94ff-91982bf81dd5
base_mod = "_M40_l2l10.1";

# ╔═╡ bb4c6151-d198-473d-b60d-9bae7eedd9bc
md"example teacher : $(@bind EXteacher Select(FISH, default=FISH[1]))"

# ╔═╡ 0f98d940-62b2-49bf-844d-821acb5fc90d
STUDENTS = [fish for fish in FISH if fish!=EXteacher];

# ╔═╡ f09a670f-947b-4004-b4bf-f451e87626a9
md"example student : $(@bind EXstudent Select(STUDENTS, default=STUDENTS[2]))"

# ╔═╡ 313f9d91-d355-4b4e-8e82-9aee2e5d2049
begin
	if numbered
		FISH_DISP = ["F$i" for i in 1:length(FISH)]
		STUDENT_DISP = ["Student $i" for i in 1:length(STUDENTS)]
	else
		FISH_DISP = FISH
		STUDENT_DISP = STUDENTS
	end
end;

# ╔═╡ a94c89ca-e407-4448-b2ca-934ad08a8583


# ╔═╡ 409c48fa-f424-4ab5-8af2-bebf79d4cee5
multiRBM_path = LOAD.load_voxRBM(
	"", 
	"vRBMr_multivoxelized_$(length(FISH))fish_$(vox_size)vox$(base_mod)"
)

# ╔═╡ 9e235d4e-976d-4858-ad5b-e69d2adb9613


# ╔═╡ 5a48755f-cd04-4ca7-b141-5a3b7e7956c0
md"""
# 1. Main Figure
"""

# ╔═╡ 1fd4a66d-b8e7-48e0-8b4b-cc6227fb4635
begin
	fig_main = Figure(size=(53, 49).*(4,4).*(4/3/0.35))
	
	g_a = fig_main[1, 1:4] = GridLayout()
	
	g_bc = fig_main[2, 1:4] = GridLayout()
	g_b = g_bc[1, 1:2] = GridLayout()
	g_c = g_bc[1, 3:4] = GridLayout()

	g_DtoJ = fig_main[3:4, 1:4] = GridLayout()
	g_d = g_DtoJ[1,1] = GridLayout()
	g_e = g_DtoJ[1,2] = GridLayout()
	g_f = g_DtoJ[1,3] = GridLayout()
	g_g = g_DtoJ[1,4] = GridLayout()
	g_h = g_DtoJ[2,1] = GridLayout()
	g_i = g_DtoJ[2,2] = GridLayout()
	g_j = g_DtoJ[2,3] = GridLayout()
	g_k = g_DtoJ[2,4] = GridLayout()

	for (label, layout) in zip(
		["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"], 
		[g_a, g_b, g_c, g_d, g_e, g_f, g_g, g_h, g_i, g_j, g_k]
	)
	    Label(layout[1, 1, TopLeft()], label,
	        fontsize = Makie.current_default_theme().Axis.titlesize.val,
	        font = :bold,
	        padding = (0, 5, 5, 0),
	        halign = :right)
	end
end

# ╔═╡ b64106e0-0000-4732-9c55-7dab86eb74d0


# ╔═╡ b9dfbce8-699c-4b8e-90d2-48932af4e361
md"""
## 1.D-E. V Stats
"""

# ╔═╡ a7038417-19fa-48e6-8cec-245c38b8549b
ACTS = [
	load_data(LOAD.load_dataVOX(fish, vox_size)).spikes for fish in FISH
];

# ╔═╡ 40837778-55e1-4a02-9033-636829cc5209


# ╔═╡ 29931e52-3d41-4f9f-a6b5-5baac64ec1b4
md"""
### 1.D-E. V corr
"""

# ╔═╡ 85225d68-9e87-40fd-990f-48092dafdc12
vCORRS = [triangle_part(cor(act')) for act in ACTS];

# ╔═╡ d5ac9557-fd7a-4dfe-acd1-837f43ed90e3


# ╔═╡ deea6b78-c7e3-4f9f-a286-6f7cea479299
md"""
#### 1.D. Example
"""

# ╔═╡ 4f8efba0-002a-4c82-a010-046b69f628b1
begin
	# fig_vcorr_ex = Figure()
	ax_vcaorr_ex = Axis(
		g_d[1,1],
		aspect=1,
		xlabel="ρmn , Fish $(findfirst(FISH .== EXteacher))",
		ylabel="ρmn , Fish $(findfirst(FISH .== EXstudent))",
	)

	H_vcaorr_id = idplotter!(
		ax_vcaorr_ex,
		vCORRS[findfirst(FISH .== EXteacher)], 
		vCORRS[findfirst(FISH .== EXstudent)],
	)

	
	# Colorbar(g_d[1,2], label="Density", height=Relative(0.85), colormap=H_vcaorr_id.colormap.val, colorrange=H_vcaorr_id.colorrange, scale=H_vcaorr_id.scale)
	# colgap!(g_d,1,5)
	
	# fig_vcorr_ex
end

# ╔═╡ 3360c555-14e8-4c20-8dca-05ecd47914b6
md"""
####  1.E. All pairs
"""

# ╔═╡ 11f81da5-296f-4a58-a647-56d66c8e4ae3
begin
	vcorr_nRMSE_matrix = Matrix{Float64}(undef, length(FISH), length(FISH))
	for (i,f1) in enumerate(FISH)
		for (j,f2) in enumerate(FISH)
			if i == j
				vcorr_nRMSE_matrix[i,i] = NaN
			else
				vcorr_nRMSE_matrix[i,j] = nRMSE(
					vCORRS[findfirst(FISH .== f1)], 
					vCORRS[findfirst(FISH .== f2)]
				)
			end
		end
	end
end

# ╔═╡ dbcef2ba-acfd-466c-af3d-c1905049d72e
begin
	# fig_vcorr_all = Figure()
	ax_vcaorr_all = Axis(
		g_e[1,1],
		aspect=1,
	    xticks=((1:length(FISH)), FISH_DISP),
	    yticks=((1:length(FISH)), FISH_DISP),
		xticklabelrotation = π/4,
		xlabel = "Fish i", ylabel="Fish j",
	)

	H_vcaorr_all = heatmap!(
		ax_vcaorr_all,
		vcorr_nRMSE_matrix,
		colormap=CONV.CMAP_GOODNESS,
		colorrange=(0,1),
	)
	Colorbar(g_e[1,2], H_vcaorr_all, label="nRMSE ρmn", height=Relative(0.85))
	colgap!(g_e,1,5)
	
	# fig_vcorr_all
end

# ╔═╡ 1e166691-f70d-4f1b-a696-37758783ea56


# ╔═╡ 3c2ecda1-36de-4d38-918c-c0fe0e6064b7
md"""
## 1.F-G-I-J. H stats
"""

# ╔═╡ f0611a3e-db56-4afd-b086-fc5e22fe9668
mrbm, _,mrbm_eval,mrbm_dsplit,mrbm_gen,_ = load_brainRBM(multiRBM_path)

# ╔═╡ bafe62c1-4481-43ab-b919-894923c9c9f4
HACTS = [translate(mrbm, act) for act in ACTS];

# ╔═╡ bec186b7-4b0d-4366-ae65-ab570afbdf0a


# ╔═╡ 9e27ead6-7261-422f-96a8-aaaf0ef858f5
md"""
### 1.F-G. H corr
"""

# ╔═╡ 5b54efc6-53ce-436f-ac6b-8c2b9e62d832
hCORRS = [triangle_part(cor(hact')) for hact in HACTS];

# ╔═╡ bc94ea3f-8f99-498e-a156-9615e4b272ff
begin
	hCORR_X, hCORR_Y = [], []
	for i in 1:length(FISH)
		for j in i+1:length(FISH)
			push!(hCORR_X, hCORRS[i])
			push!(hCORR_Y, hCORRS[j])
		end
	end
	hCORR_x = vcat(hCORR_X...)
	hCORR_y = vcat(hCORR_Y...)
end;

# ╔═╡ 874d0c99-3bb2-4899-b7b4-f19c126bebd2
md"""
#### 1.F. Example
"""

# ╔═╡ 5dabbf30-b56e-46bb-b346-fc8ea74bae69
begin
	# fig_hcorr_ex = Figure()
	ax_hcaorr_ex = Axis(
		g_f[1,1],
		aspect=1,
		xlabel="ρμν , Fish $(findfirst(FISH .== EXteacher))",
		ylabel="ρμν , Fish $(findfirst(FISH .== EXstudent))",
	)

	idplotter!(
		ax_hcaorr_ex,
		# hCORR_x, hCORR_y,
		hCORRS[findfirst(FISH .== EXteacher)], 
		hCORRS[findfirst(FISH .== EXstudent)],
		# switch_thresh=1,
		# bins=50
		color=(:black, 0.25)
	)
	
	# fig_hcorr_ex
end

# ╔═╡ e1a2aaf3-89be-4d75-a466-8a411a5b611f


# ╔═╡ bc6a323d-8d86-44d3-b1d8-977aa9fc16d9
md"""
#### 1.G. All pairs
"""

# ╔═╡ bf90ee77-0a7c-48d7-b338-62c8edb9fd57
begin
	hcorr_nRMSE_matrix = Matrix{Float64}(undef, length(FISH), length(FISH))
	for (i,f1) in enumerate(FISH)
		for (j,f2) in enumerate(FISH)
			if i == j
				hcorr_nRMSE_matrix[i,i] = NaN
			else
				hcorr_nRMSE_matrix[i,j] = nRMSE(
					hCORRS[findfirst(FISH .== f1)], 
					hCORRS[findfirst(FISH .== f2)]
				)
			end
		end
	end
end

# ╔═╡ e7822746-8055-4e56-80d2-df735b53378d
begin
	# fig_hcorr_all = Figure()
	ax_hcaorr_all = Axis(
		g_g[1,1],
		aspect=1,
	    xticks=((1:length(FISH)), FISH_DISP),
	    yticks=((1:length(FISH)), FISH_DISP),
		xticklabelrotation = π/4,
		xlabel = "Fish i", ylabel="Fish j",
	)

	H_hcaorr_all = heatmap!(
		ax_hcaorr_all,
		hcorr_nRMSE_matrix,
		colormap=CONV.CMAP_GOODNESS,
		colorrange=(0,1),
	)
	Colorbar(g_g[1,2], H_hcaorr_all, label="nRMSE ρμν", height=Relative(0.85))
	colgap!(g_g,1,5)
	
	# fig_hcorr_all
end

# ╔═╡ 303571cc-9852-49ea-bec8-ce15d1dfe796
md"""
### 1.J-K. H mean
"""

# ╔═╡ a577ab41-093b-4939-b1f1-26c843ccb467
hMEANS = [mean(hact, dims=2)[:,1] for hact in HACTS];

# ╔═╡ 4d73d700-e37e-4369-81b7-d24b5fec7ede
begin
	hMEANS_X, hMEANS_Y = [], []
	for i in 1:length(FISH)
		for j in i+1:length(FISH)
			push!(hMEANS_X, hMEANS[i])
			push!(hMEANS_Y, hMEANS[j])
		end
	end
	hMEANS_x = vcat(hMEANS_X...)
	hMEANS_y = vcat(hMEANS_Y...)
end;

# ╔═╡ c03b36c3-02af-44cf-bfba-8c063e53d1e6


# ╔═╡ 1954c0e1-7a94-44b4-bb80-154dfdbfd080
md"""
#### 1.I. Example
"""

# ╔═╡ 2df609a6-00fe-4ec4-ae0e-77d3d623426b
begin
	# fig_hmean_ex = Figure()
	ax_hmean_ex = Axis(
		g_j[1,1],
		aspect=1,
		xlabel="⟨hμ⟩ - Fish $(findfirst(FISH .== EXteacher))",
		ylabel="⟨hμ⟩ - Fish $(findfirst(FISH .== EXstudent))",
	)

	idplotter!(
		ax_hmean_ex,
		# hMEANS_x, hMEANS_y,
		hMEANS[findfirst(FISH .== EXteacher)], 
		hMEANS[findfirst(FISH .== EXstudent)],
		# switch_thresh=1,
		# bins=50
		# color=(:black, 0.25)
	)
	
	# fig_hmean_ex
end

# ╔═╡ d2bf01a7-d6a4-46ce-8a50-0d41c4b2c2b5


# ╔═╡ 561d263b-9a19-4b4d-ae61-8b5a22d99a62
md"""
####  1.J. All pairs
"""

# ╔═╡ ef1d3c49-2191-407b-9d2a-ad8b072b7b92
begin
	hmean_nRMSE_matrix = Matrix{Float64}(undef, length(FISH), length(FISH))
	for (i,f1) in enumerate(FISH)
		for (j,f2) in enumerate(FISH)
			if i == j
				hmean_nRMSE_matrix[i,i] = NaN
			else
				hmean_nRMSE_matrix[i,j] = nRMSE(
					hMEANS[findfirst(FISH .== f1)], 
					hMEANS[findfirst(FISH .== f2)]
				)
			end
		end
	end
end

# ╔═╡ e41c98ed-209a-4d39-92c9-66316ee23c03
begin
	# fig_hmean_all = Figure()
	ax_hmean_all = Axis(
		g_k[1,1],
		aspect=1,
	    xticks=((1:length(FISH)), FISH_DISP),
	    yticks=((1:length(FISH)), FISH_DISP),
		xticklabelrotation = π/4,
		xlabel = "Fish i", ylabel="Fish j",
	)

	H_hmean_all = heatmap!(
		ax_hmean_all,
		hmean_nRMSE_matrix,
		colormap=CONV.CMAP_GOODNESS,
		colorrange=(0,1),
	)
	Colorbar(g_k[1,2], H_hmean_all, label="nRMSE ⟨hμ⟩", height=Relative(0.85))
	colgap!(g_k,1,5)
	
	# fig_hmean_all
end

# ╔═╡ 72d0049b-a0c6-46f0-a58a-8bccf9f63549


# ╔═╡ 7a27f5e2-c75e-4133-9dfe-d4ab112aaed2
md"""
## 1.B-C. ACtivities
"""

# ╔═╡ 43ee56fb-2440-4152-8573-7bf5de402023
md"""
### 1.B. Voxels
"""

# ╔═╡ d5f67952-1093-48e5-a8d6-2885bcbd1328
begin
	vC = cor(ACTS[1]')
	v_act_HCLUST = hclust(1 .- vC, linkage=:ward, branchorder=:optimal)
	v_act_order = v_act_HCLUST.order
end;

# ╔═╡ 6911b174-1fa9-4e33-ba05-cdd41423de92
begin
	show_acts = []
	for (i,fish) in enumerate(FISH)
		if fish ∈ [EXstudent, EXteacher]
			push!(show_acts, ACTS[findfirst(FISH .== fish)])
		else
			push!(
				show_acts, 
				hcat(
					ACTS[findfirst(FISH .== fish)][:,1:150],
					ACTS[findfirst(FISH .== fish)][:,1:150].*0,
					ACTS[findfirst(FISH .== fish)][:,end-150:end]
				)
			)
		end
	end
end

# ╔═╡ 0de28d8d-70f2-4d44-a293-029535abd084
begin
	l_vact = 1.8
	show_acts_lenghts = [size(a,2) for a in show_acts]
	
	# fig_v_act = Figure(size=dfsize().*(2,0.6))
	
	ax_v_act = Axis(
		g_b[1,1],
		leftspinevisible=false, bottomspinevisible=false,
		xticksvisible=false, xticklabelsvisible=false,
		yticksvisible=false, yticklabelsvisible=false,
		aspect=2.5,
	)
	H_v_act = heatmap!(
		ax_v_act,
		hcat(show_acts...)[v_act_order,:]',
		colormap=:berlin,
		colorrange=(-l_vact, +l_vact),
		rasterize=5, interpolate=true,
	)

	Colorbar(g_b[1,2], H_v_act, label="Voxel Activity", height=Relative(0.85))
	colgap!(g_b,1,5)

	for i in 1:length(show_acts)
		vlines!(
			ax_v_act,
			cumsum(show_acts_lenghts),
			color=:white,
			linewidth=0.5,
		)
	end
	
	# fig_v_act
end

# ╔═╡ 9c8155e6-578d-41f4-b0a8-a469412b0574


# ╔═╡ 586c6205-545b-4d5d-b7ea-5cafcf2d0115
md"""
### 1.C. HUs
"""

# ╔═╡ e0b02c22-2674-40bd-9f3c-04f64fea6500
begin
	show_actsh = []
	for (i,fish) in enumerate(FISH)
		if fish ∈ [EXstudent, EXteacher]
			push!(show_actsh, HACTS[findfirst(FISH .== fish)])
		else
			push!(
				show_actsh, 
				hcat(
					HACTS[findfirst(FISH .== fish)][:,1:150],
					HACTS[findfirst(FISH .== fish)][:,1:150].*0,
					HACTS[findfirst(FISH .== fish)][:,end-150:end]
				)
			)
		end
	end
end

# ╔═╡ afa72f3b-1d55-4a59-ae3e-6184eeeb1448
begin
	l_hact = 1.5
	show_actsh_lenghts = [size(a,2) for a in show_actsh]
	
	# fig_h_act = Figure(size=dfsize().*(2,0.6))
	
	ax_h_act = Axis(
		g_c[1,1],
		leftspinevisible=false, bottomspinevisible=false,
		xticksvisible=false, xticklabelsvisible=false,
		yticksvisible=false, yticklabelsvisible=false,
		aspect=2.5,
	)
	H_h_act = heatmap!(
		ax_h_act,
		hcat(show_actsh...)',
		colormap=Reverse(:vanimo),
		colorrange=(-l_hact, +l_hact),
		rasterize=5, interpolate=false,
	)

	Colorbar(g_c[1,2], H_h_act, label="Hidden Activity", height=Relative(0.85))
	colgap!(g_c,1,5)

	for i in 1:length(show_actsh)
		vlines!(
			ax_h_act,
			cumsum(show_actsh_lenghts),
			color=:white,
			linewidth=0.5,
		)
	end
	
	# fig_h_act
end

# ╔═╡ 3dc4df0a-d7bf-4695-85fc-5e7f7e3a3687


# ╔═╡ 43e4f988-2e3d-4f8e-921b-3d122d2c474d
md"""
## 1.H. corr Comparison
"""

# ╔═╡ 3bb9562b-2009-4174-baf3-5dc24422dd24
begin
	hcorr_nRMSE_flat = triangle_part((hcorr_nRMSE_matrix .+ hcorr_nRMSE_matrix') ./ 2)
	vcorr_nRMSE_flat = triangle_part((vcorr_nRMSE_matrix .+ vcorr_nRMSE_matrix') ./ 2)
end

# ╔═╡ 5c5f9c1e-1817-4089-b186-b55a0c0d3870
begin
	using HypothesisTests: MannWhitneyUTest, pvalue
	stest = MannWhitneyUTest(vcorr_nRMSE_flat,hcorr_nRMSE_flat)
	pval = pvalue(stest, tail=:right)
end

# ╔═╡ f554e469-3fd9-4ecb-a1b7-4b0a386e403b
begin
	# fig3 = Figure()
	ax_corrcomp = Axis(
		g_h[1,1],
		ylabel="nRMSE",
		xticks=([0,1], ["ρmn","ρμν"]),
		bottomspinevisible=false,
		xticksvisible=false,
	)
	x_v = zeros(size(vcorr_nRMSE_flat)) .+ exp.(-(abs.(vcorr_nRMSE_flat .- mean(vcorr_nRMSE_flat))./0.1).^2) .* 0.15 .* randn(size(vcorr_nRMSE_flat))
	x_h = ones(size(hcorr_nRMSE_flat)) .+ exp.(-(abs.(hcorr_nRMSE_flat .- mean(hcorr_nRMSE_flat))./0.2).^2) .* 0.15 .* randn(size(hcorr_nRMSE_flat))

	for i in 1:length(vcorr_nRMSE_flat)
		lines!(
			ax_corrcomp,
			[x_v[i], x_h[i]], 
			[vcorr_nRMSE_flat[i], hcorr_nRMSE_flat[i]],
			color=(:grey, 0.25)
		)
	end
	
	scatter!(
		ax_corrcomp,
		x_v,
		vcorr_nRMSE_flat,
		color=:orange
	)
	vio_v = violin!(
		ax_corrcomp,
		zeros(size(vcorr_nRMSE_flat)), vcorr_nRMSE_flat, 
		color=(:orange, 0.5)
	)
	
	scatter!(
		ax_corrcomp,
		x_h,
		hcorr_nRMSE_flat,
		color=:green
	)
	violin!(
		ax_corrcomp,
		ones(size(hcorr_nRMSE_flat)), hcorr_nRMSE_flat, 
		color=(:green, 0.5)
	)

	bracket!(
		ax_corrcomp,
		0,0.9,1,0.9, 
		style = :square, 
		text="pval < $(ceil(pval, sigdigits=1))", 
		width=3
	)
	
	ylims!(ax_corrcomp,0,1.05)
	# fig3
end

# ╔═╡ 65640112-8b2d-479d-b36a-aff63ea258c5


# ╔═╡ 0e9229c5-ddda-4aa3-aa26-2f5a09ce2276


# ╔═╡ e932f290-1da7-4b6c-b10d-6ad7259ff2aa
md"""
## 1.I. Free Energy
"""

# ╔═╡ 77aaaf76-9992-4b74-a638-3ff9e35c7687
md"""
### Multifish
"""

# ╔═╡ 32b884b6-9d77-4010-8908-a65618231567
FES = [free_energy(mrbm, act) for act in ACTS];

# ╔═╡ 499428c3-8f0f-4ec4-9df5-011d352bfd3a
begin
	# fig_FE = Figure()
	ax_FE = Axis(
		g_i[1,1], 
		yscale=log10,
		ylabel="Density",
		xlabel="Free Energy , F(v)",
		xtickformat = xs -> ["$(round(x/1.e4, digits=2))" for x in xs],
	)
	Label(g_i[1,1,Right()], halign=:right, valign=:bottom, "×10⁴")
	for i in 1:length(FISH)
		density!(
			ax_FE, FES[i], 
			offset=1.e-5,
			strokewidth=2, strokecolor=(:black, 0.25),
			color=(:white, 0),
		)
	end
	xlims!(ax_FE, -1.e3, 1.e4)
	# fig_FE
end

# ╔═╡ f11d7726-1bbd-4c15-8664-2ec2c1ac10f2
md"""
## 1.A-right. Sample Activity
"""

# ╔═╡ 3749dd59-5ac9-4fe1-b8a8-e5af92c6c7af
begin
	ex_voxels = [1690,1691,825,1686,1688]
	colors = ["magenta", "blue", "green"]
	#fig = Figure(size=dfsize().*(2,1))
	g_aR = g_a[1,2] = GridLayout()
	axes = []
	for (i,v) in enumerate(ex_voxels)
		for (j,f) in enumerate([1,3,6])
			ax = Axis(
				g_aR[i,j],
				leftspinevisible=false, bottomspinevisible=false,
				xticklabelsvisible=false, yticklabelsvisible=false,
				xticksvisible=false, yticksvisible=false,
			)
			lines!(ax, ACTS[f][v,1000:1250], color=colors[j])
			push!(axes, ax)
		end
	end
	linkaxes!(axes...)
	colgap!(g_aR, 0)
	rowgap!(g_aR, 0)
	
	#fig
end

# ╔═╡ d29c0358-f5c0-421b-9814-35fa0b170ec5


# ╔═╡ fcff0be2-e1e8-4336-8f69-02f8794d105f


# ╔═╡ 9259472c-850d-4cc5-91e3-b7c4ffe8ad99
md"""
## 1.END. Adjustments
"""

# ╔═╡ 4ef6daa2-824f-4904-b6e6-2b1ca586cd13
all_axes_main = [ax for ax in fig_main.content if typeof(ax)==Axis];

# ╔═╡ ac6bf797-ffc5-4036-a9f2-4b16c5f4fd44
for ax in all_axes_main
	ax.alignmode = Mixed(left=0)
end

# ╔═╡ 0daee0c6-0030-43bb-be4e-414f18553d83
fig_main

# ╔═╡ a444dc6b-2201-4d3e-80ec-1257e24eb002
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
save(@figpath("main"), fig_main)
  ╠═╡ =#

# ╔═╡ 696bf3a1-5645-46bc-8f0f-e50b8f47c4ae


# ╔═╡ 346ebf26-0a08-438a-b018-62225ee87312


# ╔═╡ 64a75b7a-a9c7-4a7a-bf12-ec40de222280


# ╔═╡ dd655114-a3b1-4ad6-abb2-8076e790cd99
md"""
# 2. Supplementaries
"""

# ╔═╡ 20299104-25c9-4fb7-9bc0-bfe393a6ae00
md"""
## 2.1. Voxel size and Neurons
"""

# ╔═╡ ff6c386a-d149-42c5-9df9-10f3179aa248
voxelizations = [LOAD.load_voxgrid(vsize) for vsize∈[10., 15., 20., 25., 30., 35., 40., 45., 50.]]

# ╔═╡ f561975b-2ff2-4ab1-be72-f8418066e468
begin
	VOXSIZE = []
	N_VOX = []
	N_NEURONPERVOX = []
	FRAC_NEURON_CONCERVED = []
	FRAC_OCCUPIED_SPACE = []
	CORR_NRMSES = []
	for vox_path in voxelizations
		vox = VoxelGrid(vox_path)
		vox_size = vox.voxsize[1]
		N_vox = size(vox.voxel_composition, 2)
		n_neuronsPERvox = [length(vox.voxel_composition[i,j]) for i=1:size(vox.voxel_composition, 1), j=1:size(vox.voxel_composition, 2)]
		frac_conserved_neurons = [mean(vox.neuron_affiliation[i] .> -1) for i in 1:length(vox.neuron_affiliation)]
		frac_occupied_space = N_vox / prod(vox.Ns)
		
		corrs = [cor(v) for v in vox.voxel_activities]
		nrmses = zeros(Float64, length(corrs), length(corrs))
		for i∈1:length(corrs)
			for  j∈i+1:length(corrs)
				nrmses[i,j] = nrmses[j,i] = nRMSE(corrs[i], corrs[j])
			end
		end
	
		push!(VOXSIZE, vox_size)
		push!(N_VOX, N_vox)
		push!(N_NEURONPERVOX, n_neuronsPERvox)
		push!(FRAC_NEURON_CONCERVED, frac_conserved_neurons)
		push!(FRAC_OCCUPIED_SPACE, frac_occupied_space)
		push!(CORR_NRMSES, nrmses)
	end
end

# ╔═╡ 8cbbe4e5-3745-4d42-9534-dcf6bdb28f33
n_nperv_quant = reduce(hcat, [quantile(vec(N_NEURONPERVOX[i]), [0,0.25,0.5,0.75,1]) for i in 1:length(N_NEURONPERVOX)]);

# ╔═╡ 6cc3a353-acef-4564-8cab-2c7589cfea1a
quant_corr_nrmse = hcat([quantile(triangle_part(a), [0.005, 0.25, 0.5, 0.75, 0.995]) for a in CORR_NRMSES]...);

# ╔═╡ 400cf119-b1c7-4b02-a3db-8eb437f53811
begin
	ref_vox = 20.;
	ref_vox_ind = findall(VOXSIZE .== ref_vox)[1]
end;

# ╔═╡ 26887565-2982-4b01-9bc2-5b5d9412027d
mean(N_NEURONPERVOX[ref_vox_ind]), std(N_NEURONPERVOX[ref_vox_ind])

# ╔═╡ 60aabdd5-4be7-4fd9-8878-03614a6fcffd
N_VOX[ref_vox_ind]

# ╔═╡ 50bae469-a625-401a-ae5d-e4708baf89f7
vox_color = cool[VOXSIZE ./ maximum(VOXSIZE)]

# ╔═╡ c4725180-a139-4da6-9da2-e18025b4519a
begin
	fig_S_vox_size = Figure(size=(53, 49).*(3.5,3).*(4/3/0.35))

	g_abd_S_vox_size = fig_S_vox_size[1,1:3] = GridLayout()
	g_a_S_vox_size = g_abd_S_vox_size[1,1] = GridLayout()
	g_b_S_vox_size = g_abd_S_vox_size[1,2] = GridLayout()
	g_d_S_vox_size = g_abd_S_vox_size[1,3] = GridLayout()
	g_c_S_vox_size = fig_S_vox_size[2,1:3] = GridLayout()
	g_efg_S_vox_size = fig_S_vox_size[3,1:3] = GridLayout()
	g_e_S_vox_size = g_efg_S_vox_size[1,1] = GridLayout()
	g_f_S_vox_size = g_efg_S_vox_size[1,2] = GridLayout()
	g_g_S_vox_size = g_efg_S_vox_size[1,3] = GridLayout()
	
	for (label, layout) in zip(
		["A", "B", "C", "D", "E", "F", "G"], 
		[g_a_S_vox_size, g_b_S_vox_size, g_c_S_vox_size, g_d_S_vox_size, g_e_S_vox_size, g_f_S_vox_size, g_g_S_vox_size]
	)
		Label(layout[1, 1, TopLeft()], label,
			fontsize = Makie.current_default_theme().Axis.titlesize.val,
			font = :bold,
			padding = (0, 5, 5, 0),
			halign = :right)
	end
end

# ╔═╡ 02f11ee2-bb9e-4c6a-8320-1b8c0ca81bdf
md"""
### 2.1.A. N of voxels
"""

# ╔═╡ 3ed88439-0007-4472-a946-db2586fdc60c
md"""
### 2.1.B. N of neurons
"""

# ╔═╡ 7bb8d15f-3673-46de-871f-2bf4fec13d40
md"""
### 2.1.C. neurons per voxel
"""

# ╔═╡ f09ce00c-e17f-4ae9-aea5-7de07f2417cf
begin
	ax_c_S_vox_size = Axis(
		g_c_S_vox_size[1,1],
		xlabel="Neurons per voxel", 
		ylabel="Density",
		yscale=log10,
		# xscale=log10,
	)
	for i in length(VOXSIZE):-1:1
		for j in 1:size(N_NEURONPERVOX[i],1)
			density!(
				ax_c_S_vox_size,
				N_NEURONPERVOX[i][j,:],
				offset=1.e-6,
				strokewidth=1,
				strokecolor=vox_color[i],
				color=(:white, 0),
				bandwidth=5,
				# boundary=(0, 500)
			)
		end
	end
	xlims!(ax_c_S_vox_size, 0, 300)
end

# ╔═╡ aec0dcf8-4b43-4727-8a96-a89bed1696b9
md"""
### 2.1.D. neuron per voxel distribs
"""

# ╔═╡ 011ddb46-1c3f-4b70-8b80-9a1a4d9742e8


# ╔═╡ abe3f23c-31ee-4f31-969c-8d5f7eb5153e
md"""
### 2.1.E-F. Example pairwise correlations
"""

# ╔═╡ 723ebadb-627e-4f2f-84e5-0dbb9bc115a2
begin
	vox_path = voxelizations[3]
	vox = VoxelGrid(vox_path)
	corr1 = cor(vox.voxel_activities[1])
	corr2 = cor(vox.voxel_activities[2])
end;

# ╔═╡ 69c8a3b5-0aeb-4c31-bb9a-a7b6cef38c08
begin
	ax_e_S_vox_size = Axis(
		g_e_S_vox_size[1,1], 
		aspect=1,
		xlabel="Voxel m",
		ylabel="Voxel n"
	)
	h_corr_S_vox_size = corrplotter!(ax_e_S_vox_size,
		mix_triangles(corr1, corr2), 
		order=hclust(
			1 .- corr2, 
			linkage=:ward, 
			branchorder=:optimal
		).order
	)
	lines!(ax_e_S_vox_size, [0,size(corr1, 1)], [0,size(corr1, 2)], color=:black)
	
	Colorbar(g_e_S_vox_size[1,2], h_corr_S_vox_size, label="Correlation , ρₘₙ", height=Relative(0.9))
	colgap!(g_e_S_vox_size, 0)
end

# ╔═╡ 961f299d-6c41-4dcd-b7af-07f919525b28
begin
	ax_f_S_vox_size = Axis(
		g_f_S_vox_size[1,1], 
		aspect=1,
		xlabel="ρₘₙ , Training 1",
		ylabel="ρₘₙ , Training 2",
	)
	h_id_S_vox_size = idplotter!(ax_f_S_vox_size,
		corr1, corr2
	)
	Colorbar(g_f_S_vox_size[1,2], colormap=h_id_S_vox_size.colormap.val, colorrange=h_id_S_vox_size.colorrange, scale=h_id_S_vox_size.scale, label="Density", height=Relative(0.9))
	colgap!(g_f_S_vox_size, 0)
end

# ╔═╡ d03b97ea-e509-410b-8ea5-1ab0a312ef84


# ╔═╡ 0c076ecf-60f0-4660-930f-f58672ac791f
md"""
### 2.1.G. nRMSE
"""

# ╔═╡ 6b9d78ef-7140-4806-a622-1795c1139550


# ╔═╡ 67056c02-f892-4d1c-8caf-3f4b5f95ed58
md"""
### 2.1.END. Ajustments
"""

# ╔═╡ 9827331c-48f0-4ff3-9b21-0b963644a064
all_axes_S_vox_size = [ax for ax in fig_S_vox_size.content if typeof(ax)==Axis];

# ╔═╡ b3658702-1de7-41c2-9a50-01bde846ac21
for ax in all_axes_S_vox_size
	ax.alignmode = Mixed(left=0)
end

# ╔═╡ 382183a0-d066-4b24-b7c5-dbf5bd09dd2b
fig_S_vox_size

# ╔═╡ 9680752c-17f9-491a-9459-034c0a43c573
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
save(@figpath("SUPP_voxelsize"), fig_S_vox_size)
  ╠═╡ =#

# ╔═╡ 4bfb9eb4-60d2-49fa-af17-b8defb6c51b9


# ╔═╡ d015c130-47c9-4ae6-a142-ba390906ef0d
md"""
## 2.1. Cross Validation
"""

# ╔═╡ 1565bcf6-7f53-41c8-9add-19b44fc40a20
begin
	fig_S_crossval = Figure(size=(53, 49).*(7,2).*(4/3/0.35))
	
	g_ab_S_crossval = fig_S_crossval[1:2,1:3] = GridLayout()
	g_a_S_crossval = g_ab_S_crossval[1,1:2] = GridLayout()
	g_b_S_crossval = g_ab_S_crossval[1,3] = GridLayout()
	g_abCB_S_crossval = g_ab_S_crossval[1,4] = GridLayout()
	g_c_S_crossval = fig_S_crossval[1:2, 4:5] = GridLayout()
	g_d_S_crossval = fig_S_crossval[1:2, 6:7] = GridLayout()

	for (label, layout) in zip(
		["A", "B", "C", "D"], 
		[g_a_S_crossval, g_b_S_crossval, g_c_S_crossval, g_d_S_crossval]
	)
	    Label(layout[1, 1, TopLeft()], label,
	        fontsize = Makie.current_default_theme().Axis.titlesize.val,
	        font = :bold,
	        padding = (0, 5, 5, 0),
	        halign = :right)
	end
end

# ╔═╡ 538f7212-d9ba-4b63-a3b5-956d143f3f82
begin
	tries = LOAD.load_voxRBMs("Repeats", "vRBMr_multivoxelized_6fish_20.0vox_M*_l2l1*_rep")
	# 	glob(
	# 	"vRBMr_multivoxelized_6fish_20.0vox_M*_l2l1*_rep*.h5", 
	# 	joinpath(CONV.MODELPATH, "Voxelized/Repeats"),
	# )
	tries_params = unique([( parse(Int,split(split(split(t,"vox")[end],"M")[end], "_")[1]), parse(Float64, split(split(split(t,"vox")[end],"l2l1")[end], "_")[1]) ) for t in tries])
	tries_sorted = [tries[contains.(tries, "M$(t[1])") .& contains.(tries, "l2l1$(t[2])")] for t in tries_params]
end

# ╔═╡ 1e46fa45-1d1b-4bda-a290-bfe38b56ff36


# ╔═╡ 5cb568a6-8124-4685-96a6-661935349790
function gaussian_density(μ::Real,
	σ::Real,
	xmin::Real,
	xmax::Real;
	n::Integer = 1000)
		@assert σ > 0 "σ (standard deviation) must be positive"
	xs = range(xmin, xmax; length = n)
	
	# Normalization constant 1/(σ√(2π))
	norm = inv(σ * sqrt(2π))
	
	# Element-wise PDF evaluation
	f = norm .* exp.(-0.5 .* ((xs .- μ) ./ σ) .^ 2)
	
	return xs, f
end

# ╔═╡ e3a97867-38f2-42d9-8582-5a77c2521737
begin
	# fig = Figure()
	ax_vdistrib_S = Axis(
		g_c_S_crossval[1,1], 
		yscale=log10,
		xlabel="Voxel Activity , v",
		ylabel="P(v)",
	)

	for act in ACTS
		density!(
			vec(act), 
			offset=1.e-7, 
			strokewidth=2, strokecolor=(:black, 0.2), color=(:white, 0),
		)
	end
	lines!(gaussian_density(0,1,-20,+20)..., color=:green, linestyle=:dash, linewidth=2)
	ylims!(1.e-7, 1.e0)
	xlims!(-20,+20)
	
	# fig
end

# ╔═╡ 2bc1fe15-8456-4684-b9d4-eb048a337239


# ╔═╡ 3113beef-1a37-451d-8c85-f8e0f30fc78b
begin
	vCOV = [cov(vact') for vact in ACTS];
	vCOV_tr = []
	for i in 1:length(vCOV)
		for j in 1:length(vCOV)
			if i == j
				continue
			end
			push!(vCOV_tr, tr(vCOV[i] * vCOV[j])/(size(vCOV[i],1)*size(vCOV[i],2)))
			# push!(vCOV_tr, tr(vCOV[i] .* vCOV[j]))
		end
	end

	hCOV = [cov(hact') for hact in HACTS];
	hCOV_tr = []
	for i in 1:length(hCOV)
		for j in 1:length(hCOV)
			if i == j
				continue
			end
			push!(hCOV_tr, tr(hCOV[i] * hCOV[j])/(size(hCOV[i],1)*size(hCOV[i],2)))
			# push!(hCOV_tr, tr(hCOV[i] .* hCOV[j]))
		end
	end
end

# ╔═╡ 46c1cd03-c30c-46aa-b716-165a757dc09c
begin
	# fig_o = Figure()
	
	xo_v = zeros(size(vCOV_tr)) .+ exp.(-(abs.(vCOV_tr .- mean(vCOV_tr))./0.2).^2) .* 0.15 .* randn(size(vCOV_tr))
	xo_h = ones(size(hCOV_tr)) .+ exp.(-(abs.(hCOV_tr .- mean(hCOV_tr))./0.2).^2) .* 0.15 .* randn(size(hCOV_tr))
	
	ax_o = Axis(
		g_d_S_crossval[1,1],
		ylabel=L"\frac{1}{n^2}\ \mathrm{Tr}( \mathrm{cov}(x^{f1}) \cdot \mathrm{cov}(x^{f2}) )",
		xticks=([0,1], ["voxels","hidden"]),
		bottomspinevisible=false,
		xticksvisible=false,
		yscale=log10,
	)

	for i in 1:length(vCOV_tr)
		lines!(
			ax_o,
			[xo_v[i], xo_h[i]], 
			[vCOV_tr[i], hCOV_tr[i]],
			color=(:grey, 0.25)
		)
	end
	
	# violin!(
	# 	ax_o,
	# 	zeros(size(vCOV_tr)), vCOV_tr, 
	# 	color=(:orange, 0.5),
	# 	datalimits=(1.e-9,1.e10)
	# )
	scatter!(
		ax_o,
		xo_v,
		vCOV_tr,
		color=:orange,
		markersize=15,
	)
	# violin!(
	# 	ax_o,
	# 	ones(size(hCOV_tr)), hCOV_tr, 
	# 	color=(:green, 0.5),
	# 	datalimits=(1.e-9,1.e10)
	# )
	scatter!(
		ax_o,
		xo_h,
		hCOV_tr,
		color=:green,
		markersize=15,
	)
	# fig_o
end

# ╔═╡ 5088865e-a2cc-43ec-bbaf-c9fdaaf80f84


# ╔═╡ 84256c48-9afe-4fb6-8c32-10c13b4a99cc
all_axes_Scrossval = [ax for ax in fig_S_crossval.content if typeof(ax)==Axis];

# ╔═╡ 1a25cdb0-398a-43bf-94fc-05e1bf292a25
for ax in all_axes_Scrossval
	ax.alignmode = Mixed(left=0)
end

# ╔═╡ 82441764-acd6-45f4-9606-4a0b87c8463c
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
save(@figpath("SUPP_voxelcrossval"), fig_S_crossval)
  ╠═╡ =#

# ╔═╡ 83b229c9-8a5f-4153-9dc2-68e84322ad90
fig_S_crossval

# ╔═╡ 1e3e54a8-fba6-4252-99ce-2745ed704d5f


# ╔═╡ 8a15c80d-187c-4404-80fc-888b4af12669


# ╔═╡ f4f849c1-ea33-469e-b6db-707b4954bbde
md"""
## 2.2. Stats RBM multifish
"""

# ╔═╡ a0dfe8f8-5f8b-4b23-8e60-6e0af2a73e22
begin
	t_fish_labs = vcat([zeros(Int,size(ACTS[i],2)).+i for i in 1:length(FISH)]...)
	indv_moments = MomentsAggregate[]
	for (i,fish) in enumerate(FISH)
		train_fish = mrbm_dsplit.train[:, t_fish_labs[mrbm_dsplit.train_inds] .== i]
		valid_fish = mrbm_dsplit.valid[:, t_fish_labs[mrbm_dsplit.valid_inds] .== i]
		tdset = DatasetSplit([0],[0], train_fish, valid_fish)
		push!(indv_moments, compute_all_moments(mrbm, tdset, mrbm_gen))
	end
	indv_moments_nrmse = [nRMSE_from_moments(m) for m in indv_moments];
end;

# ╔═╡ e34a4c5a-d073-46f9-aa51-03755355f641
stats = ["<vv> - <v><v>", "<vh>", "<h>", "<hh> - <h><h>"];

# ╔═╡ 923aae15-4077-4cd0-b01e-88c54bb385a4
begin
	fig_S_stats_all_fish = Figure(size=(53, 49).*(length(stats)/1.,length(FISH)/1.5).*(4/3/0.35))

	
	for (j,s) in enumerate(stats)
		axs = Axis[]
		for i in 1:length(FISH)
			ax = Axis(
				fig_S_stats_all_fish[i,j],
				aspect=1,
				xticklabelsvisible=false, yticklabelsvisible=false,
			)
			push!(axs, ax)
			if i == 1
				ax.title = stats[j]
			elseif i == length(FISH)
				ax.xticklabelsvisible = true
				ax.yticklabelsvisible = true
			end
			if j==1
				Label(
					fig_S_stats_all_fish[i, 0, Right()], 
					FISH_DISP[i], 
					padding = (0, 0, 0, 0), 
					rotation = pi/2,
					fontsize=Makie.current_default_theme().Axis.titlesize.val,
					font=Makie.current_default_theme().Axis.titlefont.val,
				)
			end
			if (i==length(FISH)) && (j==1)
				ax.xlabel = "data"
				ax.ylabel = "generated"
			end

			if s == "<vv> - <v><v>"
				crange = (1.e0, 1.e4)
				binminmax = (-1.5, +1.5)
			elseif s == "<vh>"
				crange = (1.e0, 1.e3)
				binminmax = (-1., +1.5)
			elseif s == "<h>"
				crange = (1.e0, 1.e3)
				binminmax = (-0.3, +0.7)
			elseif s == "<hh> - <h><h>"
				crange = (1.e0, 1.e2)
				binminmax = (-1, +2)
			end
	

			idplotter!(
				indv_moments[i].valid[s], 
				indv_moments[i].train[s],
				nrmselabelvisible=false,
				color=(:black, 0.25);
				crange, binminmax
			)
			xlims!(binminmax...)
			ylims!(binminmax...)
			
		end
		linkaxes!(axs...)
	end
	Colorbar(fig_S_stats_all_fish[6,5], colormap=:plasma, colorrange=(1.e0, 1.e3), label="Density", scale=log10)

	colsize!(fig_S_stats_all_fish.layout,0,10)
	
	fig_S_stats_all_fish
end

# ╔═╡ baa32e6a-1794-4816-841c-778860309b00
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
save(@figpath("SUPP_stats_indvs"), fig_S_stats_all_fish)
  ╠═╡ =#

# ╔═╡ 525f73b2-e861-458c-ac9e-cd6494c4460b


# ╔═╡ 9cb477dc-8110-4393-8593-19b3a84fb06f


# ╔═╡ 84583539-0394-4eaf-8ce0-fafad6a98687


# ╔═╡ 4fcc922f-b72e-4fcc-bf3b-d44fa70cae07


# ╔═╡ fedacb74-b3a1-4ee4-b6d0-e21401a4949a


# ╔═╡ cd030063-1435-4560-b6ef-f311ed57209f
md"""
# 3. Tools
"""

# ╔═╡ 611b5330-ff6a-4e48-8e7e-ff10287bbf78
function axis_intercepts_full!(ax::Makie.Axis, x::Real, y::Real; kwargs...)
	# make sure the limits are up to date
	autolimits!(ax)

	xmin, xmax = ax.xaxis.attributes.limits[]
	ymin, ymax = ax.yaxis.attributes.limits[]

	h = lines!(ax, [xmin, x], fill(y, 2); kwargs...)  # horizontal line
	v = lines!(ax, fill(x, 2), [ymin, y]; kwargs...)  # vertical   line

	xlims!(ax, xmin, xmax)
	ylims!(ax, ymin, ymax)
	return h, v
end

# ╔═╡ 8926908c-7541-402b-bf01-14329a8e9b85
begin
	ax_a_S_vox_size = Axis(
		g_a_S_vox_size[1,1],
		xlabel="Voxel Size (μm)", 
		ylabel="Number of Voxels",
	)
	lines!(ax_a_S_vox_size, VOXSIZE, N_VOX, color=vox_color)
	axis_intercepts_full!(
		ax_a_S_vox_size, 
		ref_vox, N_VOX[ref_vox_ind], 
		color=(:grey, 0.5), linestyle=:dash,
	)
end

# ╔═╡ 406b65ee-8f1f-4dcc-8068-fc66773ef81d
begin
	ax_b_S_vox_size = Axis(
		g_b_S_vox_size[:,:],
		xlabel="Voxel Size (μm)", 
		ylabel="Concerved neurons",
	)
	for i in 1:length(FRAC_NEURON_CONCERVED[1])
		lines!(ax_b_S_vox_size, 
			VOXSIZE, 
			[a[i] for a in FRAC_NEURON_CONCERVED], 
			color=vox_color,
		)
	end
	axis_intercepts_full!(
		ax_b_S_vox_size, 
		ref_vox, mean(FRAC_NEURON_CONCERVED[ref_vox_ind]), 
		color=(:grey, 0.5), linestyle=:dash,
	)
	ylims!(ax_b_S_vox_size, 0, 1)
end

# ╔═╡ de31e32e-9402-4987-aa07-9d6f93b33470
begin
	ax_d_S_vox_size = Axis(
		g_d_S_vox_size[1,1], 
		yscale=log10,
		xlabel="Voxel Size (μm)",
		ylabel="Neurons per voxel"
	)
	
	band!(ax_d_S_vox_size, 
		Float64.(VOXSIZE), 
		n_nperv_quant[1,:], n_nperv_quant[5,:], 
		color=vox_color,
		# alpha=0.3,
		rasterize=10
	)
	band!(ax_d_S_vox_size, 
		Float64.(VOXSIZE), 
		n_nperv_quant[2,:], n_nperv_quant[4,:], 
		color=vox_color,
		# alpha=0.3,
		rasterize=10
	)

	for (i,s) in enumerate([:dot, :dash, :solid, :dash, :dot])
		lines!(ax_d_S_vox_size, 
			Float64.(VOXSIZE), 
			n_nperv_quant[i,:],
			# color=:black,
			color=vox_color,
			linestyle=s,
		)
	end

	axis_intercepts_full!(
		ax_d_S_vox_size, 
		ref_vox, n_nperv_quant[3,ref_vox_ind], 
		color=(:grey, 0.5), linestyle=:dash,
	)
	
	
	ylims!(ax_d_S_vox_size, 1, 10^2.6)
end

# ╔═╡ 92ee9fbe-4418-43fd-946f-072efc7fd982
begin
	ax_g_S_vox_size = Axis(
		g_g_S_vox_size[1,1], 
		# yscale=log10,
		xlabel="Voxel Size (μm)",
		ylabel="nRMSE ρₘₙ"
	)
	
	band!(ax_g_S_vox_size, 
		Float64.(VOXSIZE), 
		quant_corr_nrmse[1,:], quant_corr_nrmse[5,:], 
		color=vox_color,
		# alpha=0.3,
		rasterize=10
	)
	band!(ax_g_S_vox_size, 
		Float64.(VOXSIZE), 
		quant_corr_nrmse[2,:], quant_corr_nrmse[4,:], 
		color=vox_color,
		# alpha=0.3,
		rasterize=10
	)

	for (i,s) in enumerate([:dot, :dash, :solid, :dash, :dot])
		lines!(ax_g_S_vox_size, 
			Float64.(VOXSIZE), 
			quant_corr_nrmse[i,:],
			# color=:black,
			color=vox_color,
			linestyle=s,
		)
	end

	axis_intercepts_full!(
		ax_g_S_vox_size, 
		ref_vox, quant_corr_nrmse[3,ref_vox_ind], 
		color=(:grey, 0.5), linestyle=:dash,
	)
	
	
	ylims!(ax_g_S_vox_size, 0, 1)
end

# ╔═╡ 9bd3fcc4-d2f7-4f4d-ad0f-a8b957aad43b
md"""
## 3.1. Couplings
"""

# ╔═╡ 30161e5e-9b2f-494c-8595-43d6e3b9dd02
begin
	function load_Jij(pathmodel::String, pathdata::String)
		rbm, _,_,_,_,_ = load_brainRBM(pathmodel)
		spikes = load_data(pathdata).spikes
		return coupling_approx(rbm, spikes)
	end
	function load_Jij(fish::String, rep::Int)
		rbmpath = LOAD.load_voxRBMs(
			"Repeats",
			"vRBMr_$(fish)*_VOX$(vox_size)$(base_mod)_rep$(rep)."
		)
		# 	LOAD.load_model(
		# 	joinpath(
		# 		CONV.MODELPATH, 
		# 		"Voxelized/Repeats",
		# 	),
		# 	"vRBMr_$(fish)*_VOX$(vox_size)$(base_mod)_rep$(rep)."
		# )
		datapath = LOAD.load_dataVOX(fish, vox_size)
			# LOAD.load_data(joinpath(CONV.DATAPATH, "Voxelized"), "VOX$(vox_size)_$(fish)*")
		return load_Jij(rbmpath, datapath)
	end
end

# ╔═╡ e194052d-60d3-4b3b-a61e-7632915f5929
md"""
## 3.2. Cross Validation
"""

# ╔═╡ d02400bc-48cf-44a0-8879-671951857795
begin
	λ21s = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
	Ms = [10, 20, 40, 60, 80]
	Vs = [10., 20., 30., 40., 50.]
	N_rep = 5
end

# ╔═╡ 68319e97-3304-4ff1-a60d-e0a07c4d2d28
begin
	cmap_max = nRMSEs_L4(
		crossval_eval_loader(20.0, Ms[1], λ21s[1])[1], 
		max=true
	)
	scale = 3
end

# ╔═╡ ea793074-e896-4feb-80a7-2f52ac2c74de
begin
	# fig =  Figure()
	ax_multi_tries_S = Axis(
		g_b_S_crossval[1,1], 
		aspect=DataAspect(),
		yticks = (
			(1:length(tries_params)).*scale,
			["M=$(t[1]),λ₂₁=$(t[2])" for t in tries_params]
		),
		yticklabelrotation=π/4, yticklabelalign=(:right, :bottom),
		bottomspinevisible=false, leftspinevisible=false,
		xticksvisible=false, xticklabelsvisible=false, yticksvisible=false,
	)

	for i in 1:length(tries_sorted)
		evals = load_brainRBM_eval(tries_sorted[i], ignore="<v>")
		evals = [a for a in evals if all(isfinite.(values(a)))]
		if i==1
			ax_fontsize = 8
		else
			ax_fontsize = 0
		end
		norms = nRMSEs_L4(evals)
		inds = sortperm(norms, rev=true)
		multipolarnrmseplotter!(ax_multi_tries_S, 
			evals[inds], 
			norms[inds], 
			cmap_max=cmap_max,
			origin=[0,i].*scale, 
			ax_fontsize=ax_fontsize,
			# cmap=reverse(CONV.CMAP_GOODNESS),
		)
	end
	
	# fig
end

# ╔═╡ d0f4b5ce-2910-41d5-b053-e4695d2849a9
Colorbar(
	g_abCB_S_crossval[1,1], 
	colormap=CONV.CMAP_GOODNESS, 
	colorrange=(0, cmap_max),
	label="L4 norm of statistics' nRMSE",
	height=Relative(0.7),
)

# ╔═╡ 5a18f979-3698-4101-a172-17762542dc48
begin
	# fig_test = Figure(size=dfsize().*2)
	ax_20λM_S_crossval = Axis(
		g_a_S_crossval[1,1],
		xlabel="λ₂₁", xticks=((1:length(λ21s)).*scale, string.(λ21s)),
		ylabel="M", yticks=((1:length(Ms)).*scale, string.(Ms)),
		aspect=DataAspect(),
	)
	
	for (i,λ) in enumerate(λ21s)
		for (j,m) in enumerate(Ms)
			# evals = vcat([crossval_eval_loader(v, m, λ) for v in Vs]...)
			evals = crossval_eval_loader(20., m, λ)
			evals = [a for a in evals if all(isfinite.(values(a)))]
			if (i==length(λ21s)) & (j==length(Ms))
				ax_fontsize = 8
			else
				ax_fontsize = 0
			end
			norms = nRMSEs_L4(evals)
			inds = sortperm(norms, rev=true)
			multipolarnrmseplotter!(ax_20λM_S_crossval, 
				evals[inds], 
				norms[inds], 
				cmap_max=cmap_max,
				origin=[i,j].*scale, 
				ax_fontsize=ax_fontsize,
				# cmap=reverse(CONV.CMAP_GOODNESS),
			)
		end
	end
	# fig_test
end

# ╔═╡ b5513ce1-223c-45d9-ba62-62a26ed61d24
# ╠═╡ disabled = true
#=╠═╡
begin
	CV = Array{Float64}(undef, length(Vs), length(Ms), length(λ21s))
	for (i,v) in enumerate(Vs)
		for (j,m) in enumerate(Ms)
			for (k,λ) in enumerate(λ21s)
				CV[i,j,k] = minimum(nRMSEs_Lp(crossval_eval_loader(v,m, λ), 10))
			end
		end
	end
end
  ╠═╡ =#

# ╔═╡ a268a0d4-75ee-412b-9ed3-ba0c213a700a
#=╠═╡
heatmap(CV[5,:,:], colormap=CONV.CMAP_GOODNESS, colorrange=(0,1))
  ╠═╡ =#

# ╔═╡ 987d6e36-ac57-45b4-a841-65484e5a01a9


# ╔═╡ 1f7c6270-bfd8-41cb-82fe-a44813e57a27
md"""
## Free energy
"""

# ╔═╡ e4a28151-66e6-4899-a3ee-2083f4e1756b
begin
	function rand_vv(v::AbstractMatrix)
		vout = copy(v)
		for i in 1:size(v,1)
			vout[i,:] .= shuffle(vout[i,:])
		end
		@assert sum(v) ≈ sum(vout)
		return vout
	end
	
	function rand_v(v::AbstractMatrix)
		vout = rand_vv(v)
		for t in 1:size(v,2)
			vout[:,t] .= shuffle(vout[:,t])
		end
		@assert sum(v) ≈ sum(vout)
		return vout
	end
end

# ╔═╡ Cell order:
# ╠═1e0f31c2-4a8e-11f0-1d64-6b53ace7d4fe
# ╠═a874ccf6-77aa-4d6c-b483-b4860943f131
# ╠═fd2bf186-4f5c-48c7-bd05-03fcb25c068d
# ╠═b3c0d22b-f925-4d42-9a1f-893ecee98099
# ╠═7908ce1e-df3f-4b20-808a-382b7c7171e0
# ╠═ec4c0f9f-20db-43a5-8647-7a2c0b259de0
# ╠═ffe1ad1a-04c0-4f4a-bcc2-306bd8d6ffa6
# ╠═d0624ba1-c1aa-4b9d-9ae1-d1d3e079376d
# ╟─c44354cd-28aa-4df8-91b7-f7eadfa24e39
# ╠═bc79c953-18cc-43d7-9dd9-6a6a5a32dcdf
# ╟─06dd898e-3c97-456b-93b0-9a2e606db2a2
# ╠═594d27b0-b8a0-4aef-aa78-b0642c4f2f7d
# ╠═05d603b3-f8e0-4c33-94ff-91982bf81dd5
# ╟─bb4c6151-d198-473d-b60d-9bae7eedd9bc
# ╟─0f98d940-62b2-49bf-844d-821acb5fc90d
# ╟─f09a670f-947b-4004-b4bf-f451e87626a9
# ╟─313f9d91-d355-4b4e-8e82-9aee2e5d2049
# ╠═a94c89ca-e407-4448-b2ca-934ad08a8583
# ╠═409c48fa-f424-4ab5-8af2-bebf79d4cee5
# ╠═9e235d4e-976d-4858-ad5b-e69d2adb9613
# ╟─5a48755f-cd04-4ca7-b141-5a3b7e7956c0
# ╠═1fd4a66d-b8e7-48e0-8b4b-cc6227fb4635
# ╠═b64106e0-0000-4732-9c55-7dab86eb74d0
# ╟─b9dfbce8-699c-4b8e-90d2-48932af4e361
# ╠═a7038417-19fa-48e6-8cec-245c38b8549b
# ╠═40837778-55e1-4a02-9033-636829cc5209
# ╟─29931e52-3d41-4f9f-a6b5-5baac64ec1b4
# ╠═85225d68-9e87-40fd-990f-48092dafdc12
# ╠═d5ac9557-fd7a-4dfe-acd1-837f43ed90e3
# ╟─deea6b78-c7e3-4f9f-a286-6f7cea479299
# ╠═4f8efba0-002a-4c82-a010-046b69f628b1
# ╟─3360c555-14e8-4c20-8dca-05ecd47914b6
# ╠═11f81da5-296f-4a58-a647-56d66c8e4ae3
# ╠═dbcef2ba-acfd-466c-af3d-c1905049d72e
# ╠═1e166691-f70d-4f1b-a696-37758783ea56
# ╟─3c2ecda1-36de-4d38-918c-c0fe0e6064b7
# ╠═f0611a3e-db56-4afd-b086-fc5e22fe9668
# ╠═bafe62c1-4481-43ab-b919-894923c9c9f4
# ╠═bec186b7-4b0d-4366-ae65-ab570afbdf0a
# ╟─9e27ead6-7261-422f-96a8-aaaf0ef858f5
# ╠═5b54efc6-53ce-436f-ac6b-8c2b9e62d832
# ╠═bc94ea3f-8f99-498e-a156-9615e4b272ff
# ╟─874d0c99-3bb2-4899-b7b4-f19c126bebd2
# ╠═5dabbf30-b56e-46bb-b346-fc8ea74bae69
# ╠═e1a2aaf3-89be-4d75-a466-8a411a5b611f
# ╟─bc6a323d-8d86-44d3-b1d8-977aa9fc16d9
# ╠═bf90ee77-0a7c-48d7-b338-62c8edb9fd57
# ╠═e7822746-8055-4e56-80d2-df735b53378d
# ╟─303571cc-9852-49ea-bec8-ce15d1dfe796
# ╠═a577ab41-093b-4939-b1f1-26c843ccb467
# ╠═4d73d700-e37e-4369-81b7-d24b5fec7ede
# ╠═c03b36c3-02af-44cf-bfba-8c063e53d1e6
# ╟─1954c0e1-7a94-44b4-bb80-154dfdbfd080
# ╠═2df609a6-00fe-4ec4-ae0e-77d3d623426b
# ╠═d2bf01a7-d6a4-46ce-8a50-0d41c4b2c2b5
# ╟─561d263b-9a19-4b4d-ae61-8b5a22d99a62
# ╠═ef1d3c49-2191-407b-9d2a-ad8b072b7b92
# ╠═e41c98ed-209a-4d39-92c9-66316ee23c03
# ╠═72d0049b-a0c6-46f0-a58a-8bccf9f63549
# ╟─7a27f5e2-c75e-4133-9dfe-d4ab112aaed2
# ╟─43ee56fb-2440-4152-8573-7bf5de402023
# ╠═d5f67952-1093-48e5-a8d6-2885bcbd1328
# ╠═6911b174-1fa9-4e33-ba05-cdd41423de92
# ╠═0de28d8d-70f2-4d44-a293-029535abd084
# ╠═9c8155e6-578d-41f4-b0a8-a469412b0574
# ╟─586c6205-545b-4d5d-b7ea-5cafcf2d0115
# ╠═e0b02c22-2674-40bd-9f3c-04f64fea6500
# ╠═afa72f3b-1d55-4a59-ae3e-6184eeeb1448
# ╠═3dc4df0a-d7bf-4695-85fc-5e7f7e3a3687
# ╟─43e4f988-2e3d-4f8e-921b-3d122d2c474d
# ╠═3bb9562b-2009-4174-baf3-5dc24422dd24
# ╠═5c5f9c1e-1817-4089-b186-b55a0c0d3870
# ╠═f554e469-3fd9-4ecb-a1b7-4b0a386e403b
# ╠═65640112-8b2d-479d-b36a-aff63ea258c5
# ╠═0e9229c5-ddda-4aa3-aa26-2f5a09ce2276
# ╟─e932f290-1da7-4b6c-b10d-6ad7259ff2aa
# ╟─77aaaf76-9992-4b74-a638-3ff9e35c7687
# ╠═32b884b6-9d77-4010-8908-a65618231567
# ╠═499428c3-8f0f-4ec4-9df5-011d352bfd3a
# ╟─f11d7726-1bbd-4c15-8664-2ec2c1ac10f2
# ╠═3749dd59-5ac9-4fe1-b8a8-e5af92c6c7af
# ╠═d29c0358-f5c0-421b-9814-35fa0b170ec5
# ╠═fcff0be2-e1e8-4336-8f69-02f8794d105f
# ╟─9259472c-850d-4cc5-91e3-b7c4ffe8ad99
# ╠═4ef6daa2-824f-4904-b6e6-2b1ca586cd13
# ╠═ac6bf797-ffc5-4036-a9f2-4b16c5f4fd44
# ╠═0daee0c6-0030-43bb-be4e-414f18553d83
# ╠═a444dc6b-2201-4d3e-80ec-1257e24eb002
# ╠═696bf3a1-5645-46bc-8f0f-e50b8f47c4ae
# ╠═346ebf26-0a08-438a-b018-62225ee87312
# ╠═64a75b7a-a9c7-4a7a-bf12-ec40de222280
# ╟─dd655114-a3b1-4ad6-abb2-8076e790cd99
# ╟─20299104-25c9-4fb7-9bc0-bfe393a6ae00
# ╠═ff6c386a-d149-42c5-9df9-10f3179aa248
# ╠═f561975b-2ff2-4ab1-be72-f8418066e468
# ╠═8cbbe4e5-3745-4d42-9534-dcf6bdb28f33
# ╠═6cc3a353-acef-4564-8cab-2c7589cfea1a
# ╠═400cf119-b1c7-4b02-a3db-8eb437f53811
# ╠═26887565-2982-4b01-9bc2-5b5d9412027d
# ╠═60aabdd5-4be7-4fd9-8878-03614a6fcffd
# ╠═50bae469-a625-401a-ae5d-e4708baf89f7
# ╠═c4725180-a139-4da6-9da2-e18025b4519a
# ╟─02f11ee2-bb9e-4c6a-8320-1b8c0ca81bdf
# ╠═8926908c-7541-402b-bf01-14329a8e9b85
# ╟─3ed88439-0007-4472-a946-db2586fdc60c
# ╠═406b65ee-8f1f-4dcc-8068-fc66773ef81d
# ╟─7bb8d15f-3673-46de-871f-2bf4fec13d40
# ╠═f09ce00c-e17f-4ae9-aea5-7de07f2417cf
# ╟─aec0dcf8-4b43-4727-8a96-a89bed1696b9
# ╠═de31e32e-9402-4987-aa07-9d6f93b33470
# ╠═011ddb46-1c3f-4b70-8b80-9a1a4d9742e8
# ╟─abe3f23c-31ee-4f31-969c-8d5f7eb5153e
# ╠═723ebadb-627e-4f2f-84e5-0dbb9bc115a2
# ╠═d8514df1-f200-411e-b4dc-259395debb5f
# ╠═69c8a3b5-0aeb-4c31-bb9a-a7b6cef38c08
# ╠═961f299d-6c41-4dcd-b7af-07f919525b28
# ╠═d03b97ea-e509-410b-8ea5-1ab0a312ef84
# ╟─0c076ecf-60f0-4660-930f-f58672ac791f
# ╠═92ee9fbe-4418-43fd-946f-072efc7fd982
# ╠═6b9d78ef-7140-4806-a622-1795c1139550
# ╟─67056c02-f892-4d1c-8caf-3f4b5f95ed58
# ╠═9827331c-48f0-4ff3-9b21-0b963644a064
# ╠═b3658702-1de7-41c2-9a50-01bde846ac21
# ╠═382183a0-d066-4b24-b7c5-dbf5bd09dd2b
# ╠═9680752c-17f9-491a-9459-034c0a43c573
# ╠═4bfb9eb4-60d2-49fa-af17-b8defb6c51b9
# ╟─d015c130-47c9-4ae6-a142-ba390906ef0d
# ╠═1565bcf6-7f53-41c8-9add-19b44fc40a20
# ╠═68319e97-3304-4ff1-a60d-e0a07c4d2d28
# ╠═5a18f979-3698-4101-a172-17762542dc48
# ╠═538f7212-d9ba-4b63-a3b5-956d143f3f82
# ╠═ea793074-e896-4feb-80a7-2f52ac2c74de
# ╠═d0f4b5ce-2910-41d5-b053-e4695d2849a9
# ╠═1e46fa45-1d1b-4bda-a290-bfe38b56ff36
# ╠═5cb568a6-8124-4685-96a6-661935349790
# ╠═e3a97867-38f2-42d9-8582-5a77c2521737
# ╠═2bc1fe15-8456-4684-b9d4-eb048a337239
# ╠═3113beef-1a37-451d-8c85-f8e0f30fc78b
# ╠═46c1cd03-c30c-46aa-b716-165a757dc09c
# ╠═5088865e-a2cc-43ec-bbaf-c9fdaaf80f84
# ╠═84256c48-9afe-4fb6-8c32-10c13b4a99cc
# ╠═1a25cdb0-398a-43bf-94fc-05e1bf292a25
# ╠═82441764-acd6-45f4-9606-4a0b87c8463c
# ╠═83b229c9-8a5f-4153-9dc2-68e84322ad90
# ╠═1e3e54a8-fba6-4252-99ce-2745ed704d5f
# ╠═8a15c80d-187c-4404-80fc-888b4af12669
# ╟─f4f849c1-ea33-469e-b6db-707b4954bbde
# ╠═a0dfe8f8-5f8b-4b23-8e60-6e0af2a73e22
# ╠═e34a4c5a-d073-46f9-aa51-03755355f641
# ╠═923aae15-4077-4cd0-b01e-88c54bb385a4
# ╠═baa32e6a-1794-4816-841c-778860309b00
# ╠═525f73b2-e861-458c-ac9e-cd6494c4460b
# ╠═9cb477dc-8110-4393-8593-19b3a84fb06f
# ╠═84583539-0394-4eaf-8ce0-fafad6a98687
# ╠═4fcc922f-b72e-4fcc-bf3b-d44fa70cae07
# ╠═fedacb74-b3a1-4ee4-b6d0-e21401a4949a
# ╟─cd030063-1435-4560-b6ef-f311ed57209f
# ╠═611b5330-ff6a-4e48-8e7e-ff10287bbf78
# ╟─9bd3fcc4-d2f7-4f4d-ad0f-a8b957aad43b
# ╠═30161e5e-9b2f-494c-8595-43d6e3b9dd02
# ╠═ecf7149d-9a42-47f9-9f87-b41c80e6b0cc
# ╟─e194052d-60d3-4b3b-a61e-7632915f5929
# ╠═1a46c70c-3e25-49f4-bf18-784fe16f55fb
# ╠═d02400bc-48cf-44a0-8879-671951857795
# ╠═b5513ce1-223c-45d9-ba62-62a26ed61d24
# ╠═a268a0d4-75ee-412b-9ed3-ba0c213a700a
# ╠═987d6e36-ac57-45b4-a841-65484e5a01a9
# ╟─1f7c6270-bfd8-41cb-82fe-a44813e57a27
# ╠═ddf1132e-3f96-443b-89b8-36a883a8e088
# ╠═e4a28151-66e6-4899-a3ee-2083f4e1756b
