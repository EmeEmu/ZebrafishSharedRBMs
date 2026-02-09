### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ b63457db-6045-4981-8f70-96a3906c1eb7
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ 82f4a87c-636f-4159-92fd-ad773bd089cb
begin
	using BrainRBMjulia
	using LinearAlgebra: diagind
	using HDF5
	using StatsBase
	
	using CairoMakie
	using BrainRBMjulia: multipolarnrmseplotter!, idplotter!, dfsize, quantile_range, neuron2dscatter!, cmap_aseismic, polarnrmseplotter!, corrplotter!, couplingplotter
	using ColorSchemes: reverse, RdYlGn_9

	CONV = @ingredients("conventions.jl")
	include(joinpath(dirname(Base.current_project()), "Misc_Code", "fig_saving.jl"))
end

# ╔═╡ 7fe82f86-00e9-11f1-25ad-27e2873c9cb9
md"""
# Imports + Notebook Preparation
"""

# ╔═╡ b9a18b05-c269-4dfe-95d3-5efa170b243e
TableOfContents()

# ╔═╡ fc1e28f7-d098-4301-a6c2-bce96dc373d3
set_theme!(CONV.style_publication)

# ╔═╡ d5f7a808-3bc0-4359-9422-552abab1eff5
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ f868d890-6cb2-4f96-a0ad-585ba337c5b1


# ╔═╡ afee1568-5634-4170-a2db-f588773be580
md"""
# 0. Data
"""

# ╔═╡ 4c26222e-e24c-4e1d-a797-defa9b10f8ae
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ 58cea276-b76c-4b7d-bca3-f123d0423c99
begin
	teacher = FISH[3]
	student = FISH[1]
end;

# ╔═╡ 6599286f-5433-4608-933b-26d58fe1a819
begin
	Ms = [50, 80, 100, 120, 150, 200]
	λs = [0.005, 0.01, 0.02, 0.04]#, 0.08]
end;

# ╔═╡ a4e5e40b-db9b-4bd6-bbdb-d18571ee4a95


# ╔═╡ a2d12ba1-caab-49f3-a3d7-471e9d583b2d


# ╔═╡ fb765b9b-6fc5-44d6-a0df-fb6d9008fa44
md"""
# 1. Fig
"""

# ╔═╡ f910028e-93c7-4605-a38b-5fd4e38d98e5
begin
	fig_main = Figure(size=(53, 49).*(4,4).*(4/3/0.35)) #[ud, lr]

	g_ac = fig_main[1,1] = GridLayout()
	g_a = g_ac[1,1] = GridLayout()
	g_c = g_ac[1,2] = GridLayout()
	g_acCB = g_ac[1,3] = GridLayout()

	g_bd = fig_main[2,1] = GridLayout()
	g_b = g_bd[1,1] = GridLayout()
	g_d = g_bd[1,2] = GridLayout()

	for (label, layout) in zip(
		["A", "B", "C", "D"],
		[g_a, g_b, g_c, g_d],
	)
	    Label(layout[1, 1, TopLeft()], label,
	        fontsize = Makie.current_default_theme().Axis.titlesize.val,
	        font = :bold,
	        padding = (0, 5, 5, 0),
	        halign = :right)
		# Box(layout[1,1])
	end
end

# ╔═╡ 24cb7acf-3113-4f2f-9ee9-d28d4e0b68e5


# ╔═╡ 7a2615c6-3b3f-47d1-b67d-751296e3db71
md"""
## 1.A. Teacher Stats
"""

# ╔═╡ 778db0c6-ec10-41db-8a56-c3431b54f8cf
begin
	EVALS_M = Array{Vector{Dict{Any,Any}}}(undef, length(Ms))
	NORMS_M = Array{Vector{Float64}}(undef, length(Ms))
	for (i,M) in enumerate(Ms)
		paths = LOAD.load_wbscRBMs(
			"Repeats", 
			"bRBM_$(teacher)*_WBSC_M$(M)_l1$(0.02)_l2l10_*"
		)
		EVALS_M[i] = load_brainRBM_eval(paths, ignore="1-nLLH")
		NORMS_M[i] = nRMSEs_L4(EVALS_M[i])
	end
	
	EVALS_λ = Array{Vector{Dict{Any,Any}}}(undef, length(λs))
	NORMS_λ = Array{Vector{Float64}}(undef, length(λs))
	for (j, λ) in enumerate(λs)
		paths = LOAD.load_wbscRBMs(
			"Repeats", 
			"bRBM_$(teacher)*_WBSC_M$(100)_l1$(λ)_l2l10_*"
		)
		EVALS_λ[j] = load_brainRBM_eval(paths, ignore="1-nLLH")
		NORMS_λ[j] = nRMSEs_L4(EVALS_λ[j])
	end
end

# ╔═╡ 85032f09-e206-4ab6-af45-48849b869c9a
begin
	scale = 3
	fig_teachstats = g_a#Figure(size=dfsize().*(2.5,2))
	ax_fit_1 = Axis(
	    fig_teachstats[1,1],
	    xticks=((1:length(Ms)).*scale, string.(Ms)),
	    yticks=((1:length(λs)).*scale, string.(λs)),
	    aspect=DataAspect(),
		xticklabelrotation = π/4,
		xlabel = "Number of Hidden Units , M", ylabel="Regularization , λ₁",
	)
	cmap_max = nRMSEs_L4(EVALS_M[1,1], max=true)

	j = findfirst(λs .== 0.02)
	for (i,M) in enumerate(Ms)
		if M==100 
			ax_color=:dodgerblue2
		else 
			ax_color=:grey
		end
		multipolarnrmseplotter!(ax_fit_1, 
			EVALS_M[i], 
			NORMS_M[i], 
			cmap_max=cmap_max,
			origin=[i,j].*scale, 
			ax_fontsize=0,
			linewidth=2,markersize=8;
			ax_color,
		)
	end
	
	i = findfirst(Ms .== 100)
	for (j,λ) in enumerate(λs)
		if λ==0.02
			ax_color=:dodgerblue2
		else 
			ax_color=:grey
		end
		multipolarnrmseplotter!(ax_fit_1, 
			EVALS_λ[j], 
			NORMS_λ[j], 
			cmap_max=cmap_max,
			origin=[i,j].*scale, 
			ax_fontsize=0,
			linewidth=2,markersize=8;
			ax_color,
		)
	end

	Colorbar(
		g_acCB[1,1], 
		colormap=CONV.CMAP_GOODNESS, colorrange=(0,cmap_max),
		label="L4 norm of statistics' nRMSE",
		height=Relative(0.6),
	)
	fig_teachstats
end

# ╔═╡ 5a072ddd-44c1-4d04-a979-6258ea7894b9
md"""
## 1.B. Teacher orphaned
"""

# ╔═╡ f4c5a4b5-72f4-4ae3-b377-00273e1c937c
begin
	ORPHS_M = Array{Int}(undef, length(Ms), 5)
	for (i,M) in enumerate(Ms)
		paths = LOAD.load_wbscRBMs(
			"Repeats", 
			"bRBM_$(teacher)*_WBSC_M$(M)_l1$(0.02)_l2l10_*"
		)
		for (k, path) in enumerate(paths)
			rbm, _,_,_,_,_ = load_brainRBM(path)
			ORPHS_M[i,k] = sum(sum(rbm.w .> 1.e-5, dims=1)[1,:] .== 0)
		end
	end
	
	ORPHS_λ = Array{Int}(undef, length(λs), 5)
	for (j,λ) in enumerate(λs)
		paths = LOAD.load_wbscRBMs(
			"Repeats", 
			"bRBM_$(teacher)*_WBSC_M$(100)_l1$(λ)_l2l10_*"
		)
		for (k, path) in enumerate(paths)
			rbm, _,_,_,_,_ = load_brainRBM(path)
			ORPHS_λ[j,k] = sum(sum(rbm.w .> 1.e-5, dims=1)[1,:] .== 0)
		end
	end
end

# ╔═╡ 3c44dbe1-4053-4fae-a670-809a0b55b60f
begin
	fig_teachorph = g_b#Figure(size=dfsize().*(2,2))
	ax_teachorph_M = Axis(
		fig_teachorph[1,1],
		ytickformat="{:d}%",
		xticks=Ms, yticks=[0, 25, 50],
		xlabel = "Number of Hidden Units , M", 
		# ylabel="∣wᵢ_μ > 10⁻⁵ ∀i∣_μ / M",
		ylabel=L"\frac{1}{M} \mid w_{i\mu} < 10^{-5} \ , \ \forall i \mid_\mu"
	)
	ax_teachorph_λ = Axis(
		fig_teachorph[2,1],
		ytickformat="{:d}%",
		xticks=λs, yticks=[0, 25, 50], yticklabelsvisible=true,
		xlabel = "Regularization , λ₁", 
		# ylabel="∣wᵢ_μ > 10⁻⁵ ∀i∣_μ / M",
		ylabel=L"\frac{1}{M} \mid w_{i\mu} < 10^{-5} \ , \ \forall i \mid_\mu"
	)

	ms_M = mean(ORPHS_M, dims=2)[:,1] ./ Ms .* 100
	stds_M = std(ORPHS_M, dims=2)[:,1] ./ Ms .* 100
	lines!(
		ax_teachorph_M, 
		Ms, ms_M,
		color=:black,
	)
	scatter!(
		ax_teachorph_M, 
		Ms, ms_M,
		color=[if M==100 Symbol("dodgerblue2") else Symbol("black") end for M∈Ms]
	)
	errorbars!(
		ax_teachorph_M, 
		Ms, ms_M,
		stds_M,
		whiskerwidth = 5,
		color=[if M==100 Symbol("dodgerblue2") else Symbol("black") end for M∈Ms]
	)
	text!(
		ax_teachorph_M,
		(Ms[end]+Ms[end-1])/2, 
		(ms_M[end]-stds_M[end]+ms_M[end-1]-stds_M[end-1])/2,
		text="λ₁=$(0.02)",
		align=(:center, :top), 
		rotation=0.12pi,
	)

	ms_λ = mean(ORPHS_λ, dims=2)[:,1] ./ 100 .* 100
	stds_λ = std(ORPHS_λ, dims=2)[:,1] ./ 100 .* 100
	lines!(
		ax_teachorph_λ, 
		λs, ms_λ,
		color=:black,
	)
	scatter!(
		ax_teachorph_λ, 
		λs, ms_λ,
		color=[if λ==0.02 Symbol("dodgerblue2") else Symbol("black") end for λ∈λs]
	)
	errorbars!(
		ax_teachorph_λ, 
		λs, ms_λ,
		stds_λ,
		whiskerwidth = 5,
		color=[if λ==0.02 Symbol("dodgerblue2") else Symbol("black") end for λ∈λs]
	)
	text!(
		ax_teachorph_λ,
		(λs[end]+λs[end-1])/2, 
		(ms_λ[end]-stds_λ[end]+ms_λ[end-1]-stds_λ[end-1])/2,
		text="M=$(100)",
		align=(:center, :top), 
		rotation=0.08pi,
	)

	linkyaxes!(ax_teachorph_M, ax_teachorph_λ)
	
	fig_teachorph
end

# ╔═╡ 8a52d83c-837e-4d48-9def-1380238c75c2
md"""
## 1.C. Student Stats
"""

# ╔═╡ 0fb2efcf-67af-4982-8a83-3edbaa588ad4
begin
	sEVALS_M = Array{Dict{Any,Any}}(undef, length(Ms))
	sNORMS_M = Array{Float64}(undef, length(Ms))
	for (i,M) in enumerate(Ms)
		path = LOAD.load_wbscRBM(
			"biRBMs", 
			"biRBM_$(student)_FROM_$(teacher)_WBSC_M$(M)_l1$(0.02)_l2l10*"
		)
		sEVALS_M[i] = load_brainRBM_eval(path, ignore="1-nLLH")
		sNORMS_M[i] = nRMSEs_L4(sEVALS_M[i])
	end
	
	sEVALS_λ = Array{Dict{Any,Any}}(undef, length(λs))
	sNORMS_λ = Array{Float64}(undef, length(λs))
	for (j, λ) in enumerate(λs)
		path = LOAD.load_wbscRBM(
			"biRBMs", 
			"biRBM_$(student)_FROM_$(teacher)_WBSC_M$(100)_l1$(λ)_l2l10*"
		)
		sEVALS_λ[j] = load_brainRBM_eval(path, ignore="1-nLLH")
		sNORMS_λ[j] = nRMSEs_L4(sEVALS_λ[j])
	end
end

# ╔═╡ dc0de2ab-b50e-4bfd-a1c2-62bf284d8c25
begin
	fig_studstats = g_c#Figure(size=dfsize().*(2,2))
	ax_fit_2 = Axis(
	    fig_studstats[1,1],
	    xticks=((1:length(Ms)).*scale, string.(Ms)),
	    yticks=((1:length(λs)).*scale, string.(λs)),
	    aspect=DataAspect(),
		xticklabelrotation = π/4,
		xlabel = "Number of Hidden Units , M", ylabel="Regularization , λ₁",
	)

	J = findfirst(λs .== 0.02)
	for (i,M) in enumerate(Ms)
		if M==100 
			ax_color=:dodgerblue2
		else 
			ax_color=:grey
		end
		multipolarnrmseplotter!(ax_fit_2, 
			[sEVALS_M[i]], 
			[sNORMS_M[i]], 
			cmap_max=cmap_max,
			origin=[i,J].*scale, 
			ax_fontsize=0,
			linewidth=2,markersize=8;
			ax_color,
		)
	end
	
	I = findfirst(Ms .== 100)
	for (j,λ) in enumerate(λs)
		if λ==0.02
			ax_color=:dodgerblue2
		else 
			ax_color=:grey
		end
		multipolarnrmseplotter!(ax_fit_2, 
			[sEVALS_λ[j]], 
			[sNORMS_λ[j]], 
			cmap_max=cmap_max,
			origin=[I,j].*scale, 
			ax_fontsize=0,
			linewidth=2,markersize=8;
			ax_color,
		)
	end

	# Colorbar(
	# 	fig_studstats[1,2], 
	# 	colormap=CONV.CMAP_GOODNESS, colorrange=(0,cmap_max),
	# 	label="L4 norm of statistics' nRMSE",
	# 	height=Relative(0.6),
	# )
	fig_studstats
end

# ╔═╡ 204c881f-49c9-4f72-9115-9947cd47346f
md"""
## 1.D. Free Energy
"""

# ╔═╡ 2df4cd4e-9247-4344-bb92-d0dd717d23b7
function free_en(M::Int, λ::Float64)
	teachpath = LOAD.load_wbscRBM(
		"bRBMs", 
		"bRBM_$(teacher)*_WBSC_M$(M)*_l1$(λ)_*"
	)
	studpath = LOAD.load_wbscRBM(
		"biRBMs", 
		"biRBM_$(student)_FROM_$(teacher)_WBSC_M$(M)_l1$(λ)*"
	)
	Trbm,_,_,_,_,_ = load_brainRBM(teachpath)
	Srbm,_,_,_,_,_ = load_brainRBM(studpath)

	Tspikes = load_data(LOAD.load_dataWBSC(teacher)).spikes
	Sspikes = load_data(LOAD.load_dataWBSC(student)).spikes

	T_T = free_energy(Trbm, mean_v_from_h(Trbm, mean_h_from_v(Trbm, Tspikes)))
	T_S = free_energy(Srbm, mean_v_from_h(Srbm, mean_h_from_v(Trbm, Tspikes)))
	S_S = free_energy(Srbm, mean_v_from_h(Srbm, mean_h_from_v(Srbm, Sspikes)))
	S_T = free_energy(Trbm, mean_v_from_h(Trbm, mean_h_from_v(Srbm, Sspikes)))
	
	return (
		T_T,# .- median(T_T), 
		T_S,# .- median(T_S), 
		S_S,# .- median(S_S), 
		S_T,# .- median(S_T)
	)
end

# ╔═╡ 475b9e50-78d6-4b82-a79c-50fdc7054ee9
begin
	Ms_T_T, Ms_T_S, Ms_S_S, Ms_S_T = [], [], [], []
	for (i,M) in enumerate(Ms)
		T_T, T_S, S_S, S_T = free_en(M, 0.02)
		push!(Ms_T_T, T_T)
		push!(Ms_T_S, T_S)
		push!(Ms_S_S, S_S)
		push!(Ms_S_T, S_T)
	end
end

# ╔═╡ 9ca8826c-b803-4af8-ae93-ae44e24c70ce
begin
	λs_T_T, λs_T_S, λs_S_S, λs_S_T = [], [], [], []
	for (i,λ) in enumerate(λs)
		T_T, T_S, S_S, S_T = free_en(100, λ)
		push!(λs_T_T, T_T)
		push!(λs_T_S, T_S)
		push!(λs_S_S, S_S)
		push!(λs_S_T, S_T)
	end
end

# ╔═╡ 8fc05b19-043c-44c8-99cf-a8adc4d21303
begin
	Ms_Y = vcat([
		vcat(y...) 
		for y in [Ms_T_T, Ms_T_S, Ms_S_S, Ms_S_T]
	]...)
	
	Ms_X = vcat([
		vcat([
			fill(i, size(y[i])) 
			for (i,M)∈enumerate(Ms)
		]...) 
		for y in [Ms_T_T, Ms_T_S, Ms_S_S, Ms_S_T]
	]...)
	
	Ms_side = vcat([
		fill(side,size(vcat(y...)))
		for (side, y) in zip(
			[:left, :right, :left, :right], 
			[Ms_T_T, Ms_T_S, Ms_S_S, Ms_S_T]
		)
	]...)
	
	Ms_dodge = vcat([
		fill(dodge,size(vcat(y...)))
		for (dodge, y) in zip(
			[1,2,2,1], 
			[Ms_T_T, Ms_T_S, Ms_S_S, Ms_S_T]
		)
	]...)
	
	Ms_color = vcat([
		fill(c,size(vcat(y...)))
		for (c, y) in zip(
			[CONV.COLOR_TEACHER, CONV.COLOR_STUDENT, CONV.COLOR_STUDENT, CONV.COLOR_TEACHER, ], 
			[Ms_T_T, Ms_T_S, Ms_S_S, Ms_S_T]
		)
	]...)
end

# ╔═╡ 03502a39-1689-4f67-9cc4-83f508bf0a30
begin
	λs_Y = vcat([
		vcat(y...) 
		for y in [λs_T_T, λs_T_S, λs_S_S, λs_S_T]
	]...)
	
	λs_X = vcat([
		vcat([
			fill(j, size(y[j])) 
			for (j,λ)∈enumerate(λs)
		]...) 
		for y in [λs_T_T, λs_T_S, λs_S_S, λs_S_T]
	]...)
	
	λs_side = vcat([
		fill(side,size(vcat(y...)))
		for (side, y) in zip(
			[:left, :right, :left, :right], 
			[λs_T_T, λs_T_S, λs_S_S, λs_S_T]
		)
	]...)
	
	λs_dodge = vcat([
		fill(dodge,size(vcat(y...)))
		for (dodge, y) in zip(
			[1,2,2,1], 
			[λs_T_T, λs_T_S, λs_S_S, λs_S_T]
		)
	]...)
	
	λs_color = vcat([
		fill(c,size(vcat(y...)))
		for (c, y) in zip(
			[CONV.COLOR_TEACHER, CONV.COLOR_STUDENT, CONV.COLOR_STUDENT, CONV.COLOR_TEACHER, ], 
			[λs_T_T, λs_T_S, λs_S_S, λs_S_T]
		)
	]...)
end

# ╔═╡ e87a0a3e-7113-4eab-8734-d96afc6a9674
λs_side

# ╔═╡ 41ada424-a3a0-4257-8fe0-9b4dc22829e3
begin
	fig_FE = g_d#Figure(size=dfsize().*(2,2))
	
	ax_FE_MS = Axis(
		fig_FE[1,1],
		xticks=(1:length(Ms), string.(Ms)), xlabel="Number of Hidden Units , M", 
		ytickformat = ys -> ["$(round(Int, y/1.e3))" for y in ys],
		ylabel="Free Energy , F(v)",
	)
	Label(fig_FE[1,1,Top()], halign=:left, "×10³")
	violin!(
		ax_FE_MS, 
		Ms_X, Ms_Y, 
		side=Ms_side, dodge=Ms_dodge, color=Ms_color,
		dodge_gap=0., gap=0.4, scale=:area, width=1.2, show_median=true,
		strokewidth=1,
	)
	ylims!(ax_FE_MS, 0, +5.5e3)

	
	ax_FE_λS = Axis(
		fig_FE[2,1],
		xticks=(1:length(λs), string.(λs)), xlabel="Regularization , λ₁",
		ytickformat = ys -> ["$(round(Int, y/1.e3))" for y in ys],
		ylabel="Free Energy , F(v)",
	)
	Label(fig_FE[2,1,Top()], halign=:left, "×10³")
	violin!(
		ax_FE_λS, 
		λs_X, λs_Y, 
		side=λs_side, 
		dodge=λs_dodge, color=λs_color,
		dodge_gap=0., gap=0.4, scale=:area, width=1.2, show_median=true,
		strokewidth=1,
	)
	ylims!(ax_FE_λS, 0, +5.5e3)
	
	fig_FE
end

# ╔═╡ eeece34a-eaa2-49fc-9f27-a7ce42bdaa94
md"""
## 1.END Adjustments
"""

# ╔═╡ 0cc09cbc-03e0-4211-8e0d-958eec8a7292
all_axes = [ax for ax in fig_main.content if typeof(ax)==Axis];

# ╔═╡ 9b5ebddf-4776-4a45-9933-2e8fb9c53976
for ax in all_axes
	ax.alignmode = Mixed(left=0)
end

# ╔═╡ e3fd2cf9-95fd-41b3-8c42-2956e0016a9f
fig_main

# ╔═╡ 22f76d27-05ab-4f87-b77c-d2603038ede5
save(@figpath("Supp_hyperparams"), fig_main)

# ╔═╡ Cell order:
# ╟─7fe82f86-00e9-11f1-25ad-27e2873c9cb9
# ╠═b63457db-6045-4981-8f70-96a3906c1eb7
# ╠═82f4a87c-636f-4159-92fd-ad773bd089cb
# ╠═b9a18b05-c269-4dfe-95d3-5efa170b243e
# ╠═fc1e28f7-d098-4301-a6c2-bce96dc373d3
# ╠═d5f7a808-3bc0-4359-9422-552abab1eff5
# ╠═f868d890-6cb2-4f96-a0ad-585ba337c5b1
# ╟─afee1568-5634-4170-a2db-f588773be580
# ╠═4c26222e-e24c-4e1d-a797-defa9b10f8ae
# ╠═58cea276-b76c-4b7d-bca3-f123d0423c99
# ╠═6599286f-5433-4608-933b-26d58fe1a819
# ╠═a4e5e40b-db9b-4bd6-bbdb-d18571ee4a95
# ╠═a2d12ba1-caab-49f3-a3d7-471e9d583b2d
# ╟─fb765b9b-6fc5-44d6-a0df-fb6d9008fa44
# ╠═f910028e-93c7-4605-a38b-5fd4e38d98e5
# ╠═24cb7acf-3113-4f2f-9ee9-d28d4e0b68e5
# ╟─7a2615c6-3b3f-47d1-b67d-751296e3db71
# ╠═778db0c6-ec10-41db-8a56-c3431b54f8cf
# ╠═85032f09-e206-4ab6-af45-48849b869c9a
# ╟─5a072ddd-44c1-4d04-a979-6258ea7894b9
# ╠═f4c5a4b5-72f4-4ae3-b377-00273e1c937c
# ╠═3c44dbe1-4053-4fae-a670-809a0b55b60f
# ╟─8a52d83c-837e-4d48-9def-1380238c75c2
# ╠═0fb2efcf-67af-4982-8a83-3edbaa588ad4
# ╠═dc0de2ab-b50e-4bfd-a1c2-62bf284d8c25
# ╟─204c881f-49c9-4f72-9115-9947cd47346f
# ╠═2df4cd4e-9247-4344-bb92-d0dd717d23b7
# ╠═475b9e50-78d6-4b82-a79c-50fdc7054ee9
# ╠═9ca8826c-b803-4af8-ae93-ae44e24c70ce
# ╠═8fc05b19-043c-44c8-99cf-a8adc4d21303
# ╠═03502a39-1689-4f67-9cc4-83f508bf0a30
# ╠═e87a0a3e-7113-4eab-8734-d96afc6a9674
# ╠═41ada424-a3a0-4257-8fe0-9b4dc22829e3
# ╟─eeece34a-eaa2-49fc-9f27-a7ce42bdaa94
# ╠═0cc09cbc-03e0-4211-8e0d-958eec8a7292
# ╠═9b5ebddf-4776-4a45-9933-2e8fb9c53976
# ╠═e3fd2cf9-95fd-41b3-8c42-2956e0016a9f
# ╠═22f76d27-05ab-4f87-b77c-d2603038ede5
