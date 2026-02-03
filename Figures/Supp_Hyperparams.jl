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
	
	using CairoMakie
	using BrainRBMjulia: multipolarnrmseplotter!, idplotter!, dfsize, quantile_range, neuron2dscatter!, cmap_aseismic, polarnrmseplotter!, corrplotter!, couplingplotter
	using ColorSchemes: reverse, RdYlGn_9

	CONV = @ingredients("conventions.jl")
	include(joinpath(CONV.UTILSPATH, "fig_saving.jl"))
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
	λs = [0.02]
end;

# ╔═╡ a4e5e40b-db9b-4bd6-bbdb-d18571ee4a95


# ╔═╡ a2d12ba1-caab-49f3-a3d7-471e9d583b2d


# ╔═╡ fb765b9b-6fc5-44d6-a0df-fb6d9008fa44
md"""
# 1. Fig
"""

# ╔═╡ f910028e-93c7-4605-a38b-5fd4e38d98e5


# ╔═╡ 7a2615c6-3b3f-47d1-b67d-751296e3db71
md"""
## 1.A. Teacher Stats
"""

# ╔═╡ 778db0c6-ec10-41db-8a56-c3431b54f8cf
begin
	EVALS = Array{Vector{Dict{Any,Any}}}(undef, length(Ms), length(λs))
	NORMS = Array{Vector{Float64}}(undef, length(Ms), length(λs))
	for (i,M) in enumerate(Ms)
		for (j, λ) in enumerate(λs)
			paths = LOAD.load_wbscRBMs(
				"Repeats", 
				"bRBM_$(teacher)*_WBSC_M$(M)_l1$(λ)_l2l10_*"
			)
			EVALS[i,j] = load_brainRBM_eval(paths, ignore="1-nLLH")
			NORMS[i,j] = nRMSEs_L4(EVALS[i,j])
		end
	end
end

# ╔═╡ c37859bf-744b-40e5-b4d4-143ab2bcd58c


# ╔═╡ 85032f09-e206-4ab6-af45-48849b869c9a
begin
	scale = 3
	fig_teachstats = Figure(size=dfsize().*(3,2))
	ax_fit_1 = Axis(
	    fig_teachstats[1,1],
	    xticks=((1:length(Ms)).*scale, string.(Ms)),
	    yticks=((1:length(λs)).*scale, string.(λs)),
	    aspect=DataAspect(),
		xticklabelrotation = π/4,
		xlabel = "Number of Hidden Units , M", ylabel="Regularization , λ₁",
	)
	cmap_max = nRMSEs_L4(EVALS[1,1], max=true)

	for i in 1:length(Ms)
		for j in 1:length(λs)
			ax_fontsize = 0
			multipolarnrmseplotter!(ax_fit_1, 
				EVALS[i,j], 
				NORMS[i,j], 
				cmap_max=cmap_max,
				origin=[i,j].*scale, 
				ax_fontsize=ax_fontsize,
				# cmap=reverse(CONV.CMAP_GOODNESS),
			)
		end
	end

	Colorbar(
		fig_teachstats[1,2], 
		colormap=CONV.CMAP_GOODNESS, colorrange=(0,cmap_max),
		label="L4 norm of statistics' nRMSE",
		height=Relative(0.6),
	)
	fig_teachstats
end

# ╔═╡ 6f24fddf-c5ee-42f1-826e-014b1e6b3da6


# ╔═╡ 5a072ddd-44c1-4d04-a979-6258ea7894b9
md"""
## 1.B. Teacher orphaned
"""

# ╔═╡ f4c5a4b5-72f4-4ae3-b377-00273e1c937c
begin
	ORPHS = Array{Int}(undef, length(Ms), length(λs), 5)
	for (i,M) in enumerate(Ms)
		for (j, λ) in enumerate(λs)
			paths = LOAD.load_wbscRBMs(
				"Repeats", 
				"bRBM_$(teacher)*_WBSC_M$(M)_l1$(λ)_l2l10_*"
			)
			for (k, path) in enumerate(paths)
				rbm, _,_,_,_,_ = load_brainRBM(path)
				ORPHS[i,j, k] = sum(sum(rbm.w .> 1.e-5, dims=1)[1,:] .== 0)
			end
		end
	end
end

# ╔═╡ 3c44dbe1-4053-4fae-a670-809a0b55b60f
begin
	fig_teachorph = Figure(size=dfsize().*(1.5,1))
	ax_teachorph_M = Axis(
		fig_teachorph[1,1],
		ytickformat="{:d}%",
		xticks=Ms, yticks=[0, 25, 50],
		xlabel = "Number of Hidden Units , M", 
		# ylabel="∣wᵢ_μ > 10⁻⁵ ∀i∣_μ / M",
		ylabel=L"\frac{1}{M} \mid w_{i\mu} < 10^{-5} \ , \ \forall i \mid_\mu"
	)

	for (j,λ) in enumerate([0.02])
		ms = mean(ORPHS[:,j,:], dims=2)[:,1] ./ Ms .* 100
		stds = std(ORPHS[:,j,:], dims=2)[:,1] ./ Ms .* 100
		lines!(
			ax_teachorph_M, 
			Ms, ms,
			color=:black,
		)
		errorbars!(
			ax_teachorph_M, 
			Ms, ms,
			stds,
			whiskerwidth = 5, color=:black,
		)
		text!(
			ax_teachorph_M,
			(Ms[end]+Ms[end-1])/2, 
			(ms[end]-stds[end]+ms[end-1]-stds[end-1])/2,
			text="λ₁=$(λ)",
			align=(:center, :top), rotation=0.1pi,
		)
	end
	
	fig_teachorph
end

# ╔═╡ 14eb81d1-587e-4c87-b3b9-e3e1b20d1334


# ╔═╡ eb816bc3-7db9-4799-a960-920fff87522f


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
# ╟─7a2615c6-3b3f-47d1-b67d-751296e3db71
# ╠═778db0c6-ec10-41db-8a56-c3431b54f8cf
# ╠═c37859bf-744b-40e5-b4d4-143ab2bcd58c
# ╠═85032f09-e206-4ab6-af45-48849b869c9a
# ╠═6f24fddf-c5ee-42f1-826e-014b1e6b3da6
# ╟─5a072ddd-44c1-4d04-a979-6258ea7894b9
# ╠═f4c5a4b5-72f4-4ae3-b377-00273e1c937c
# ╠═3c44dbe1-4053-4fae-a670-809a0b55b60f
# ╠═14eb81d1-587e-4c87-b3b9-e3e1b20d1334
# ╠═eb816bc3-7db9-4799-a960-920fff87522f
