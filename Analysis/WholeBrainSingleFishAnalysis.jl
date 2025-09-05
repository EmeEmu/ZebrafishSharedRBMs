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

# ╔═╡ 9437164a-106d-11f0-1caf-b7d9c6f95f39
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ cf286b16-4948-4730-b096-4f21ef4668c1
begin
	# loading modules
	using BrainRBMjulia
	using CairoMakie

	using BrainRBMjulia: cmap_aseismic, neuron2dscatter!, corrplotter!, dfsize, quantile_range, hu_params_plotter, stats_plotter, generate_energy_plotter, polarnrmseplotter, multipolarnrmseplotter
end

# ╔═╡ 311a23c8-128c-417d-b9df-291426fefb48
TableOfContents()

# ╔═╡ 9e8dcc5d-51ff-4863-a890-b808bea365a7
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ 5124593a-35db-4c8a-b09e-336ffef270bc


# ╔═╡ 6aa48d70-e1aa-475e-a68a-5d034365251a
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ 3eb2db58-5112-4c2b-9a23-dadb02ed0b9b


# ╔═╡ 8f59e4ab-b030-4055-8dae-e21a5142f38b
md"""
# 1. Single fish analysis
"""

# ╔═╡ 2d1dee61-18b7-4647-847d-644b2ea0f9a7
md"selected fish : $(@bind fish Select(FISH))"

# ╔═╡ 7b785c14-8284-4871-9290-4b2ba632b261
# analyse repeated trainings (true) or single training (false)
repeats = true;

# ╔═╡ a1e4746e-5422-457b-a47b-0bb78ffc9a78


# ╔═╡ a65cd79a-f035-4fd0-96bf-bf9308374a52
md"""
## 1.1. Load Dataset
"""

# ╔═╡ 707c9b54-dc3e-4af8-ae6d-50c9b51f8451
begin
	dataset = load_data(LOAD.load_dataWBSC(fish))
	dataset.name
end

# ╔═╡ fa3dfbcd-cd4a-4e22-8a5a-67fff772841f


# ╔═╡ 26619d61-6f82-4252-ad12-cff52eb37e77
md"""
## 1.2. Evaluate training(s)
"""

# ╔═╡ 44699638-977a-4794-9461-85a4cf2f68ac
if repeats
	rbm_paths = LOAD.load_wbscRBMs("Repeats", "bRBM_$(dataset.name)*_rep*")
else
	rbm_paths = LOAD.load_wbscRBM("bRBMs", "bRBM_$(dataset.name)*")
end

# ╔═╡ b657a23f-7f77-4b9a-b831-c53b6e4bd803
evals = load_brainRBM_eval(rbm_paths, ignore="1-nLLH");

# ╔═╡ 7885f940-f1a5-4519-ad55-21edeb3d3fd5


# ╔═╡ 183e0555-c8e0-4242-80af-65c30d0e6b7a
if repeats
	multipolarnrmseplotter(evals)
else
	polarnrmseplotter
end

# ╔═╡ c1cabed3-4e3d-4856-9c4b-20d2a45f78b7


# ╔═╡ e04a81ed-1212-4eb1-bec5-af9350ea9884
md"""
## 1.3. Find best training (if analysing repeated trainings)
"""

# ╔═╡ cdae517c-ed57-4894-be97-ddbdb1dcce52
if repeats
	norms = nRMSEs_Lp(evals, 4)
	best = argmin(norms)
	rbm_path = rbm_paths[best]
	rbm_eval = evals[best]
	rbm_norm = norms[best]
	println("best rbm : \n\t$(rbm_path)\n\teval : $(rbm_eval)\n\tnorm : $(rbm_norm)")
else
	rbm_path = rbm_paths
	rbm_eval = evals
	rbm_norm = nRMSEs_Lp(evals, 4)
	println("single rbm : \n\t$(rbm_path)\n\teval : $(rbm_eval)\n\tnorm : $(rbm_norm)")
end

# ╔═╡ 1856dc8e-5c91-47af-b400-39197011997d
rbm, train_params, evaluation, dsplit, gen, translated = load_brainRBM(rbm_path)

# ╔═╡ 77a1624c-2a2c-4103-bbbd-03ee9e8d7fbf


# ╔═╡ 8eac9910-c1cf-4540-90ee-1799d20c741d
md"""
## 1.4. In depth analysis
"""

# ╔═╡ 36bd93b2-ecae-46d2-8e64-745362413113
# ╠═╡ disabled = true
#=╠═╡
include(joinpath([CONV.MODULESPATH, "BrainRBMjulia/graphs/rbm_graphs.jl"]))
  ╠═╡ =#

# ╔═╡ ecefdf2d-2ac6-4dd0-9a6d-73cb48943558
md"""
### 1.4.1. Generation Energy
"""

# ╔═╡ c1c6fdfb-8e29-41cf-af76-fab81bbe1245


# ╔═╡ fc1f0766-8fb9-4554-bd7b-fb19a333dfc6
generate_energy_plotter(rbm, gen, dsplit)

# ╔═╡ 6357a560-55ad-48f2-af16-bd95e66ae81e
md"""
### 1.4.2. Statistics
"""

# ╔═╡ e9232319-cfbe-4239-aeb6-71f9e9543b82


# ╔═╡ 9c023023-8eae-4860-9bed-fae1352d6150
begin
	moments = compute_all_moments(rbm, dsplit, gen, max_vv=1000)
	nLLH = reconstruction_likelihood(rbm, dsplit.valid, dsplit.train)
	fig = stats_plotter(moments;nrmses=evaluation)
end

# ╔═╡ 20003768-f8d5-443d-89c0-e1a1af3ad64f
md"""
### 1.4.3. Weights
"""

# ╔═╡ 92892579-da39-49a4-a427-96fe5755a71d
begin
	fig_weights = Figure()
	Axis(fig_weights[1, 1], title="Weights", yscale=log10, xlabel=L"w_{i,j}", ylabel="Density")

	hist!(vec(rbm.w), offset=1.e-0, bins=100, color=:black, normalization=:density)
	fig_weights
end

# ╔═╡ 74975a56-fe4e-4e8a-bfbe-845e8a062155
md""" 
### 1.4.4. Hidden Params
"""

# ╔═╡ e4586923-9fdc-4d57-ac98-d093fb15f207
begin
	fig_huparams = Figure()
	hu_params_plotter(fig_huparams[1,1], rbm.hidden)
	fig_huparams
end

# ╔═╡ ae7aed24-f3ef-4b80-a92c-5ee9a78fc3b4


# ╔═╡ ff78392e-22f4-4396-b743-ed7a080d3cdd
md"""
### 1.4.5. Inputs to the hidden layer
"""

# ╔═╡ 585c907e-3d97-4510-b2fe-19aeb6120ac1
I_vh = inputs_h_from_v(rbm, dataset.spikes);

# ╔═╡ bf17d796-cd72-48a5-80e5-ed0c955155b2
begin
	fing_ivh = Figure()
	Axis(fing_ivh[1, 1], title="Inputs to Hidden", xlabel=L"I_{μ}", ylabel="Density")
	hist!(vec(I_vh), offset=1.e-0, bins=100, color=:black, normalization=:density)
	fing_ivh
end

# ╔═╡ 73b0f369-7299-4a5f-84d5-161fa65871bc
md"""
### 1.4.6. Hidden values
"""

# ╔═╡ 793180cc-4c3e-48bf-8a53-2df3d20c2bca
begin
	fig_hdistr = Figure()
	ax_Ph = Axis(fig_hdistr[1, 1], title="Hidden Values", ylabel="P(h)", xlabel="h")
	density!(vec(sample_from_inputs(rbm.hidden, I_vh)), label="sampled from inputs", color=:grey)
	density!(vec(translated), label="translated", color=(:orange, 0.5))
	fig_hdistr
end

# ╔═╡ c7487833-6cda-4508-8239-df59e7115903
md"""
### 1.4.7. Hidden means and var
"""

# ╔═╡ 7284a1d5-1490-4c5d-a0ba-e12124207bec
begin
	fig_hms = Figure(size=dfsize().*(2,1))
	ax_mh = Axis(fig_hms[1, 1], title="Hidden Mean", xlabel="Mean h", ylabel="#")
	hist!(ax_mh, vec(mean(translated, dims=2)), bins=10, color=:black)
	ax_vh = Axis(fig_hms[1, 2], title="Hidden Variance", xlabel="Var h", ylabel="#")
	hist!(ax_vh, vec(var(translated, dims=2)), bins=10, color=:orange)
	fig_hms
end

# ╔═╡ a86a7868-ffe8-494b-a119-d2b029604818
md"""
### 1.4.8. Translated Activity (v → h)
"""

# ╔═╡ 4f89983e-3050-4026-ab05-ab52fbf8b91a
begin
	fig_htrans = Figure(size=dfsize().*(4,1))
	Axis(fig_htrans[1,1], title="Hidden Activity", xlabel="Time (frames)", ylabel="Hidden Units")
	l = quantile_range(translated)
	heatmap!(translated', colormap=:berlin, colorrange=(-l, +l))
	fig_htrans
end

# ╔═╡ 7c8d75d4-cea0-4115-b119-12024a77cb02
md"""
### 1.4.9. Hidden Correlations
"""

# ╔═╡ ab5791a1-6f13-4473-a7a6-6ed8b1d4ab87
C = cor(translated');

# ╔═╡ b3c83c24-8043-4723-9427-6151591cf8aa
begin
	fig_hcor = Figure()
	Axis(fig_hcor[1,1], title="Hidden Correlations", xlabel="Hidden Units", ylabel="Hidden Units", aspect=1)
	corrplotter!(C)
	fig_hcor
end

# ╔═╡ 04019cd3-dfe0-4c5e-b85d-d8ba8dbd5e79


# ╔═╡ 990e6cfa-2bbc-43f7-a809-ac3645f1a2ac
md"""
### 1.4.10. Spatial Distribution of Weights
"""

# ╔═╡ 56bfbdbf-7bfa-4c96-a6a4-26b2a0f3be0e
@bind h PlutoUI.Slider(1:size(rbm.w,2), default=size(rbm.w,2)//2)

# ╔═╡ dde490fb-7723-4cf3-be1e-e5c186d98ea5
begin
	fig_spatial = Figure()
	Axis(fig_spatial[1,1], aspect=DataAspect())
	neuron2dscatter!(
		dataset.coords[:,1], dataset.coords[:,2],
		rbm.w[:,h],
		cmap=cmap_aseismic(),
		range=(-2,+2),
		edgewidth=0.1, edgecolor=(:grey, 0.1),
	)
	fig_spatial
end

# ╔═╡ 14054ad4-25fa-4d06-920d-82e26cb8c47c


# ╔═╡ a1c524b0-8927-45d6-9a77-5c922a1acd96


# ╔═╡ 3169efe7-ea7e-46f6-ae0e-73e65a91a721


# ╔═╡ 31cc5d3e-fe90-4af5-9f49-56fdb8f8e0d8


# ╔═╡ 1f512e3a-edda-462c-be07-79557684905d


# ╔═╡ b7aca21f-fd58-451d-9aec-e351c2342a24


# ╔═╡ Cell order:
# ╠═9437164a-106d-11f0-1caf-b7d9c6f95f39
# ╠═cf286b16-4948-4730-b096-4f21ef4668c1
# ╠═311a23c8-128c-417d-b9df-291426fefb48
# ╠═9e8dcc5d-51ff-4863-a890-b808bea365a7
# ╠═5124593a-35db-4c8a-b09e-336ffef270bc
# ╠═6aa48d70-e1aa-475e-a68a-5d034365251a
# ╠═3eb2db58-5112-4c2b-9a23-dadb02ed0b9b
# ╟─8f59e4ab-b030-4055-8dae-e21a5142f38b
# ╟─2d1dee61-18b7-4647-847d-644b2ea0f9a7
# ╠═7b785c14-8284-4871-9290-4b2ba632b261
# ╠═a1e4746e-5422-457b-a47b-0bb78ffc9a78
# ╟─a65cd79a-f035-4fd0-96bf-bf9308374a52
# ╠═707c9b54-dc3e-4af8-ae6d-50c9b51f8451
# ╠═fa3dfbcd-cd4a-4e22-8a5a-67fff772841f
# ╟─26619d61-6f82-4252-ad12-cff52eb37e77
# ╠═44699638-977a-4794-9461-85a4cf2f68ac
# ╠═b657a23f-7f77-4b9a-b831-c53b6e4bd803
# ╠═7885f940-f1a5-4519-ad55-21edeb3d3fd5
# ╠═183e0555-c8e0-4242-80af-65c30d0e6b7a
# ╠═c1cabed3-4e3d-4856-9c4b-20d2a45f78b7
# ╟─e04a81ed-1212-4eb1-bec5-af9350ea9884
# ╠═cdae517c-ed57-4894-be97-ddbdb1dcce52
# ╠═1856dc8e-5c91-47af-b400-39197011997d
# ╠═77a1624c-2a2c-4103-bbbd-03ee9e8d7fbf
# ╟─8eac9910-c1cf-4540-90ee-1799d20c741d
# ╠═36bd93b2-ecae-46d2-8e64-745362413113
# ╟─ecefdf2d-2ac6-4dd0-9a6d-73cb48943558
# ╠═c1c6fdfb-8e29-41cf-af76-fab81bbe1245
# ╠═fc1f0766-8fb9-4554-bd7b-fb19a333dfc6
# ╟─6357a560-55ad-48f2-af16-bd95e66ae81e
# ╠═e9232319-cfbe-4239-aeb6-71f9e9543b82
# ╠═9c023023-8eae-4860-9bed-fae1352d6150
# ╟─20003768-f8d5-443d-89c0-e1a1af3ad64f
# ╠═92892579-da39-49a4-a427-96fe5755a71d
# ╟─74975a56-fe4e-4e8a-bfbe-845e8a062155
# ╠═e4586923-9fdc-4d57-ac98-d093fb15f207
# ╠═ae7aed24-f3ef-4b80-a92c-5ee9a78fc3b4
# ╟─ff78392e-22f4-4396-b743-ed7a080d3cdd
# ╠═585c907e-3d97-4510-b2fe-19aeb6120ac1
# ╠═bf17d796-cd72-48a5-80e5-ed0c955155b2
# ╟─73b0f369-7299-4a5f-84d5-161fa65871bc
# ╠═793180cc-4c3e-48bf-8a53-2df3d20c2bca
# ╟─c7487833-6cda-4508-8239-df59e7115903
# ╠═7284a1d5-1490-4c5d-a0ba-e12124207bec
# ╟─a86a7868-ffe8-494b-a119-d2b029604818
# ╠═4f89983e-3050-4026-ab05-ab52fbf8b91a
# ╟─7c8d75d4-cea0-4115-b119-12024a77cb02
# ╠═ab5791a1-6f13-4473-a7a6-6ed8b1d4ab87
# ╠═b3c83c24-8043-4723-9427-6151591cf8aa
# ╠═04019cd3-dfe0-4c5e-b85d-d8ba8dbd5e79
# ╟─990e6cfa-2bbc-43f7-a809-ac3645f1a2ac
# ╠═56bfbdbf-7bfa-4c96-a6a4-26b2a0f3be0e
# ╠═dde490fb-7723-4cf3-be1e-e5c186d98ea5
# ╠═14054ad4-25fa-4d06-920d-82e26cb8c47c
# ╠═a1c524b0-8927-45d6-9a77-5c922a1acd96
# ╠═3169efe7-ea7e-46f6-ae0e-73e65a91a721
# ╠═31cc5d3e-fe90-4af5-9f49-56fdb8f8e0d8
# ╠═1f512e3a-edda-462c-be07-79557684905d
# ╠═b7aca21f-fd58-451d-9aec-e351c2342a24
