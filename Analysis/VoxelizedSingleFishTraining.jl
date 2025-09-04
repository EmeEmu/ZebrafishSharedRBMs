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

# ╔═╡ e12a4f9a-3015-11f0-2c9b-efb82594ceaf
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ fe14dce0-ac8c-4037-8ad0-8b7b3d292e11
begin
	# loading modules
	using BrainRBMjulia
	using CairoMakie

	using BrainRBMjulia: idplotter!, dfsize
end

# ╔═╡ 6ad2f14c-8daf-45b4-9a8c-2ab8160c8619
TableOfContents()

# ╔═╡ b47416b6-1c69-4aca-b6d8-bffd3f573b4f
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ cee43f7e-d5a2-4ff6-ad5f-461e4a200c9f
begin
	using BrainRBMjulia: RBM, Gaussian, xReLU, initialize!, standardize
	using Statistics: median
	
	function BuildFitEvaluate(data, dsplit; 
			M=30, 
			l1=0., l2l1=0.09, 
			lr_start=5.e-4, lr_stop=1f-5, 
			steps=50, iters=20000,
			ϵv=100., ϵh=0., damping=1f-1,
			save="",
			verbose=false
		)
		N = size(dsplit.train, 1)
		rbm = RBM(Gaussian((N,)), xReLU((M,)), randn(Float64, (N,M))*0.01);
		initialize!(rbm, dsplit.train);
		rbm = standardize(rbm)
		
		rbm, x = gpu(rbm), gpu(dsplit.train)
		history, params = training_wrapper(
			rbm, x; 
			l1=l1, l2l1=l2l1, 
			iters=iters, batchsize=100,
			lr_start=lr_start, lr_stop=lr_stop, decay_from=0.5,
			steps=steps,
			ϵv=ϵv, ϵh=ϵh, damping=damping,
			record_ps=false,
			verbose=verbose,
		)
		rbm = swap_hidden_sign(rbm)
		reorder_hus!(rbm, dsplit.train)
		rbm = gpu(rbm)
		gen = gen_data(rbm, nsamples=1500, nthermal=500, nstep=100, init="prior", verbose=verbose)
		rbm = cpu(rbm)
		moments = compute_all_moments(rbm, dsplit, gen)
		nrmses = nRMSE_from_moments(moments)
		
		if ~isempty(save)
			dump_brainRBM(
				save, 
				rbm, params, 
				nrmses, 
				dsplit, gen, 
				translate(rbm, data.spikes) ; 
				comment="from crossvalidation (no swap, no reorder)",
			)
			return nrmses
		else
			return rbm, params, nrmses, dsplit, gen, translate(rbm, data.spikes)
		end
	end

	function MultiBuildFitEvaluatefunction(data, dsplit; 
			M=30, 
			l1=0., l2l1=0.09, 
			lr_start=5.e-4, lr_stop=1f-5, 
			steps=50, iters=20000,
			ϵv=100., ϵh=0., damping=1f-1,
			nrep=10,
		)
	outpathg = joinpath(
		LOAD.RBM_VOXREPEAT,
		"vRBMr_$(data.name)_M$(M)_l2l1$(l2l1)",
	)
	for i in 1:nrep
		outpath = outpathg .* "_rep$i.h5"
		if isfile(outpath)
			@warn "$(outpath) already exists. Skipping."
			continue
		end
		BuildFitEvaluate(
			data, dsplit;
			M, l1, l2l1, lr_start, lr_stop, steps, iters, ϵv, ϵh, damping,
			save=outpath
		)
	end
	return outpathg
end
	
	
end

# ╔═╡ cd473272-9bec-4af3-a894-092ee341dce9


# ╔═╡ a8db3f09-c4a7-4e9e-9b65-9f81b8368048
md"""
# 1. Fish Selection
"""

# ╔═╡ da8b811e-9267-48cc-9727-862c05fe4945
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ db4b1ece-42cc-4e86-a58f-cc9ca46dd05b
voxel_size = 20.; # in μm

# ╔═╡ 98b68e8c-e404-4123-b72d-976bdb001b96
md"selected fish : $(@bind fish Select(FISH))"

# ╔═╡ c661e373-2ff5-4d9f-9692-6655aff9bd01
md"""
## 2.2. Training Params
"""

# ╔═╡ e05a2b0a-421b-4c47-8d97-9534cbfb24c1
md"M (nb of hiddens) = $(@bind M NumberField(1:1000, default=40))"

# ╔═╡ f1574487-fd33-49a6-bb56-9d8f56ca5f4b
md"λ21 (L2L1 reg) = $(@bind λ21 NumberField(0:0.01:1, default=0.2))"

# ╔═╡ 4602d616-6034-4d7e-a15d-325a51084449
md"""
# 2. Single Training
"""

# ╔═╡ ae9a0684-a98e-4689-a9f2-f6e5e3f9830d
md"""
## 2.1. Dataset Split
"""

# ╔═╡ 8708da79-3df1-4af6-8d9f-559485da40fb
dataset = load_data(LOAD.load_dataVOX(fish, voxel_size))

# ╔═╡ d960ac09-961f-49c1-bf33-26abff3df64f
md"Lauch datasplit? $(@bind lauch_datasplit CheckBox(default=false))"

# ╔═╡ 988ce019-8c58-401c-a5d3-86816c4a869e
if lauch_datasplit
	ssplt = SectionSplit(dataset.spikes, 0.7, N_vv=1000);
	dsplit = split_set(dataset.spikes, ssplt, q=0.1);
	mvtrain, mvtest, mvvtrain, mvvtest = section_moments(
		dsplit.train, 
		dsplit.valid, 
		N_vv=1000
	);

	fig_datasplit = Figure()#size=dfsize().*(2,1))#(size=(4*100,2*100))
	ax = Axis(
		fig_datasplit[1,1], 
		title=L"$\langle v \rangle$", 
		xlabel="train", 
		ylabel="valid"
	)
	idplotter!(mvtrain, mvtest)
	ax = Axis(
	    fig_datasplit[1,2], 
	    title=L"$\langle vv \rangle - \langle v \rangle\langle v \rangle$", 
	    xlabel="train", ylabel="valid"
	)
	idplotter!(mvvtrain, mvvtest)
	fig_datasplit
end

# ╔═╡ d3cc5d7a-612c-4405-b4c3-2690d4390e51
md"""
## 2.3. Single Training
"""

# ╔═╡ 1f0ff5b1-1997-465c-a0a4-2efcb68e9aa7
md"Lauch single training? $(@bind lauch_single CheckBox(default=false))"

# ╔═╡ 707ab618-07f2-4b25-a5fa-e70a56fcb9d0
if lauch_single
	rbm_path = BuildFitEvaluate(
		dataset, dsplit;
		M=M, l2l1=λ21,
		iters=20000,
		verbose=true,
		save=joinpath(LOAD.RBM_VOX, "vRBM_$(dataset.name)_M$(M)_l2l1_$(λ21)"),
	)
end

# ╔═╡ a0349c26-42cb-44e0-8d75-be3cfd061322


# ╔═╡ ac720bee-cfea-49eb-9263-689b427d37d9
md"""
# 3. Multiple Trainings
"""

# ╔═╡ 9fece380-a520-4025-a94c-bd328872b4da
md"Lauch multiple trainings ? $(@bind lauch_multi CheckBox(default=false))"

# ╔═╡ ddb6a2f7-45ce-444d-8350-e38d08abe4b0
if lauch_multi
	for fish in FISH
		dataset = load_data(LOAD.load_dataVOX(fish, voxel_size))
		# ssplt = SectionSplit(dataset.spikes, 0.7, N_vv=1000);
		# dsplit = split_set(dataset.spikes, ssplt, q=0.1);
		dsplit = split_set(dataset.spikes, p_train=0.75)

		MultiBuildFitEvaluatefunction(
			dataset, dsplit,
			M=M, l2l1=λ21,
		)
	end
end

# ╔═╡ 9c52524b-c894-4e5c-b8bc-80302f4b2dc3


# ╔═╡ f747874a-3718-4d5f-956b-01426beb340b


# ╔═╡ 908166f3-c57f-4cc8-b6ac-58370d643fde


# ╔═╡ Cell order:
# ╠═e12a4f9a-3015-11f0-2c9b-efb82594ceaf
# ╠═fe14dce0-ac8c-4037-8ad0-8b7b3d292e11
# ╠═6ad2f14c-8daf-45b4-9a8c-2ab8160c8619
# ╠═b47416b6-1c69-4aca-b6d8-bffd3f573b4f
# ╠═cd473272-9bec-4af3-a894-092ee341dce9
# ╠═a8db3f09-c4a7-4e9e-9b65-9f81b8368048
# ╠═da8b811e-9267-48cc-9727-862c05fe4945
# ╠═db4b1ece-42cc-4e86-a58f-cc9ca46dd05b
# ╟─98b68e8c-e404-4123-b72d-976bdb001b96
# ╟─c661e373-2ff5-4d9f-9692-6655aff9bd01
# ╟─e05a2b0a-421b-4c47-8d97-9534cbfb24c1
# ╟─f1574487-fd33-49a6-bb56-9d8f56ca5f4b
# ╟─cee43f7e-d5a2-4ff6-ad5f-461e4a200c9f
# ╟─4602d616-6034-4d7e-a15d-325a51084449
# ╟─ae9a0684-a98e-4689-a9f2-f6e5e3f9830d
# ╠═8708da79-3df1-4af6-8d9f-559485da40fb
# ╟─d960ac09-961f-49c1-bf33-26abff3df64f
# ╠═988ce019-8c58-401c-a5d3-86816c4a869e
# ╟─d3cc5d7a-612c-4405-b4c3-2690d4390e51
# ╟─1f0ff5b1-1997-465c-a0a4-2efcb68e9aa7
# ╠═707ab618-07f2-4b25-a5fa-e70a56fcb9d0
# ╠═a0349c26-42cb-44e0-8d75-be3cfd061322
# ╟─ac720bee-cfea-49eb-9263-689b427d37d9
# ╟─9fece380-a520-4025-a94c-bd328872b4da
# ╠═ddb6a2f7-45ce-444d-8350-e38d08abe4b0
# ╠═9c52524b-c894-4e5c-b8bc-80302f4b2dc3
# ╠═f747874a-3718-4d5f-956b-01426beb340b
# ╠═908166f3-c57f-4cc8-b6ac-58370d643fde
