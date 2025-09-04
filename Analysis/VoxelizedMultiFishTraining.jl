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

# ╔═╡ 759a851c-560e-11f0-1748-91c34f28a7fd
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ 67cb1a72-2756-4dec-a88a-e1383b393aea
begin
	# loading modules
	using BrainRBMjulia
	using CairoMakie

	using BrainRBMjulia: idplotter!, dfsize
end

# ╔═╡ 3efe6faf-e390-4b15-a236-30bf8b208cff
TableOfContents()

# ╔═╡ 6c978a6d-6db6-453b-ad77-fef99f59da51
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ da2a83da-6435-474a-aa34-ae0bc7b4abc2
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
				comment="multifish voxelized",
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

# ╔═╡ 93e534f0-b1e6-4b15-8681-00cf6e4fdbbe


# ╔═╡ 97c8709f-28f2-4963-9866-6a3d594c1d5c
md"""
# 1. Fish Selection
"""

# ╔═╡ 6a6b8723-96c2-4760-9f54-98c1a3be8bc1
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ a9490d1e-679d-4807-9fee-78ddd0a92497
voxel_size = 20.; # in μm

# ╔═╡ eba28467-25da-4688-a5b2-b00b6a03ee9f


# ╔═╡ 1a272691-5209-4567-8aec-be72b1839fcd
md"""
## 2.2. Training Params
"""

# ╔═╡ 40da4bb7-20a5-405c-a1b9-c0ff0e772126
md"M (nb of hiddens) = $(@bind M NumberField(1:1000, default=40))"

# ╔═╡ c2c77d58-e796-4b2e-b040-6641bfdfca9f
md"λ21 (L2L1 reg) = $(@bind λ21 NumberField(0:0.01:1, default=0.1))"

# ╔═╡ fc0e29d7-ba36-4c92-a30c-2e652b71a944


# ╔═╡ 02e06f03-c3b0-4681-b878-f3810e68cb71
md"""
# 2.3. Dataset Building
"""

# ╔═╡ fbcdc2f3-3ca1-4467-9dc2-bc4ba095b1da
spikes = hcat(
	[load_data(LOAD.load_dataVOX(fish, voxel_size)).spikes for fish in FISH]...
)

# ╔═╡ c4a0b6b4-d1e3-4979-8b6b-c9041ffbc15d
dataset = Data(
	"multivoxelized_$(length(FISH))fish_$(voxel_size)vox",
	spikes,
	load_data(LOAD.load_dataVOX(FISH[1], voxel_size)).coords
)

# ╔═╡ b3df530a-fbe5-4b93-8217-40d2ef9d062d
md"""
## 2.4. Training
"""

# ╔═╡ 3640f899-1da1-4727-b292-69780e63229b
md"Lauch multiple trainings ? $(@bind lauch_multi CheckBox(default=false))"

# ╔═╡ 10c1145a-80c1-4d35-8f95-35db4c6f3eb2
if lauch_multi
	dsplit = split_set(spikes, p_train=0.75)
	MultiBuildFitEvaluatefunction(
		dataset, dsplit,
		M=M, l2l1=λ21,
		iters=20000*length(FISH),
		nrep=10,
	)
end

# ╔═╡ 2658545b-1887-459d-a8c9-403c96a3bcc6


# ╔═╡ Cell order:
# ╠═759a851c-560e-11f0-1748-91c34f28a7fd
# ╠═67cb1a72-2756-4dec-a88a-e1383b393aea
# ╠═3efe6faf-e390-4b15-a236-30bf8b208cff
# ╠═6c978a6d-6db6-453b-ad77-fef99f59da51
# ╠═93e534f0-b1e6-4b15-8681-00cf6e4fdbbe
# ╟─97c8709f-28f2-4963-9866-6a3d594c1d5c
# ╠═6a6b8723-96c2-4760-9f54-98c1a3be8bc1
# ╠═a9490d1e-679d-4807-9fee-78ddd0a92497
# ╠═eba28467-25da-4688-a5b2-b00b6a03ee9f
# ╟─1a272691-5209-4567-8aec-be72b1839fcd
# ╟─40da4bb7-20a5-405c-a1b9-c0ff0e772126
# ╟─c2c77d58-e796-4b2e-b040-6641bfdfca9f
# ╟─da2a83da-6435-474a-aa34-ae0bc7b4abc2
# ╠═fc0e29d7-ba36-4c92-a30c-2e652b71a944
# ╟─02e06f03-c3b0-4681-b878-f3810e68cb71
# ╠═fbcdc2f3-3ca1-4467-9dc2-bc4ba095b1da
# ╠═c4a0b6b4-d1e3-4979-8b6b-c9041ffbc15d
# ╟─b3df530a-fbe5-4b93-8217-40d2ef9d062d
# ╟─3640f899-1da1-4727-b292-69780e63229b
# ╠═10c1145a-80c1-4d35-8f95-35db4c6f3eb2
# ╠═2658545b-1887-459d-a8c9-403c96a3bcc6
