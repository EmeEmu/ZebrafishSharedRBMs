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

# ╔═╡ b00e264e-4167-11f0-1d02-17351c33e978
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ afa1779b-0f47-4613-abcd-b428d54629f3
begin
	# loading modules
	using BrainRBMjulia
	using CairoMakie

	using BrainRBMjulia: idplotter!, dfsize
end

# ╔═╡ 9d8e8e73-c8b1-4670-8a0b-e02f2e9c244b
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
	
	
end

# ╔═╡ 0f205755-f0ab-4714-acdb-f906120b3b26


# ╔═╡ 3fde0b95-1a1a-42a2-bc3c-d011986c9aa6
TableOfContents()

# ╔═╡ b3c5e6a6-5e3c-409d-97fe-8d3e682b716f
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ 2549c48c-9145-409c-bbd9-35ab66e12845


# ╔═╡ 2a552981-72a5-47f8-8df1-ad30e9bb8072
md"""
# 1. Fish Selection
"""

# ╔═╡ b526951b-e9c1-44dc-9160-4bcaadfc4631
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ 17667723-69aa-46eb-bdcf-f21d1b01bc47
md"selected fish : $(@bind fish Select(FISH))"

# ╔═╡ 01bd126b-578e-4615-b56e-7cf1f68aef6e


# ╔═╡ 5aee1e9a-d89a-4fad-821d-0646d754397c


# ╔═╡ 78a882d2-fb23-491d-b2fe-91967f9f2476
md"""
# 2. Cross val Ranges
"""

# ╔═╡ beef9225-be47-4a4d-a5fc-31adb40529a5
λ21s = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

# ╔═╡ e930500d-2d27-4aca-b646-de5ae8cf72c3
Ms = [10, 20, 40, 60, 80]

# ╔═╡ 208c2d4a-f93d-40ab-b9bb-266daa17b996
Vs = [20]#[10, 20, 30, 40, 50]

# ╔═╡ 17457e73-3059-41c5-a025-37af422f104f
N_rep = 5

# ╔═╡ 59e6b03f-df53-41c1-a322-6781189a7e59
n_tot_trainings = length(λ21s) * length(Ms) * length(Vs) * N_rep

# ╔═╡ 533b9f97-76c8-401a-8c19-1937b89db15a
estimated_training_time = 20 #mins

# ╔═╡ 2d050d0e-8aa2-43e2-8b88-233e5dabb401
estimated_total_duration = n_tot_trainings * estimated_training_time / 60 / 24 #days

# ╔═╡ 9e865f08-23e8-4757-aeba-bf604ad71a5c


# ╔═╡ 1b6194b6-ec5a-48f7-b94b-384cfe9342ae


# ╔═╡ 6aa666d3-2a52-4f66-91de-7e5739138311
md"""
# 3. Training
"""

# ╔═╡ 1aa7bbfe-8436-4afc-a509-3eab47bb0e7e
function MultiBuildFitEvaluatefunction(data, dsplit; 
			M=30, 
			l1=0., l2l1=0.09, 
			lr_start=5.e-4, lr_stop=1f-5, 
			steps=50, iters=20000,
			ϵv=100., ϵh=0., damping=1f-1,
			nrep=5,
		)
	outpathg = joinpath(
		LOAD.RBM_VOXCROSSVAL,
		"$(data.name)_M$(M)_l2l1$(l2l1)",
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

# ╔═╡ 605138a1-236b-43e4-8133-6c3c4019312e


# ╔═╡ f063acf3-ea57-4a5c-a57a-f5f6caa49655


# ╔═╡ 5502e653-db18-4ef0-8b04-d75c1627b0d1


# ╔═╡ 1ddf4821-57aa-45a5-bf67-3e44dbb6e1bf
md"Lauch trainings ? $(@bind lauch_crossval CheckBox(default=false))"

# ╔═╡ e21be015-b843-42d0-9f14-eccfd13c3850
if lauch_crossval
	for voxsize in Vs
		dataset = load_data(LOAD.load_dataVOX(fish, voxsize))
		# ssplt = SectionSplit(
		# 	dataset.spikes, 
		# 	0.7, 
		# 	N_vv=min(1000, size(dataset.spikes, 1))
		# );
		# dsplit = split_set(dataset.spikes, ssplt, q=0.1);
		dsplit = split_set(dataset.spikes, p_train=0.75)

		for M in Ms
			for λ in λ21s
				MultiBuildFitEvaluatefunction(
					dataset, dsplit,
					M=M, l2l1=λ,
				)
			end
		end
		
	end
end

# ╔═╡ Cell order:
# ╠═b00e264e-4167-11f0-1d02-17351c33e978
# ╠═0f205755-f0ab-4714-acdb-f906120b3b26
# ╠═afa1779b-0f47-4613-abcd-b428d54629f3
# ╠═3fde0b95-1a1a-42a2-bc3c-d011986c9aa6
# ╠═b3c5e6a6-5e3c-409d-97fe-8d3e682b716f
# ╠═2549c48c-9145-409c-bbd9-35ab66e12845
# ╟─2a552981-72a5-47f8-8df1-ad30e9bb8072
# ╠═b526951b-e9c1-44dc-9160-4bcaadfc4631
# ╟─17667723-69aa-46eb-bdcf-f21d1b01bc47
# ╠═01bd126b-578e-4615-b56e-7cf1f68aef6e
# ╠═5aee1e9a-d89a-4fad-821d-0646d754397c
# ╟─78a882d2-fb23-491d-b2fe-91967f9f2476
# ╟─beef9225-be47-4a4d-a5fc-31adb40529a5
# ╟─e930500d-2d27-4aca-b646-de5ae8cf72c3
# ╟─208c2d4a-f93d-40ab-b9bb-266daa17b996
# ╟─17457e73-3059-41c5-a025-37af422f104f
# ╟─59e6b03f-df53-41c1-a322-6781189a7e59
# ╟─533b9f97-76c8-401a-8c19-1937b89db15a
# ╠═2d050d0e-8aa2-43e2-8b88-233e5dabb401
# ╠═9e865f08-23e8-4757-aeba-bf604ad71a5c
# ╠═1b6194b6-ec5a-48f7-b94b-384cfe9342ae
# ╟─6aa666d3-2a52-4f66-91de-7e5739138311
# ╠═9d8e8e73-c8b1-4670-8a0b-e02f2e9c244b
# ╠═1aa7bbfe-8436-4afc-a509-3eab47bb0e7e
# ╠═605138a1-236b-43e4-8133-6c3c4019312e
# ╠═f063acf3-ea57-4a5c-a57a-f5f6caa49655
# ╠═5502e653-db18-4ef0-8b04-d75c1627b0d1
# ╟─1ddf4821-57aa-45a5-bf67-3e44dbb6e1bf
# ╠═e21be015-b843-42d0-9f14-eccfd13c3850
