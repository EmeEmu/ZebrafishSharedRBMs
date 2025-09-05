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

# ╔═╡ a8c48b92-dfee-11ef-150a-ef88c101f70b
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ 2e029a2e-02b7-430c-8ce5-0bed1b891505
begin
	# loading modules
	using BrainRBMjulia
	using CairoMakie

	using BrainRBMjulia: idplotter!, dfsize
end

# ╔═╡ 897caebc-b6de-4135-9004-b0bb2d200096
using BrainRBMjulia: section_moments

# ╔═╡ 434d37fc-0b63-4a21-a212-329fd2ea8dd5
begin
	using Statistics: median
		
	function BuildFitEvaluate(data, dsplit; 
			M=200, 
			l1=0.02, l2l1=0, 
			lr_start=5.e-4, lr_stop=1f-5, 
			steps=15, iters=200000,
			ϵv=1f0, ϵh=0f0, damping=1f-1,
			save="",
			verbose=false
		)
		#dsplit = split_set(data.spikes)
		rbm = BrainRBM(dsplit.train, M)
		rbm, x = gpu(rbm), gpu(dsplit.train)
		history, params = training_wrapper(
			rbm, x; 
			l1=l1, l2l1=l2l1, 
			iters=iters, batchsize=256,
			lr_start=lr_start, lr_stop=lr_stop, decay_from=0.25,
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
		nLLH = reconstruction_likelihood(rbm, dsplit.valid, dsplit.train)
		nrmses["1-nLLH"] = 1-median(nLLH)
		
		if ~isempty(save)
			dump_brainRBM(
				save*".h5", 
				rbm, params, 
				nrmses, 
				dsplit, gen, 
				translate(rbm, data.spikes) ; 
				comment="from crossvalidation (no swap, no reorder)",
			)
		end
		return nrmses
	end
end

# ╔═╡ bcc54b46-aeab-4f05-a0c1-7bc46eebf0d4
TableOfContents()

# ╔═╡ e2a478e6-80db-4649-a309-44d7dd33b38f
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ 7d6ac972-0fb7-4216-8c45-6bf0c95280e0


# ╔═╡ 1087f852-636b-4cee-86ba-535defe005ed
md"""
# 2. Training
"""

# ╔═╡ c0dcfb88-7ed3-4e85-997d-af022e4742f4
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ a5b73bf2-9fe8-4dee-8678-acd3976fdc2c
md"selected fish : $(@bind fish Select(FISH))"

# ╔═╡ dcb712da-d9f5-4c42-b50e-c6471029ccb6
dataset = load_data(LOAD.load_dataWBSC(fish))

# ╔═╡ 66fcd824-e6e2-4e14-93c6-7524f39d8f0e
md"""
## 2.1 Dataset split (van der Plas & Tubiana method)
"""

# ╔═╡ 4a0ce045-c791-498f-9e81-73fdb7400f1f
md"Lauch datasplit? $(@bind lauch_datasplit CheckBox(default=false))"

# ╔═╡ c4d36f28-9085-4eda-94d1-dc9dbd001387
if lauch_datasplit
	ssplt = SectionSplit(dataset.spikes, 0.7);
	dsplit = split_set(dataset.spikes, ssplt, q=0.1);
	mvtrain, mvtest, mvvtrain, mvvtest = section_moments(
		dsplit.train, 
		dsplit.valid, 
		N_vv=10000
	);

	fig_datasplit = Figure(size=dfsize().*(2,1))#(size=(4*100,2*100))
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
	# save("/tmp/datasplit.svg", fig_datasplit)
	fig_datasplit
end

# ╔═╡ b8f985b2-9014-41ab-83a4-8694772d6644


# ╔═╡ 39e15cc5-5357-4c9c-b7d0-73002c3ea8e7
md"""
## 2.2. Training Params
"""

# ╔═╡ 93fed8e1-683c-445b-97e7-e0a65624d415
md"M (nb of hiddens) = $(@bind M NumberField(1:1000, default=100))"

# ╔═╡ 69d80f06-bd38-4a2a-916a-0b6e76d11cee
md"λ1  (L1   reg) = $(@bind λ1 NumberField(0:0.001:1, default=0.02))"

# ╔═╡ 01badb8d-ab31-4b64-a360-83b934dde9d4
md"λ21 (L2L1 reg) = $(@bind λ21 NumberField(0:1, default=0.0))"

# ╔═╡ 93426b0a-ac79-4ab8-9e97-81da2ef59f05


# ╔═╡ 378248a6-0f3b-49ff-ac2f-c410b0b9bb6f
md"""
## 2.3. Single Trainging
"""

# ╔═╡ 1901fc1c-aec4-4b45-96c8-22a1c9baf762
md"Lauch single training? $(@bind lauch_single CheckBox(default=false))"

# ╔═╡ 52836015-ea0a-4cb1-b91f-29aaa6e7a6a1
if lauch_single
	BuildFitEvaluate(
		dataset, dsplit; 
		M=M, 
		l1=λ1, l2l1=λ21, 
		lr_start=5.e-4, lr_stop=1f-5, 
		steps=15, iters=200000,
		ϵv=1f0, ϵh=0f0, damping=1f-1,
		save=joinpath(
			LOAD.RBM_WBSC_REPEAT, 
			"bRBM_$(dataset.name)_M$(M)_l1$(λ1)_l2l1$(λ21)"
		),
		verbose=false
	)
end

# ╔═╡ 7f6e1d2f-076b-47ec-a29b-475ca412d1ff


# ╔═╡ d7fd048f-2121-4ec8-9aa0-c54ca7398b16
md"""
## 2.4. Multitraining
"""

# ╔═╡ 48782a04-af2e-45dc-a925-6e4ef6db9449
begin
	import Statistics.mean
	mean(dico::Vector{Dict}) = Dict([(k,mean([d[k] for d in dico])) for k in keys(dico[1])])
	
	function BuildFitEvaluate_repeats(data, dsplit; 
	        M=200, 
	        l1=0.02, l2l1=0, 
	        lr_start=5.e-4, lr_stop=1f-5, 
	        steps=15, iters=200000,
	        ϵv=1f0, ϵh=0f0, damping=1f-1,
	        save="",
	        R=10,
	    )
	    D = Dict[]
	    for r in 1:R
			print("repeat r=$(r)")
	        push!(D,BuildFitEvaluate(data, dsplit;
	                M=M, 
	                l1=l1, l2l1=l2l1, 
	                lr_start=lr_start, lr_stop=lr_stop, 
	                steps=steps, iters=iters,
	                ϵv=ϵv, ϵh=ϵh, damping=damping,
	                save=save*"_rep$(r)",
	            ))
	    end
	    return mean(D)
	end
end

# ╔═╡ 690a16fe-cd02-4931-bb66-c7a29af747f8
md"Reps (nb of repetitions) = $(@bind R NumberField(1:1000, default=5))"

# ╔═╡ 7e8a2610-aadc-42f2-a128-bb084f191d73
md"Lauch multiple trainings ? $(@bind lauch_multi CheckBox(default=false))"

# ╔═╡ 9f6eaea7-fe7d-47b9-8443-933354d8d415
if lauch_multi
	BuildFitEvaluate_repeats(
		dataset, dsplit,
		save=joinpath(
			LOAD.RBM_WBSC_REPEAT, 
			"bRBM_$(dataset.name)_M$(M)_l1$(λ1)_l2l1$(λ21)"
		),
		M=M,
		steps=15,
		iters=200000,
		ϵv=1,
		l1=λ1, l2l1=λ21, 
		R=5,
	)
end

# ╔═╡ a5470e2f-b825-46fd-8765-4073364b2a5a


# ╔═╡ d5855a3c-2f28-43e2-86b5-8182f3532d17


# ╔═╡ 5f4bf2b8-7a05-425b-a024-e85f9f0c61e3


# ╔═╡ Cell order:
# ╠═a8c48b92-dfee-11ef-150a-ef88c101f70b
# ╠═2e029a2e-02b7-430c-8ce5-0bed1b891505
# ╠═bcc54b46-aeab-4f05-a0c1-7bc46eebf0d4
# ╠═e2a478e6-80db-4649-a309-44d7dd33b38f
# ╠═7d6ac972-0fb7-4216-8c45-6bf0c95280e0
# ╟─1087f852-636b-4cee-86ba-535defe005ed
# ╠═c0dcfb88-7ed3-4e85-997d-af022e4742f4
# ╠═a5b73bf2-9fe8-4dee-8678-acd3976fdc2c
# ╠═dcb712da-d9f5-4c42-b50e-c6471029ccb6
# ╟─66fcd824-e6e2-4e14-93c6-7524f39d8f0e
# ╠═897caebc-b6de-4135-9004-b0bb2d200096
# ╟─4a0ce045-c791-498f-9e81-73fdb7400f1f
# ╠═c4d36f28-9085-4eda-94d1-dc9dbd001387
# ╠═b8f985b2-9014-41ab-83a4-8694772d6644
# ╟─39e15cc5-5357-4c9c-b7d0-73002c3ea8e7
# ╟─93fed8e1-683c-445b-97e7-e0a65624d415
# ╟─69d80f06-bd38-4a2a-916a-0b6e76d11cee
# ╟─01badb8d-ab31-4b64-a360-83b934dde9d4
# ╠═93426b0a-ac79-4ab8-9e97-81da2ef59f05
# ╟─378248a6-0f3b-49ff-ac2f-c410b0b9bb6f
# ╟─434d37fc-0b63-4a21-a212-329fd2ea8dd5
# ╟─1901fc1c-aec4-4b45-96c8-22a1c9baf762
# ╠═52836015-ea0a-4cb1-b91f-29aaa6e7a6a1
# ╠═7f6e1d2f-076b-47ec-a29b-475ca412d1ff
# ╟─d7fd048f-2121-4ec8-9aa0-c54ca7398b16
# ╟─48782a04-af2e-45dc-a925-6e4ef6db9449
# ╟─690a16fe-cd02-4931-bb66-c7a29af747f8
# ╟─7e8a2610-aadc-42f2-a128-bb084f191d73
# ╠═9f6eaea7-fe7d-47b9-8443-933354d8d415
# ╠═a5470e2f-b825-46fd-8765-4073364b2a5a
# ╠═d5855a3c-2f28-43e2-86b5-8182f3532d17
# ╠═5f4bf2b8-7a05-425b-a024-e85f9f0c61e3
