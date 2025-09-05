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

# ╔═╡ 164901da-1482-11f0-0f83-6fdbfdebcbae
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ 9408efa2-47a7-4134-9844-7d82f4b0515e
TableOfContents()

# ╔═╡ f2bbd897-7e9a-4915-998b-0dbdb2261800
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ c14b18d9-6e43-46ad-a1d5-4b478fafab8c
md"""
# 1. Fish Selection
"""

# ╔═╡ 8fb54a11-e591-47e6-b2d2-377030825988
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ 4a25ab9e-d07a-45b4-9828-37e406c81494
base_mod = "*_WBSC_M100_l10.02_l2l10";

# ╔═╡ 5e49710d-da07-48e8-8acb-169a26680d85
md"selected teacher : $(@bind teacher Select(FISH))"

# ╔═╡ 2008b514-90da-4d76-9323-666de8adce8b
md"selected student : $(@bind student Select(FISH, default=FISH[2]))"

# ╔═╡ 70f80ad8-6434-4bce-9c25-407a142799a5
if teacher != student
	goodToGo = true
	teacher_rbm_path = LOAD.load_wbscRBM("bRBMs", "bRBM_".*teacher.*base_mod)
	#student_rbm_path = LOAD.load_model(CONV.MODELPATH, "bRBM_".*student.*base_mod)
	teacher_data_path = LOAD.load_dataWBSC(teacher)
	student_data_path = LOAD.load_dataWBSC(student)
	out_rbm_name = "biRBM_$(student)_FROM_$(teacher)$(base_mod[2:end]).h5"
	out_rbm_path = LOAD.RBM_WBSC_biRBM .* "/" .* out_rbm_name
else
	goodToGo = false
end;

# ╔═╡ 96c89860-63ab-42c8-b22c-15a3205f7d5a
md"""
BiTraining :

- Teacher : $(teacher)
  - RBM : $(teacher_rbm_path)
  - DATA : $(teacher_data_path)

- Student : $(student)
  - DATA : $(student_data_path)

→ $(out_rbm_path)
"""

# ╔═╡ 3778a0de-48e1-490b-914e-84510d12929d


# ╔═╡ 2bd24898-6f0c-4a77-afa8-4b43a8c0bf4e


# ╔═╡ 328be301-34de-455f-9ec7-69c75a1d7f38


# ╔═╡ 0797ba8d-61c8-40e0-8636-763f892bed8d


# ╔═╡ 03016241-6e07-4ff8-8be6-a4e27767680c
md"""
# 2. Gathering from Teacher
"""

# ╔═╡ 44790b36-7567-4f64-9f55-5dcc44c7b85b
md"Load Teacher? $(@bind load_teacher CheckBox(default=false))"

# ╔═╡ 7c6427b7-134b-4e03-94eb-50cfa8d0e02d
if load_teacher
	teacher_dataset = load_data(teacher_data_path);
	Trbm, Ttrain_params, Tevaluation, Tsplit, Tgen, Ttranslated = load_brainRBM(teacher_rbm_path);
end

# ╔═╡ 777c4582-0291-42a6-94f8-c36971790628
if load_teacher
	fid = h5open(teacher_rbm_path, "r")
	teacher_maps = Maps(fid["Weight Maps"])
	close(fid)
end

# ╔═╡ b3ddd995-27fa-4cde-8229-dc651f02bca9
md"""
## 2.1. Creating hidden test/train
"""

# ╔═╡ 3c3d232b-c4f8-42db-8cca-64920dd490e6
begin
	n = 1
	Tsamph = sample_h_from_v(Trbm, repeat(teacher_dataset.spikes, 1, 1, n));
	Tsamph = reshape(Tsamph, (size(Tsamph,1), size(Tsamph,2)*size(Tsamph,3)));
	size(Tsamph)
end

# ╔═╡ 9eec1730-ead0-4f55-a598-a20d463cef4b
Tsplitsamph = split_set(Tsamph);

# ╔═╡ 58e3a3f0-8ab2-4f90-8f05-f453375c0547


# ╔═╡ eec84371-bf0a-4bfd-abde-6a2ef103bd87
md"""
# 3. PreInitialising Student
"""

# ╔═╡ d580a6ce-768f-4f6d-9b76-4ca9ff77a861
md"Load Student? $(@bind load_student CheckBox(default=false))"

# ╔═╡ cca50582-372b-4433-b9b7-43a64f21c1fb
if load_student
	student_dataset = load_data(student_data_path);
end

# ╔═╡ 134b1219-d5dd-46bc-9da5-77185a03a098
teacher_dataset

# ╔═╡ 9edd8d62-ecf8-46be-92d3-cd39ecb2da7d
student_dataset

# ╔═╡ 8bc74646-9f64-4870-b95b-efd93857fc8e
md"""
## 3.1. Weights
"""

# ╔═╡ 2e21ca58-42d4-4258-bccf-62e57bc58c28
student_weights = interpolation(
	teacher_maps,
	student_dataset.coords,
	verbose=false,
);

# ╔═╡ f1d19dda-537c-4c53-b750-0596aa800a2c


# ╔═╡ 386caa31-f6b4-4292-b6ca-46e8c223bbb7
md"""
## 3.2. Fields
"""

# ╔═╡ 4a4543e3-4f11-4987-8380-a24ce3f083f2
begin
	function logistic(x::AbstractArray)
	    return @. 1 /(1 + exp(-x))
	end
	function logit(x::AbstractArray;ϵ=1.e-9)
	    clamp!(x, ϵ, 1-ϵ)
	    return @. -log(1/x -1)
	end
end

# ╔═╡ e3943e8f-9ba6-47d5-9732-6643f4322819
begin
	μiS = mean(student_dataset.spikes, dims=2)[:,1];
	θiS = logit(μiS);
end;

# ╔═╡ f0cfebd8-195a-4fdb-8648-92e264f00032


# ╔═╡ 9b1367c3-0f75-4c00-bd9b-7067ad72cad8
md"""
## 3.3. Building student RBM
"""

# ╔═╡ 52574200-8556-433d-a535-628772d646bf
function scale_v(spikes::AbstractMatrix)
   return var(spikes, dims=2)[:,1].*0.5 .+ 1
end

# ╔═╡ 1bb7ce92-b4df-482c-865a-ddd005c8ae76
Srbm_ut = StandardizedRBM(
	Binary(Float32.(θiS[[CartesianIndex()],:])),
	Trbm.hidden,
	student_weights,
	mean(student_dataset.spikes, dims=2)[:,1],
	Trbm.offset_h,
	scale_v(student_dataset.spikes),
	Trbm.scale_h,
);

# ╔═╡ a1757c3b-5d31-471b-8d42-596813e81536
md"""
## 3.4. Evaluation before training
"""

# ╔═╡ 718568a1-5803-4208-a475-e267a22432ac
Ssplit_simple = split_set(student_dataset.spikes);

# ╔═╡ e6d1e867-faa0-4126-8ef0-afbbe8a14e74
Sgen_before = gen_data(
	gpu(Srbm_ut), 
	nsamples=1500,
	nthermal=500,
	nstep=100,
	init="prior",
	verbose=true,
);

# ╔═╡ 46ffef63-3aee-46d3-a539-a47a0efd13fe
Smoments_before = compute_all_moments(Srbm_ut, Ssplit_simple, Sgen_before);

# ╔═╡ 95dc3e1c-e20d-4839-bb33-dab91e060a57
Snrmse_before = nRMSE_from_moments(Smoments_before)

# ╔═╡ fd429c52-735e-40bc-b1ea-5703a9fe3dea
dump_brainRBM(
	joinpath(LOAD.RBM_WBSC_biRBMbefore, out_rbm_name.*"_untrained"),
	Srbm_ut,
	Dict([("trained",false)]),
	Snrmse_before,
	Ssplit_simple,
	Sgen_before,
	translate(cpu(Srbm_ut), student_dataset.spikes);
	comment="untrained $(student) initiated from $(teacher_rbm_path)",
)

# ╔═╡ ddd54f3e-820a-496c-9794-92e1fb63aa5b


# ╔═╡ bb4e637b-8841-4b32-bfaa-241446319d13
md"""
# 4. Training Student
"""

# ╔═╡ 255617d3-efaf-4932-adb4-1c4c5678f24d
#=╠═╡
function training_wrapper_birbm(
  rbm::StandardizedRBM,
  data_v::AbstractArray,
  data_h::AbstractArray,
  λ::Real; 
        
  iters::Int=20000, # 50000
  batchsize::Int=256, # 256
  steps::Int=50, # 50
  lr_start::Number=5.0f-4, # 1f-4
  lr_stop::Number=1.0f-5, # 1f-5
  decay_from::Number=0.25, # 0.25
  l2l1::Number=0.001, # 0.001
  l1::Number=0, # 0
  ϵv::Number=1.0f-1, # 1f-1
  ϵh::Number=0.0f0, # 0f0
  damping::Number=1.0f-1,
  record_ps::Bool=true, # true
  verbose::Bool=true
)

  decay_g = (lr_stop / lr_start)^(1 / (iters * (1 - decay_from)))
  history = MVHistory()
  if verbose
    pbar_id = uuid4()
	  pbar_name = "Training BrainRBM"
    @logmsg ProgressLevel pbar_name progress=nothing _id=pbar_id
  end

  function callback(; rbm, optim, state, iter, vm, hm, vd, hd)
    # learning rate section
    lr = state.w.rule.eta
    if iter > decay_from * iters
      adjust!(state, Float32(lr * decay_g))
    end
    @trace history iter lr


    # progress bar section
    if verbose
      @logmsg ProgressLevel pbar_name progress=iter/iters _id=pbar_id
    end


    # pseudolikelihood section

    if iszero(iter % 200) & record_ps
      lpl = mean(log_pseudolikelihood(rbm, data_v))
      @trace history iter lpl
    end


  end

  optim = Adam(lr_start, (9.0f-1, 999.0f-3), 1.0f-6) # (0f0, 999f-3), 1f-6
  n = size(rbm.visible,1)
  m = size(rbm.hidden,1)
  vm = sample_from_inputs(rbm.visible, gpu(zeros(n, batchsize)))
  hm = sample_from_inputs(rbm.hidden, gpu(zeros(m, batchsize)))
  state, ps = bipcd!(
    rbm, data_v, data_h, λ;
    optim,
    steps=steps,
    batchsize,
    iters=iters,
    vm,hm,
    l2l1_weights=l2l1,
    l1_weights=l1,
    ϵv=ϵv, # 1f-1
    ϵh=ϵh, # 0f0
    damping=damping,
    callback
  )

  return history, Dict([
    ("λ", λ),
    ("iters", iters),
    ("batchsize", batchsize),
    ("steps", steps),
    ("lr_start", lr_start),
    ("lr_stop", lr_stop),
    ("decay_from", decay_from),
    ("l2l1", l2l1),
    ("l1", l1),
    ("ϵv", ϵv),
    ("ϵh", ϵh),
    ("damping", damping),
  ])
end
  ╠═╡ =#

# ╔═╡ 7298f06b-2009-4822-b351-cfc8f33fa7dd
md"""
## 4.1. Creating visible test/train
"""

# ╔═╡ db1029ba-80cd-4d0c-95a2-3299aaf5f05c
md"Lauch datasplit? $(@bind lauch_datasplit CheckBox(default=false))"

# ╔═╡ 2a2ef2d5-accf-44c8-9565-16e80324d7a8
if lauch_datasplit
	ssplt = SectionSplit(student_dataset.spikes, 0.7);
	dsplit = split_set(student_dataset.spikes, ssplt, q=0.1);
end

# ╔═╡ 8aecf36c-fe99-4681-abc9-a81e8adaf0be


# ╔═╡ 060c3292-ecf6-41d6-991e-56ac4bf3ad18


# ╔═╡ 1e1fcd6f-3f42-440d-ba3a-b5758f76d68e
md"""
## 4.3. Training
"""

# ╔═╡ 75e46b25-c85b-4d69-b680-1cbdf06a74b1
md"Lauch training? $(@bind lauch_training CheckBox(default=false))"

# ╔═╡ 0e5d6c38-3651-484f-bd89-ee7a252e1feb
if lauch_training
	Srbm_tt = gpu(Srbm_ut);
end

# ╔═╡ f3645e46-7e35-4cd3-ad04-b2c45010ea50
#=╠═╡
hist, params = training_wrapper_birbm(
    Srbm_tt,
    gpu(dsplit.train),
    gpu(Tsplitsamph.train),
    0.5;
    batchsize=Ttrain_params["batchsize"],
    iters=20000,#Ttrain_params["iters"],
    steps=Ttrain_params["steps"],
    l2l1=Ttrain_params["l2l1"],
    l1=Ttrain_params["l1"],
    record_ps=false,
    ϵv=Ttrain_params["ϵv"],
    ϵh=Ttrain_params["ϵh"],
    damping=Ttrain_params["damping"],
)
  ╠═╡ =#

# ╔═╡ ac37b669-8c25-461a-b22e-1d31e969f853
Sgen = gen_data(Srbm_tt, nsamples=1500, nthermal=500, nstep=100, init="prior", verbose=true);

# ╔═╡ 47321f2c-6a28-41e3-88bd-fa4c889e3cfa
Srbm = cpu(Srbm_tt);

# ╔═╡ 3936dfc8-6c7d-46fa-98f7-649eb9e9bc2f
Smoments = compute_all_moments(Srbm, dsplit, Sgen);

# ╔═╡ 0a2ebb12-e58d-4950-9ad2-8a43e5d8fd0b
Snrmses = nRMSE_from_moments(Smoments)

# ╔═╡ d3d24033-93d5-4046-971e-54dea4115410
#=╠═╡
dump_brainRBM(
	out_rbm_path, 
	Srbm, params, 
	Snrmses, 
	dsplit, Sgen, 
	translate(Srbm, student_dataset.spikes) ; 
	comment="$(student) initiated and bitrained from $(teacher_rbm_path)",
)
  ╠═╡ =#

# ╔═╡ 4729229a-0323-42af-a90b-0d1b44507b2f


# ╔═╡ cb9b67b2-8833-4106-bcb4-efe2826d6778


# ╔═╡ 953da13e-ab1e-418b-838b-f6911a380ac6


# ╔═╡ d8d424e9-c173-47c8-9254-5bdc85161f2a


# ╔═╡ d2ffdbee-4cc2-46b5-ba01-af31cf960e26
#=╠═╡
begin

	# loading modules
	using BrainRBMjulia
	using BrainRBMjulia: StandardizedRBM, Binary
	using BrainRBMjulia: @trace, Adam, MVHistory, ProgressLevel, @logmsg, uuid4, adjust!
	using BiTrainedRBMs: bipcd!
	using HDF5
end
  ╠═╡ =#

# ╔═╡ cca94fb0-aa23-47fe-bb1b-349bdbd45976
# ╠═╡ disabled = true
#=╠═╡
begin
	push!( LOAD_PATH, "/home/matteo/Programs/BiTrainedRBMs.jl/" )
	using BiTrainedRBMs:bipcd!, biinit!
	
	using BrainRBMjulia: Adam, @trace, MVHistory, adjust!, sample_from_inputs, Progress, next!, log_pseudolikelihood
	
	using BiTrainedRBMs
	using RestrictedBoltzmannMachines
	using RestrictedBoltzmannMachines:moments_from_samples, AbstractRule, setup, zerosum!, infinite_minibatches, sample_h_from_h
	using StandardizedRestrictedBoltzmannMachines
	using StandardizedRestrictedBoltzmannMachines:standardize_visible_from_data!, sample_v_from_v, standardize_hidden_from_v!
	
	import BiTrainedRBMs.bipcd!
	import RestrictedBoltzmannMachines.∂free_energy_v
	import RestrictedBoltzmannMachines.∂free_energy_h
	using RestrictedBoltzmannMachines: ∂energy_from_moments, ∂cgfs, grad2ave, ∂interaction_energy, wmean, ∂RBM
	using StandardizedRestrictedBoltzmannMachines: inputs_v_from_h, ∂regularize!, update!, rescale_hidden_activations!
	
	function ∂free_energy(
	    rbm::StandardizedRBM, v::AbstractArray; wts = nothing,
	    moments = moments_from_samples(rbm.visible, v; wts)
	)
	    inputs = inputs_h_from_v(rbm, v)
	    ∂v = ∂energy_from_moments(rbm.visible, moments)
	    ∂Γ = ∂cgfs(rbm.hidden, inputs)
	    h = grad2ave(rbm.hidden, ∂Γ)
	    ∂h = reshape(wmean(-∂Γ; wts, dims = (ndims(rbm.hidden.par) + 1):ndims(∂Γ)), size(rbm.hidden.par))
	    ∂w = ∂interaction_energy(rbm, v, h; wts)
	    return ∂RBM(∂v, ∂h, ∂w)
	end
	∂free_energy_v(rbm::StandardizedRBM, v::AbstractArray; kwargs...) = ∂free_energy(rbm, v; kwargs...)
	
	function ∂free_energy_h(
	    rbm::StandardizedRBM, h::AbstractArray; wts = nothing,
	    moments = moments_from_samples(rbm.hidden, h; wts)
	)
	    inputs = inputs_v_from_h(rbm, h)
	    ∂Γ = ∂cgfs(rbm.visible, inputs)
	    v = grad2ave(rbm.visible, ∂Γ)
	    ∂v = -reshape(wmean(∂Γ; wts, dims = (ndims(rbm.visible.par) + 1):ndims(∂Γ)), size(rbm.visible.par))
	    ∂h = ∂energy_from_moments(rbm.hidden, moments)
	    ∂w = ∂interaction_energy(rbm, v, h; wts)
	    return ∂RBM(∂v, ∂h, ∂w)
	end
	
	
	
	
	function bipcd!(
	    rbm::StandardizedRBM,
	    data_v::AbstractArray,
	    data_h::AbstractArray,
	    λ::Real = 0.5;
	
	    batchsize::Int = 1,
	    iters::Int = 1,
	    steps::Int = 1,
	    shuffle::Bool = true,
	
	    moments_v = moments_from_samples(rbm.visible, data_v),
	    moments_h = moments_from_samples(rbm.hidden, data_h),
	
	    # regularization
	    l2_fields::Real = 0, # visible fields L2 regularization
	    l1_weights::Real = 0, # weights L1 regularization
	    l2_weights::Real = 0, # weights L2 regularization
	    l2l1_weights::Real = 0, # weights L2/L1 regularization
	        
	    # "pseudocount" for estimating variances of v and h and damping
	    damping::Real = 1//100,
	    ϵv::Real = 0, ϵh::Real = 0,
	
	    # Absorb the scale_h into the hidden unit activation (for continuous hidden units).
	    # Results in hidden units with var(h) ~ 1.
	    rescale_hidden::Bool = true,
	
	    zerosum::Bool = true, # zerosum gauge for Potts layers
	
	    vm = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize)),
	    hm = sample_from_inputs(rbm.hidden, Falses(size(rbm.hidden)..., batchsize)),
	
	    # optimiser
	    optim::AbstractRule = Adam(),
	    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w),
	    state = setup(optim, ps), # initialize optimiser state
	        
	    callback = Returns(nothing),
	)
	    @assert size(data_v) == (size(rbm.visible)..., size(data_v)[end])
	    @assert size(data_h) == (size(rbm.hidden)..., size(data_h)[end])
	    @assert 0 ≤ damping ≤ 1
	
	    standardize_visible_from_data!(rbm, data_v; ϵ = ϵv)
	    zerosum && zerosum!(rbm)
	
	    for (iter, (vd,), (hd,)) in zip(
	            1:iters, 
	            infinite_minibatches(data_v; batchsize, shuffle), 
	            infinite_minibatches(data_h; batchsize, shuffle)
	        )
	        # update fantasy chains
	        vm .= sample_v_from_v(rbm, vm; steps)
	        hm .= sample_h_from_h(rbm, hm; steps)
	
	        # update standardization
	        standardize_hidden_from_v!(rbm, vd; damping, ϵ=ϵh)
	
	        # compute gradient
	        ∂d_v = ∂free_energy_v(rbm, vd; moments = moments_v)
	        ∂d_h = ∂free_energy_h(rbm, hd; moments = moments_h)
	        ∂m_v = ∂free_energy_v(rbm, vm)
	        ∂m_h = ∂free_energy_h(rbm, hm)
	        ∂_v = ∂d_v - ∂m_v
	        ∂_h = ∂d_h - ∂m_h
	        ∂ = (1 - λ) * ∂_v + λ * ∂_h
	
	        # weight decay
	        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)
	
	        # feed gradient to Optimiser rule
	        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
	        state, ps = update!(state, ps, gs)
	
	        rescale_hidden && rescale_hidden_activations!(rbm)
	        zerosum && zerosum!(rbm)
	
	        callback(; rbm, optim, state, iter, vm, hm, vd, hd)
	    end
	    return state, ps
	end
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═164901da-1482-11f0-0f83-6fdbfdebcbae
# ╠═d2ffdbee-4cc2-46b5-ba01-af31cf960e26
# ╠═9408efa2-47a7-4134-9844-7d82f4b0515e
# ╠═f2bbd897-7e9a-4915-998b-0dbdb2261800
# ╟─c14b18d9-6e43-46ad-a1d5-4b478fafab8c
# ╠═8fb54a11-e591-47e6-b2d2-377030825988
# ╠═4a25ab9e-d07a-45b4-9828-37e406c81494
# ╠═5e49710d-da07-48e8-8acb-169a26680d85
# ╠═2008b514-90da-4d76-9323-666de8adce8b
# ╠═70f80ad8-6434-4bce-9c25-407a142799a5
# ╟─96c89860-63ab-42c8-b22c-15a3205f7d5a
# ╠═3778a0de-48e1-490b-914e-84510d12929d
# ╠═2bd24898-6f0c-4a77-afa8-4b43a8c0bf4e
# ╠═328be301-34de-455f-9ec7-69c75a1d7f38
# ╠═0797ba8d-61c8-40e0-8636-763f892bed8d
# ╟─03016241-6e07-4ff8-8be6-a4e27767680c
# ╟─44790b36-7567-4f64-9f55-5dcc44c7b85b
# ╠═7c6427b7-134b-4e03-94eb-50cfa8d0e02d
# ╠═777c4582-0291-42a6-94f8-c36971790628
# ╟─b3ddd995-27fa-4cde-8229-dc651f02bca9
# ╠═3c3d232b-c4f8-42db-8cca-64920dd490e6
# ╠═9eec1730-ead0-4f55-a598-a20d463cef4b
# ╠═58e3a3f0-8ab2-4f90-8f05-f453375c0547
# ╟─eec84371-bf0a-4bfd-abde-6a2ef103bd87
# ╟─d580a6ce-768f-4f6d-9b76-4ca9ff77a861
# ╠═cca50582-372b-4433-b9b7-43a64f21c1fb
# ╠═134b1219-d5dd-46bc-9da5-77185a03a098
# ╠═9edd8d62-ecf8-46be-92d3-cd39ecb2da7d
# ╟─8bc74646-9f64-4870-b95b-efd93857fc8e
# ╠═2e21ca58-42d4-4258-bccf-62e57bc58c28
# ╠═f1d19dda-537c-4c53-b750-0596aa800a2c
# ╟─386caa31-f6b4-4292-b6ca-46e8c223bbb7
# ╠═4a4543e3-4f11-4987-8380-a24ce3f083f2
# ╠═e3943e8f-9ba6-47d5-9732-6643f4322819
# ╠═f0cfebd8-195a-4fdb-8648-92e264f00032
# ╟─9b1367c3-0f75-4c00-bd9b-7067ad72cad8
# ╠═52574200-8556-433d-a535-628772d646bf
# ╠═1bb7ce92-b4df-482c-865a-ddd005c8ae76
# ╟─a1757c3b-5d31-471b-8d42-596813e81536
# ╠═718568a1-5803-4208-a475-e267a22432ac
# ╠═e6d1e867-faa0-4126-8ef0-afbbe8a14e74
# ╠═46ffef63-3aee-46d3-a539-a47a0efd13fe
# ╠═95dc3e1c-e20d-4839-bb33-dab91e060a57
# ╠═fd429c52-735e-40bc-b1ea-5703a9fe3dea
# ╠═ddd54f3e-820a-496c-9794-92e1fb63aa5b
# ╟─bb4e637b-8841-4b32-bfaa-241446319d13
# ╟─cca94fb0-aa23-47fe-bb1b-349bdbd45976
# ╟─255617d3-efaf-4932-adb4-1c4c5678f24d
# ╟─7298f06b-2009-4822-b351-cfc8f33fa7dd
# ╟─db1029ba-80cd-4d0c-95a2-3299aaf5f05c
# ╠═2a2ef2d5-accf-44c8-9565-16e80324d7a8
# ╠═8aecf36c-fe99-4681-abc9-a81e8adaf0be
# ╠═060c3292-ecf6-41d6-991e-56ac4bf3ad18
# ╟─1e1fcd6f-3f42-440d-ba3a-b5758f76d68e
# ╟─75e46b25-c85b-4d69-b680-1cbdf06a74b1
# ╠═0e5d6c38-3651-484f-bd89-ee7a252e1feb
# ╠═f3645e46-7e35-4cd3-ad04-b2c45010ea50
# ╠═ac37b669-8c25-461a-b22e-1d31e969f853
# ╠═47321f2c-6a28-41e3-88bd-fa4c889e3cfa
# ╠═3936dfc8-6c7d-46fa-98f7-649eb9e9bc2f
# ╟─0a2ebb12-e58d-4950-9ad2-8a43e5d8fd0b
# ╠═d3d24033-93d5-4046-971e-54dea4115410
# ╠═4729229a-0323-42af-a90b-0d1b44507b2f
# ╠═cb9b67b2-8833-4106-bcb4-efe2826d6778
# ╠═953da13e-ab1e-418b-838b-f6911a380ac6
# ╠═d8d424e9-c173-47c8-9254-5bdc85161f2a
