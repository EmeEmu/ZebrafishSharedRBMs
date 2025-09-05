### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 859e9214-3568-11f0-2108-85a2152204f2
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ 6c2c8e99-3c8b-4823-80ca-d3af7049a897
begin
	using BrainRBMjulia
	using CairoMakie
	using BrainRBMjulia: idplotter!, neuron2dscatter!, cmap_aseismic, quantile_range, dfsize
	using BrainRBMjulia: StandardizedRBM
	using Random
	using HDF5
	using FLoops

	include(joinpath(dirname(Base.current_project()), "Misc_Code", "KernelInterpolation.jl"))
end

# ╔═╡ 8ac34421-413d-4b3a-abd5-97189c6a3e11
TableOfContents()

# ╔═╡ 770fcfab-ebf9-4a6a-bace-7cc2c017d2cc
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ 15d89d71-c46d-4a47-abcd-e3d903cca1d3
md"""
!!! warn "Warning"
	This notebook precomputes many things around the translation of neuronal activity between fish. This computation is VERY long and requires a lot of disk space (~50GB), therefore we provide the resulting file as a DataDeps.

	If you still want to recompute it, enable the cells bellow.
"""

# ╔═╡ af393427-aaef-4261-ad91-b913ba964ce9
md"""
# 1. Fish and Training Base
"""

# ╔═╡ 8516860a-86dc-474b-b4c7-54fe8aa39835
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ f93a12f9-ab6e-49e6-ac6d-0d32753f3756
base_mod = "*_WBSC_M100_l10.02_l2l10";

# ╔═╡ 27183b44-1dca-40b5-a239-5586f6f23b76


# ╔═╡ 89249d61-b795-46b8-bc13-644c609f9b60
md"""
# 2. Statistics
"""

# ╔═╡ 433a9ab5-0739-4abf-8c6b-c152e52e2517
begin	
	function compute_v_moments(
	    v::AbstractMatrix,
	    selected_indices::Union{Nothing,AbstractVector{<:Integer}}=nothing
	)::Dict{String,Matrix}
	
	    lv = size(v, 2)
	    v = Float32.(v)
	
	    mu_v = mean(v, dims=2)
	
	    if selected_indices !== nothing
	        v_subset = v[selected_indices, :]
	        cov_vv = cov(v_subset, dims=2, corrected=false)
	    else
	        cov_vv = cov(v, dims=2, corrected=false)
	    end
	
	    return Dict(
	        "<v>" => mu_v,
	        "<vv> - <v><v>" => cov_vv,
	    )
	end
	function compute_v_moments(
		v1::AbstractMatrix,
		v2::AbstractMatrix;
		max_vv::Union{Nothing, Int}=1000
	)
		@assert size(v1,1) == size(v2,1)

		selected_indices = max_vv === nothing || max_vv >= size(v1, 1) ?
				   nothing :
				   sort(randperm(size(v1, 1))[1:max_vv])

		mv1 = compute_v_moments(v1, selected_indices)
		mv2 = compute_v_moments(v2, selected_indices)
		return (
			permutedims(hcat(mv1["<v>"], mv2["<v>"])),
			permutedims(cat(mv1["<vv> - <v><v>"], mv2["<vv> - <v><v>"], dims=3), (3,1,2))
		)
	end
	
	function repeat_samp_vh(rbm::StandardizedRBM, v::AbstractMatrix; R::Int=5)
		return sample_h_from_v(rbm, repeat(v,1,R))
	end
	
	function repeat_samp_vv(rbm::StandardizedRBM, v::AbstractMatrix; R::Int=5)
		return sample_v_from_h(rbm, repeat_samp_vh(rbm, v;R))
	end

	function repeat_samp_vv(rbm1::StandardizedRBM, rbm2::StandardizedRBM, v::AbstractMatrix; R::Int=5)
		@assert size(v,1) == size(rbm1.visible, 1)
		@assert size(rbm1.hidden, 1) == size(rbm2.hidden, 1)
		return sample_v_from_h(rbm2, sample_h_from_v(rbm1, repeat(v,1,R)))
	end
	
	
end

# ╔═╡ eb63e47e-5128-4d81-a00c-9d8c9afbcc82
begin
	function get_teacher_student(
		f::HDF5.File, 
		teacher_i::Int, 
		student_j::Int,
		moment::String,
		rtype::String,
	)
		@assert moment ∈ ["<v>", "<vv>"]
		@assert rtype ∈ ["samp", "TS", "ST", "TST", "STS"]
	
		if (teacher_i == student_j)
			if (rtype == "samp")
				# looking at self-teacher
				m1 = f["$moment/Teacher_$teacher_i/data"]
				m2 = f["$moment/Teacher_$teacher_i/samp"]
			else
				# impossible
				@warn "impossible request i=$teacher_i j=$student_j type=$rtype"
				return nothing, nothing
			end
			
		elseif rtype == "samp"
			m1 = f["$moment/Teacher_$teacher_i/Student_$student_j/data"]
			m2 = f["$moment/Teacher_$teacher_i/Student_$student_j/samp"]
			
		elseif rtype ∈ ["TS", "ST", "TST", "STS"]
			end_ = rtype[end]
			if end_ == 'T'
				m1 = f["$moment/Teacher_$teacher_i/data"]
			else
				m1 = f["$moment/Teacher_$teacher_i/Student_$student_j/data"]
			end
			m2 = f["$moment/Teacher_$teacher_i/Student_$student_j/$rtype"]
				
		else
			@warn "impossible request i=$teacher_i j=$student_j type=$rtype"
			return nothing, nothing
		end
		return read(m1), read(m2)
	end
	
	function get_teacher_student_nrmse(
		f::HDF5.File, 
		teacher_i::Int, 
		student_j::Int,
		moment::String,
		rtype::String,
	)
		m1, m2 = get_teacher_student(f,teacher_i, student_j, moment, rtype)
		if m1 == nothing
			return NaN
		else
			return nRMSE(m1, m2)
		end
	end
end

# ╔═╡ 74021679-2aac-4d7c-8c4a-e326760ceaa1


# ╔═╡ 599bab8b-1ca6-4684-8d3b-1a53c1c7b234
# ╠═╡ disabled = true
#=╠═╡
outpath = joinpath(
	LOAD.MISC, "DeepFakeStats_$(length(FISH))fish_$(base_mod[3:end]).h5"
)
  ╠═╡ =#

# ╔═╡ 384eef48-10cc-4b91-8663-a7c56567c982
#=╠═╡
begin
	max_vv = 1000;
	R = 5;
	
	outfile = h5open(outpath, "cw")
	
	outfile["fish_list"] = FISH
	
	grp_v = create_group(outfile, "<v>")
	grp_vv = create_group(outfile, "<vv>")
	
	for (i,teacher) in enumerate(FISH)
		println("Teacher_$i")
		grp_teach_v = create_group(grp_v, "Teacher_$i")
		grp_teach_vv = create_group(grp_vv, "Teacher_$i")
	
		rbmT,_,_,_,_,_ = load_brainRBM(LOAD.load_wbscRBM("bRBMs", "bRBM_".*teacher.*base_mod))
		dataT = load_data(LOAD.load_dataWBSC(teacher))
		selt_inds_T = max_vv === nothing || max_vv >= size(dataT.spikes, 1) ? nothing : sort(randperm(size(dataT.spikes, 1))[1:max_vv])
	
		mT_data = compute_v_moments(dataT.spikes, selt_inds_T)
		mT_samp = compute_v_moments(repeat_samp_vv(rbmT, rbmT, dataT.spikes; R), selt_inds_T)
		grp_teach_v["data"] = mT_data["<v>"]
		grp_teach_vv["data"] = mT_data["<vv> - <v><v>"]
		grp_teach_v["samp"] = mT_samp["<v>"]
		grp_teach_vv["samp"] = mT_samp["<vv> - <v><v>"]
		
		for (j,student) in enumerate(FISH)
			if student == teacher
				continue
			end
			grp_stud_v = create_group(grp_teach_v, "Student_$j")
			grp_stud_vv = create_group(grp_teach_vv, "Student_$j")
			
			rbmS,_,_,_,_,_ = load_brainRBM(LOAD.load_wbscRBM("biRBMs",  "biRBM_$(student)_FROM_$(teacher)$(base_mod)"))
			dataS = load_data(LOAD.load_dataWBSC(student))
			selt_inds_S = max_vv === nothing || max_vv >= size(dataS.spikes, 1) ? nothing : sort(randperm(size(dataS.spikes, 1))[1:max_vv])
	
			mS_data = compute_v_moments(dataS.spikes, selt_inds_S)
			mS_samp = compute_v_moments(repeat_samp_vv(rbmS, rbmS, dataS.spikes; R), selt_inds_S)
			grp_stud_v["data"] = mS_data["<v>"]
			grp_stud_vv["data"] = mS_data["<vv> - <v><v>"]
			grp_stud_v["samp"] = mS_samp["<v>"]
			grp_stud_vv["samp"] = mS_samp["<vv> - <v><v>"]
	
			mTS = compute_v_moments(repeat_samp_vv(rbmT, rbmS, dataT.spikes; R), selt_inds_S)
			mST = compute_v_moments(repeat_samp_vv(rbmS, rbmT, dataS.spikes; R), selt_inds_T)
			grp_stud_v["TS"] = mTS["<v>"]
			grp_stud_vv["TS"] = mTS["<vv> - <v><v>"]
			grp_stud_v["ST"] = mST["<v>"]
			grp_stud_vv["ST"] = mST["<vv> - <v><v>"]
	
			mTST = compute_v_moments(repeat_samp_vv(rbmS, rbmT, repeat_samp_vv(rbmT, rbmS, dataT.spikes; R), R=1), selt_inds_T)
			mSTS = compute_v_moments(repeat_samp_vv(rbmT, rbmS, repeat_samp_vv(rbmS, rbmT, dataS.spikes; R), R=1), selt_inds_S)
			grp_stud_v["TST"] = mTST["<v>"]
			grp_stud_vv["TST"] = mTST["<vv> - <v><v>"]
			grp_stud_v["STS"] = mSTS["<v>"]
			grp_stud_vv["STS"] = mSTS["<vv> - <v><v>"]
	
		end
	end
	
	close(outfile)
end
  ╠═╡ =#

# ╔═╡ b090783f-a392-4abe-b013-50e4b3a5f7c7
#=╠═╡
begin
	iofile = h5open(outpath, "r+");
	iofile["<v>"]["nrmse_samp"] = [get_teacher_student_nrmse(iofile, i, j, "<v>", "samp") for i=1:length(FISH) , j=1:length(FISH)]
	iofile["<vv>"]["nrmse_samp"] = [get_teacher_student_nrmse(iofile, i, j, "<vv>", "samp") for i=1:length(FISH) , j=1:length(FISH)]
	
	iofile["<v>"]["nrmse_ST"] = Float64.([get_teacher_student_nrmse(iofile, i, j, "<v>", "ST") for i=1:length(FISH) , j=1:length(FISH)])
	iofile["<vv>"]["nrmse_ST"] = Float64.([get_teacher_student_nrmse(iofile, i, j, "<vv>", "ST") for i=1:length(FISH) , j=1:length(FISH)])
	
	iofile["<v>"]["nrmse_TS"] = Float64.([get_teacher_student_nrmse(iofile, i, j, "<v>", "TS") for i=1:length(FISH) , j=1:length(FISH)])
	iofile["<vv>"]["nrmse_TS"] = Float64.([get_teacher_student_nrmse(iofile, i, j, "<vv>", "TS") for i=1:length(FISH) , j=1:length(FISH)])
	
	iofile["<v>"]["nrmse_TST"] = Float64.([get_teacher_student_nrmse(iofile, i, j, "<v>", "TST") for i=1:length(FISH) , j=1:length(FISH)])
	iofile["<vv>"]["nrmse_TST"] = Float64.([get_teacher_student_nrmse(iofile, i, j, "<vv>", "TST") for i=1:length(FISH) , j=1:length(FISH)])
	
	iofile["<v>"]["nrmse_STS"] = Float64.([get_teacher_student_nrmse(iofile, i, j, "<v>", "STS") for i=1:length(FISH) , j=1:length(FISH)])
	iofile["<vv>"]["nrmse_STS"] = Float64.([get_teacher_student_nrmse(iofile, i, j, "<vv>", "STS") for i=1:length(FISH) , j=1:length(FISH)])
	close(iofile)
end;
  ╠═╡ =#

# ╔═╡ d7c6f012-5648-47c9-882d-0e65b0916633


# ╔═╡ 6cc3e240-5d07-4a8c-951d-359d0eef0306
md"""
# 3. Comparing the 2 methods of v → h → v
"""

# ╔═╡ e914c854-3b59-46c8-9ac5-a057318cf179
begin
	function translate_vhv(
		rbm1::R, 
		rbm2::R, 
		spikes::AbstractMatrix; 
		N::Int=2, 
		use_gpu::Bool=false
	) where R <: Union{BrainRBMjulia.RBM, BrainRBMjulia.StandardizedRBM}
		v1 = repeat(spikes, 1, 1, N)
		if use_gpu
			rbm1 = gpu(rbm1)
			rbm2 = gpu(rbm2)
			v1 = gpu(v1)
		end
		h1 = sample_h_from_v(rbm1, v1)
		v2 = sample_v_from_h(rbm2, h1)
		mv2 = mean(v2, dims=3)[:,:,1]
		if use_gpu
			return cpu(mv2)
		else
			return mv2
		end
	end
	
	function cumulant_translate_vhv(
		rbm1::R, 
		rbm2::R, 
		spikes::AbstractMatrix,
		Ns::Vector{Int},
		ref::AbstractMatrix,
	) where R <: Union{BrainRBMjulia.RBM, BrainRBMjulia.StandardizedRBM}
		ns = diff(Ns)
		prepend!(ns, Ns[1])

		Vs = []
		nrmses = Float64[]
		for i in 1:length(ns)
			v1 = repeat(spikes, 1, 1, ns[i])
			h1 = sample_h_from_v(rbm1, v1)
			v2 = sample_v_from_h(rbm2, h1)
			# v2 = mean_v_from_h(rbm2, h1)
			push!(Vs, v2)
			mv2 = mean(cat(Vs..., dims=3), dims=3)[:,:,1]
			push!(nrmses, nRMSE(ref, mv2))
		end
		
		return mean(cat(Vs..., dims=3), dims=3)[:,:,1], nrmses
	end
end

# ╔═╡ 325f4816-cb36-4e3a-83d0-a680bdf8d789
n_samples = [1,5,10,50,100,500,1000,5000,10000];

# ╔═╡ 4764bc6b-71fb-4fe1-a4fb-7eea818a924c
# ╠═╡ disabled = true
#=╠═╡
outpathtransfer = joinpath(
	LOAD.MISC, "DeepFakeTransferMethods_$(length(FISH))fish_$(base_mod[3:end]).h5"
)
  ╠═╡ =#

# ╔═╡ d309bbe8-347d-4332-88a9-d920a13e4e0a
#=╠═╡
begin
	outfiletransfer = h5open(outpathtransfer, "cw")
	outfiletransfer["fish_list"] = FISH
	outfiletransfer["n_samples"] = n_samples
	
	for (i,teacher) in enumerate(FISH)
		grp_teach = create_group(outfiletransfer, "Teacher_$i")
	
		rbmT,_,_,_,_,_ = load_brainRBM(LOAD.load_wbscRBM("bRBMs",  "bRBM_".*teacher.*base_mod))
		dataT = load_data(LOAD.load_dataWBSC(teacher))
	
		inds = shuffle(1:size(dataT.spikes,2))[1:10]
		
		v_old_TT = mean_v_from_h(rbmT, mean_h_from_v(rbmT, dataT.spikes[:,inds]))
		v_new_TT, nrmses_TT = cumulant_translate_vhv(
			rbmT, 
			rbmT, 
			dataT.spikes[:,inds],
			n_samples,
			v_old_TT,
		);
		grp_teach["v_old"] = v_old_TT
		grp_teach["v_new"] = v_new_TT
		grp_teach["nrmse"] = nrmses_TT
		
		
		for (j,student) in enumerate(FISH)
			if student == teacher
				continue
			end
			grp_stud = create_group(grp_teach, "Student_$j")
			
			rbmS,_,_,_,_,_ = load_brainRBM(LOAD.load_wbscRBM("biRBMs",  "biRBM_$(student)_FROM_$(teacher)$(base_mod)"))
			dataS = load_data(LOAD.load_dataWBSC(student))
	
			v_old_TS = mean_v_from_h(rbmS, mean_h_from_v(rbmT, dataT.spikes[:,inds]))
			v_new_TS, nrmses_TS = cumulant_translate_vhv(
				rbmT, 
				rbmS, 
				dataT.spikes[:,inds],
				n_samples,
				v_old_TS,
			);
			grp_stud["v_old"] = v_old_TS
			grp_stud["v_new"] = v_new_TS
			grp_stud["nrmse"] = nrmses_TS
		end
	end
	
	close(outfiletransfer)
end
  ╠═╡ =#

# ╔═╡ d0aba702-a08f-4ab0-99de-ef5a13c022f0


# ╔═╡ f1de6da3-24c0-4745-9e78-9ad5409b5dcd
md"""
# 4. Free Energy
"""

# ╔═╡ f173b501-54c0-48b4-89fe-f54704973670
begin
	function rand_vv(v::AbstractMatrix)
		vout = copy(v)
		for i in 1:size(v,1)
			vout[i,:] .= shuffle(vout[i,:])
		end
		@assert sum(v) == sum(vout)
		return vout
	end
	
	function rand_v(v::AbstractMatrix)
		vout = rand_vv(v)
		for t in 1:size(v,2)
			vout[:,t] .= shuffle(vout[:,t])
		end
		@assert sum(v) == sum(vout)
		return vout
	end
end

# ╔═╡ 391d58b9-0ee4-4895-b9de-994e6e9ceabc
# ╠═╡ disabled = true
#=╠═╡
outpathfreenergy = joinpath(
	LOAD.MISC, "DeepFakeFreeEnergy_$(length(FISH))fish_$(base_mod[3:end]).h5"
)
  ╠═╡ =#

# ╔═╡ e72fb167-7044-4f72-8569-56d3fd8fab2a
#=╠═╡
begin
	outfilefreeenergy = h5open(outpathfreenergy, "cw")
	outfilefreeenergy["fish_list"] = FISH
	
	for (i,teacher) in enumerate(FISH)
		grp_teach = create_group(outfilefreeenergy, "Teacher_$i")
	
		rbmT,_,_,_,_,_ = load_brainRBM(LOAD.load_wbscRBM("bRBMs", "bRBM_".*teacher.*base_mod))
		vT = load_data(LOAD.load_dataWBSC(teacher)).spikes

		F_T = free_energy(rbmT, vT)
		F_T_randvv = free_energy(rbmT, rand_vv(vT))
		F_T_randv = free_energy(rbmT, rand_v(vT))
		F_TT = free_energy(rbmT, sample_v_from_h(rbmT, sample_h_from_v(rbmT, vT)))

		grp_teach["F_T"] = F_T
		grp_teach["F_T_randvv"] = F_T_randvv
		grp_teach["F_T_randv"] = F_T_randv
		grp_teach["F_TT"] = F_TT
		
		
		for (j,student) in enumerate(FISH)
			if student == teacher
				continue
			end
			grp_stud = create_group(grp_teach, "Student_$j")

			stud_path = LOAD.load_wbscRBM("biRBMs", "biRBM_$(student)_FROM_$(teacher)$(base_mod)")
			stud_before_training_path = LOAD.load_wbscRBM("biRBMs_before_training", "biRBM_$(student)_FROM_$(teacher)$(base_mod)")
			rbmS,_,_,_,_,_ = load_brainRBM(stud_path)
			rbmSb,_,_,_,_,_ = load_brainRBM(stud_before_training_path)
			vS = load_data(LOAD.load_dataWBSC(student)).spikes

			F_S = free_energy(rbmS, vS)
			F_S_randvv = free_energy(rbmS, rand_vv(vS))
			F_S_randv = free_energy(rbmS, rand_v(vS))
			F_Sb = free_energy(rbmSb, vS)
			F_SS = free_energy(rbmS, sample_v_from_h(rbmS, sample_h_from_v(rbmS, vS)))
			F_ST = free_energy(rbmT, sample_v_from_h(rbmT, sample_h_from_v(rbmS, vS)))
			F_TS = free_energy(rbmS, sample_v_from_h(rbmS, sample_h_from_v(rbmT, vT)))

			grp_stud["F_S"] = F_S
			grp_stud["F_S_randvv"] = F_S_randvv
			grp_stud["F_S_randv"] = F_S_randv
			grp_stud["F_Sb"] = F_Sb
			grp_stud["F_SS"] = F_SS
			grp_stud["F_ST"] = F_ST
			grp_stud["F_TS"] = F_TS

			
		end
	end
	
	close(outfilefreeenergy)
end
  ╠═╡ =#

# ╔═╡ 562e706e-d7d2-4049-86e4-e97545a9c1e9


# ╔═╡ 43b4bf86-b756-4664-99d2-8f3b5ea4e892
md"""
# 5. Activity
"""

# ╔═╡ 558189d3-6c8a-4198-ac60-d616d233934b
# ╠═╡ disabled = true
#=╠═╡
outpathact = joinpath(
	LOAD.MISC, "DeepFakeActivity_$(length(FISH))fish_$(base_mod[3:end]).h5"
)
  ╠═╡ =#

# ╔═╡ cc55ee40-9f5d-46e2-bed0-78765cfa278d
#=╠═╡
begin
	outfileact = h5open(outpathact, "cw")
	outfileact["fish_list"] = FISH
	for (i,teacher) in enumerate(FISH)
		grp_teach = create_group(outfileact, "Teacher_$i")
	
		rbmT,_,_,_,_,_ = load_brainRBM(LOAD.load_wbscRBM("bRBMs",  "bRBM_".*teacher.*base_mod))
		dataT = load_data(LOAD.load_dataWBSC(teacher))
	
		# T -> h -> T
		Thm = mean_h_from_v(rbmT, dataT.spikes)
		TvmT = mean_v_from_h(rbmT, Thm)
		grp_teach["TT"] = TvmT
		
		for (j,student) in enumerate(FISH)
			if student == teacher
				continue
			end
			grp_stud = create_group(grp_teach, "Student_$j")
			
			rbmS,_,_,_,_,_ = load_brainRBM(LOAD.load_wbscRBM("biRBMs", "biRBM_$(student)_FROM_$(teacher)$(base_mod)"))
			dataS = load_data(LOAD.load_dataWBSC(student))
	
			# T -> h -> S
			TvmS = mean_v_from_h(rbmS, Thm)
			grp_stud["TS"] = TvmS
	
			# S -> h -> S
			Shm = mean_h_from_v(rbmS, dataS.spikes)
			SvmS = mean_v_from_h(rbmS, Shm)
			grp_stud["SS"] = SvmS
	
			# S -> h -> T
			SvmT = mean_v_from_h(rbmT, Shm)
			grp_stud["ST"] = SvmT
	
		end
	end
	close(outfileact)
end
  ╠═╡ =#

# ╔═╡ 824613c8-1b5e-46d0-9ff7-c7f4818a218a


# ╔═╡ dff0ed0a-d56c-4188-9a77-ee0c6c944742


# ╔═╡ 08f4a5d1-f484-4ed7-9d09-ccb7c7b903b2
md"""
# 6. Activity Distance
"""

# ╔═╡ 2626dd2f-715d-46bd-bab9-211a3a64a6e3
# ╠═╡ disabled = true
#=╠═╡
outpathactdist = joinpath(
	LOAD.MISC, "DeepFakeActivityDistance_$(length(FISH))fish_$(base_mod[3:end]).h5"
)
  ╠═╡ =#

# ╔═╡ 8c15e435-4b33-4c01-a4c4-0269b46bf01b
#=╠═╡
begin
	outfileactdist = h5open(outpathactdist, "cw")
	outfileactdist["fish_list"] = FISH
	for (i,teacher) in enumerate(FISH)
		grp_teach = create_group(outfileactdist, "Teacher_$i")
		coordsT = load_data(LOAD.load_dataWBSC(teacher)).coords
	
		TT = h5read(outpathact, "Teacher_$(i)/TT")
	
		TT_inT = KernelInterpolation.gaussian_kernel_interpolate(
			coordsT,
			TT,
			coordsT,
			σ=8.,
		);
		TT_inT[isnan.(TT_inT)] .= 0
	
		
		
		for (j,student) in enumerate(FISH)
			if student == teacher
				continue
			end
			grp_stud = create_group(grp_teach, "Student_$j")
			coordsS = load_data(LOAD.load_dataWBSC(student)).coords
	
			SS = h5read(outpathact, "Teacher_$(i)/Student_$(j)/SS")
			TS = h5read(outpathact, "Teacher_$(i)/Student_$(j)/TS")
			ST = h5read(outpathact, "Teacher_$(i)/Student_$(j)/ST")
			
			SS_inS = KernelInterpolation.gaussian_kernel_interpolate(
				coordsS,
				SS,
				coordsS,
				σ=8.,
			);
			SS_inS[isnan.(SS_inS)] .= 0
	
			ST_inS = KernelInterpolation.gaussian_kernel_interpolate(
				coordsT,
				ST,
				coordsS,
				σ=8.,
			);
			ST_inS[isnan.(ST_inS)] .= 0
	
			TS_inT = KernelInterpolation.gaussian_kernel_interpolate(
				coordsS,
				TS,
				coordsT,
				σ=8.,
			);
			TS_inT[isnan.(TS_inT)] .= 0
	
			ρT = KernelInterpolation.filtered_correlation(TT_inT,TS_inT)
			ρT_shuff = KernelInterpolation.filtered_correlation_shuffled(TT_inT,TS_inT,N=10)
			ρS = KernelInterpolation.filtered_correlation(SS_inS,ST_inS)
			ρS_shuff = KernelInterpolation.filtered_correlation_shuffled(SS_inS,ST_inS,N=10)
	
			grp_stud["rhoT_"] = ρT
			grp_stud["rho_Tshuff"] = ρT_shuff
			grp_stud["rho_S"] = ρS
			grp_stud["rho_Sshuff"] = ρS_shuff
	
		end
	end
	close(outfileactdist)
end
  ╠═╡ =#

# ╔═╡ 2215affc-b38c-488a-aa2b-1048c403b17f


# ╔═╡ 941e8196-a1c5-4c9a-a021-6e28bb1f2c64


# ╔═╡ 9999aac7-85a4-436c-9bdc-824f9c66ab0c


# ╔═╡ Cell order:
# ╠═859e9214-3568-11f0-2108-85a2152204f2
# ╠═6c2c8e99-3c8b-4823-80ca-d3af7049a897
# ╠═8ac34421-413d-4b3a-abd5-97189c6a3e11
# ╠═770fcfab-ebf9-4a6a-bace-7cc2c017d2cc
# ╟─15d89d71-c46d-4a47-abcd-e3d903cca1d3
# ╟─af393427-aaef-4261-ad91-b913ba964ce9
# ╠═8516860a-86dc-474b-b4c7-54fe8aa39835
# ╠═f93a12f9-ab6e-49e6-ac6d-0d32753f3756
# ╠═27183b44-1dca-40b5-a239-5586f6f23b76
# ╟─89249d61-b795-46b8-bc13-644c609f9b60
# ╟─433a9ab5-0739-4abf-8c6b-c152e52e2517
# ╟─eb63e47e-5128-4d81-a00c-9d8c9afbcc82
# ╠═74021679-2aac-4d7c-8c4a-e326760ceaa1
# ╠═599bab8b-1ca6-4684-8d3b-1a53c1c7b234
# ╠═384eef48-10cc-4b91-8663-a7c56567c982
# ╠═b090783f-a392-4abe-b013-50e4b3a5f7c7
# ╠═d7c6f012-5648-47c9-882d-0e65b0916633
# ╟─6cc3e240-5d07-4a8c-951d-359d0eef0306
# ╟─e914c854-3b59-46c8-9ac5-a057318cf179
# ╠═325f4816-cb36-4e3a-83d0-a680bdf8d789
# ╠═4764bc6b-71fb-4fe1-a4fb-7eea818a924c
# ╠═d309bbe8-347d-4332-88a9-d920a13e4e0a
# ╠═d0aba702-a08f-4ab0-99de-ef5a13c022f0
# ╟─f1de6da3-24c0-4745-9e78-9ad5409b5dcd
# ╟─f173b501-54c0-48b4-89fe-f54704973670
# ╠═391d58b9-0ee4-4895-b9de-994e6e9ceabc
# ╠═e72fb167-7044-4f72-8569-56d3fd8fab2a
# ╠═562e706e-d7d2-4049-86e4-e97545a9c1e9
# ╟─43b4bf86-b756-4664-99d2-8f3b5ea4e892
# ╠═558189d3-6c8a-4198-ac60-d616d233934b
# ╠═cc55ee40-9f5d-46e2-bed0-78765cfa278d
# ╠═824613c8-1b5e-46d0-9ff7-c7f4818a218a
# ╠═dff0ed0a-d56c-4188-9a77-ee0c6c944742
# ╟─08f4a5d1-f484-4ed7-9d09-ccb7c7b903b2
# ╠═2626dd2f-715d-46bd-bab9-211a3a64a6e3
# ╠═8c15e435-4b33-4c01-a4c4-0269b46bf01b
# ╠═2215affc-b38c-488a-aa2b-1048c403b17f
# ╠═941e8196-a1c5-4c9a-a021-6e28bb1f2c64
# ╠═9999aac7-85a4-436c-9bdc-824f9c66ab0c
