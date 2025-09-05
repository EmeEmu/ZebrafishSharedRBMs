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

# ╔═╡ c3cd9cc4-b7d7-11ef-0315-5d2b6196f61f
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ d51f120c-baf8-4342-8d8d-2581e77b7a5a
begin
	using BrainRBMjulia
	using CairoMakie
	using BrainRBMjulia: idplotter, idplotter!, neuron2dscatter!, cmap_aseismic, quantile_range, dfsize

	using Statistics
	using Distances
	using LinearAlgebra

	include(joinpath(dirname(Base.current_project()), "Misc_Code", "KernelInterpolation.jl"))
end

# ╔═╡ da4e4084-ccf4-46e1-a923-aeeb305f5117
using HDF5

# ╔═╡ f30ff77c-b87b-43bf-acb4-a6ab8024a9de
TableOfContents()

# ╔═╡ 592a59db-71d6-46b6-a831-5df1f101d309
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ bf5f89a3-f195-44d9-808f-6e6101c3fa03


# ╔═╡ e3fb7712-5014-43d3-9139-d6ecf3025490
md"""
# 1. Fish and Training Base
"""

# ╔═╡ 628b1130-e86b-4426-beb6-fe0c2dc01f73
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ f7eead4c-76f4-4f58-abf3-f0e06d3a6f77
base_mod = "*_WBSC_M100_l10.02_l2l10";

# ╔═╡ 55ca2891-d668-4bc2-a376-b1fa6be5a550
md"test teacher : $(@bind testteacher Select(FISH, default=FISH[4]))"

# ╔═╡ 879d712e-9a85-40a0-afe1-3d692251df34
md"test student : $(@bind teststudent Select(FISH, default=FISH[5]))"

# ╔═╡ ce5971e6-3258-44f5-829a-6aca2008ce0e
md"test hidden unit : $(@bind μtest NumberField(0:200, default=56))"

# ╔═╡ 1d717fbd-e0a3-4797-84f3-9719d39df458


# ╔═╡ 03da6d76-9f93-476a-ae93-ddbd389e36dc
ϵ = 1.e-5; # absolute weight cutoff for correlations

# ╔═╡ b4f32a66-f817-440f-91b6-b43e7fef4158
σ = 4; # kernel width in μm

# ╔═╡ 6ca5c1c3-be58-4a89-b4c0-0287871eb888
N = 100; # repetitions for shuffled

# ╔═╡ b7dd2822-efa7-4d6e-aa48-fabac0092bc8
md"""
# 2. Example fish
"""

# ╔═╡ 630ea536-1842-4758-af8f-6b7070864133
md"""
## 2.1. Loading data
"""

# ╔═╡ 988d4bd2-9dd9-4147-ad5a-16006630074f
begin
	rbmT,_,_,_,_,_ = load_brainRBM( # teacher RBM
		LOAD.load_wbscRBM(
			"bRBMs", 
			"bRBM_".*testteacher.*base_mod
		)
	)
	
	student_path = LOAD.load_wbscRBM(
			"biRBMs", 
			"biRBM_$(teststudent)_FROM_$(testteacher)$(base_mod)"
		)
	student_before_training_path = LOAD.load_wbscRBM(
			"biRBMs_before_training", 
			"biRBM_$(teststudent)_FROM_$(testteacher)$(base_mod)"
		)
	rbmTS,_,_,_,_,_ = load_brainRBM(student_path) # bitrained student RBM
	rbmSu,_,_,_,_,_ = load_brainRBM(student_before_training_path)
end;

# ╔═╡ 2ec52b04-1643-43f5-a355-5874a38c04a9
begin
	dataT = load_data(LOAD.load_dataWBSC(testteacher))
	dataS = load_data(LOAD.load_dataWBSC(teststudent))
end;

# ╔═╡ fe6d64d9-80e9-48c9-8395-e2f2ae16c697
l = quantile_range(rbmT.w[abs.(rbmT.w) .> 1.e-2], 0.95)

# ╔═╡ ae64923b-0384-4bd4-84e1-0925e0bb6d93
md"""
## 2.2. Single HU
"""

# ╔═╡ 64c6d7f7-e476-4da6-8a01-293d7cbec1b7
begin
	fig = Figure(size=dfsize().*(2,1.5))
	
	a1 = Axis(fig[1,1], aspect=DataAspect(), title="Teacher")
	neuron2dscatter!(
		dataT.coords[:,1], dataT.coords[:,2],
		rbmT.w[:,μtest],
		cmap=cmap_aseismic(),
		range=(-l,+l),
		radius=4,
		edgewidth=1, edgecolor=(:grey, 0.1),
	)
	
	a2 = Axis(fig[1,2], aspect=DataAspect(), title="Student before training")
	neuron2dscatter!(
		dataS.coords[:,1], dataS.coords[:,2],
		rbmSu.w[:,μtest],
		cmap=cmap_aseismic(),
		range=(-l,+l),
		radius=4,
		edgewidth=1, edgecolor=(:grey, 0.1),
	)
	
	a3 = Axis(fig[1,3], aspect=DataAspect(), title="Student after training")
	neuron2dscatter!(
		dataS.coords[:,1], dataS.coords[:,2],
		rbmTS.w[:,μtest],
		cmap=cmap_aseismic(),
		range=(-l,+l),
		radius=4,
		edgewidth=1, edgecolor=(:grey, 0.1),
	)

	linkaxes!(fig.content...)
	fig
end

# ╔═╡ 351afed0-a4fc-4fea-8222-54f43c6b7c36
md"""
### 2.2.1. Interpolated Weights
"""

# ╔═╡ 7ab42a25-00aa-4b4e-861d-d4722bcd44da
Y_T_S = KernelInterpolation.gaussian_kernel_interpolate(
	dataT.coords,
	rbmT.w[:,μtest],
	dataS.coords,
	σ=σ
);

# ╔═╡ 85083f58-0cc7-48d3-bfb4-db795876bb6a
Y_T_T = KernelInterpolation.gaussian_kernel_interpolate(
	dataT.coords,
	rbmT.w[:,μtest],
	dataT.coords,
	σ=σ
);

# ╔═╡ 665b15a1-1a51-41cf-a807-7e94b8089eb4
begin
	fig2 = Figure(size=dfsize().*(2,1.5))
	
	Axis(fig2[1,1], aspect=DataAspect(), title="Teacher")
	neuron2dscatter!(
		dataT.coords[:,1], dataT.coords[:,2],
		rbmT.w[:,μtest],
		cmap=cmap_aseismic(),
		range=(-l,+l),
		radius=4,
		edgewidth=1, edgecolor=(:grey, 0.1),
	)
	
	Axis(fig2[1,2], aspect=DataAspect(), title="T -> S")
	neuron2dscatter!(
		dataS.coords[:,1], dataS.coords[:,2],
		Y_T_S,
		cmap=cmap_aseismic(),
		range=(-l,+l),
		radius=4,
		edgewidth=1, edgecolor=(:grey, 0.1),
	)
	
	Axis(fig2[1,3], aspect=DataAspect(), title="T -> T")
	neuron2dscatter!(
		dataT.coords[:,1], dataT.coords[:,2],
		Y_T_T,
		cmap=cmap_aseismic(),
		range=(-l,+l),
		radius=4,
		edgewidth=1, edgecolor=(:grey, 0.1),
	)

	linkaxes!(fig2.content...)
	fig2
end

# ╔═╡ 74ad6a2c-2146-4787-9619-c22a9a677767


# ╔═╡ 2354b78c-4aec-4c40-94f7-eccd2070c6f8
md"""
### 2.2.2. Map correlations
"""

# ╔═╡ c1d16162-0dad-4c03-b8f5-8a921f0794ec
ρ_beforeTraining = KernelInterpolation.filtered_correlation(Y_T_S, rbmSu.w[:,μtest]; ϵ)

# ╔═╡ 0962d174-a824-47ae-802a-ddedf4f1c379
ρ_afterTraining = KernelInterpolation.filtered_correlation(Y_T_S, rbmTS.w[:,μtest]; ϵ)

# ╔═╡ edc77a38-d182-4093-aa25-9ef68e009a0a
ρ_selfteacher = KernelInterpolation.filtered_correlation(Y_T_T, rbmT.w[:,μtest]; ϵ)

# ╔═╡ ff6ff8cf-4e4b-4e92-a2bf-364dcee4359a
begin
	fig3 = Figure(size=dfsize().*(2,1.5))
	
	Axis(fig3[1,1], aspect=DataAspect(), title="Before Training", xlabel="Interpolated Teacher", ylabel="Student")
	scatter!(Y_T_S, rbmSu.w[:,μtest], color=(:black, 0.1))
	
	Axis(fig3[1,2], aspect=DataAspect(), title="After Training", xlabel="Interpolated Teacher", ylabel="Student")
	scatter!(Y_T_S, rbmTS.w[:,μtest], color=(:black, 0.1))

	linkaxes!(fig3.content...)
	fig3
end

# ╔═╡ 4e4ae65e-3828-4701-9e32-def65d1b517d


# ╔═╡ e8154de1-9608-44f7-b38c-f395a01b1997
md"""
### 2.2.3. Lost and gained fraction
"""

# ╔═╡ 7257296b-724a-48f2-9d6d-88636ef29955
gained = (abs.(Y_T_S) .< ϵ) .& (abs.(rbmTS.w[:,μtest]) .> ϵ)

# ╔═╡ bd9274a1-b341-4b3b-939f-916928c6ab63
lost = (abs.(Y_T_S) .> ϵ) .& (abs.(rbmTS.w[:,μtest]) .< ϵ)

# ╔═╡ 6e6e8009-0767-4cca-ab4c-f9025867557c
f_gained = sum(gained)/sum((abs.(Y_T_S) .< ϵ))

# ╔═╡ 3ce08f79-fe49-436a-beb0-e9bcd820f66a
f_lost = sum(lost)/sum(abs.(Y_T_S) .> ϵ)

# ╔═╡ ee9cf060-b090-481b-b5a1-68dcc0b6a7b2


# ╔═╡ 24b0e9b9-c27d-4a1f-b54f-0444fcf56cc9
T_on = sum(rbmT.w[:,μtest] .> ϵ)

# ╔═╡ 92c9b2ec-c055-4d1a-b55d-38987fda4d2a
S_on = sum(rbmTS.w[:,μtest] .> ϵ)

# ╔═╡ 771c459e-2b12-407a-a82e-512963f0a63f
f_on = S_on/T_on

# ╔═╡ 8979518b-923a-46f7-9fc2-8e7d48502077


# ╔═╡ 6e60b5e9-ea61-44f1-92e1-3d0d44fe2778


# ╔═╡ a62e8db9-2781-4d0b-9a69-269cac890d0b


# ╔═╡ da9de94d-4b8c-4401-92b5-c75ecd93bd10
md"""
## 2.3. All HU
"""

# ╔═╡ 63990b14-f1fa-49f0-8745-99648088c337
md"""
### 2.3.1. Interpolated Weights
"""

# ╔═╡ 77e24749-f3dc-4876-864e-0af4d2bdc09e
Y_T_Ss = KernelInterpolation.gaussian_kernel_interpolate(
	dataT.coords,
	rbmT.w,
	dataS.coords,
	σ=σ
);

# ╔═╡ 0a01214d-c978-47af-b7d1-7cc456c072a5
Y_T_Ts = KernelInterpolation.gaussian_kernel_interpolate(
	dataT.coords,
	rbmT.w,
	dataT.coords,
	σ=σ
);

# ╔═╡ af402543-97e0-4652-900b-1b5cce9e4d58
md"""
### 2.3.2. Map correlations
"""

# ╔═╡ abea8382-4373-4b2b-8df6-0f7546025b76
ρs_beforeTraining = KernelInterpolation.filtered_correlation(Y_T_Ss, rbmSu.w; ϵ);

# ╔═╡ 63720567-9838-4b89-9c21-1edd02936fb0
ρs_afterTraining = KernelInterpolation.filtered_correlation(Y_T_Ss, rbmTS.w; ϵ);

# ╔═╡ 649d750f-434d-4a1f-a29e-6fa5cc57a796
ρs_selfteacher = KernelInterpolation.filtered_correlation(Y_T_Ts, rbmT.w; ϵ);

# ╔═╡ 3ad52c1b-aaf7-4291-952f-72e5f3cd8435
ρs_shuff = KernelInterpolation.filtered_correlation_shuffled(Y_T_Ss, rbmTS.w; ϵ, N);

# ╔═╡ 19498a21-c95d-40fc-86f1-40e4f1cf126d
begin
	fig22 = Figure()
	Axis(fig22[1,1], xlabel="ρ", ylabel="PDF")
	# hist!(ρs[isfinite.(ρs)], normalization=:pdf)
	# hist!(ρs_shuff[isfinite.(ρs_shuff)], normalization=:pdf, bins=100)
	#density!(ρs[isfinite.(ρs)])
	density!(ρs_shuff[isfinite.(ρs_shuff)], color=:orange, label="shuffled")
	density!(ρs_selfteacher[isfinite.(ρs_selfteacher)], color=:black, label="Teacher self")
	density!(ρs_beforeTraining[isfinite.(ρs_beforeTraining)], color=:red, label="T->S before")
	density!(ρs_afterTraining[isfinite.(ρs_afterTraining)], color=:green, label="T->S after")
	axislegend(position=:lt)
	fig22
end

# ╔═╡ 3156b645-88a8-487e-b411-d0e896661c8e
md"""
### 2.3.3. Pairwise map correlations
"""

# ╔═╡ f0666fa0-aa1c-4ddd-a2e7-a428177b5a34
ρij = KernelInterpolation.filtered_correlation_pairwise(Y_T_Ss, rbmTS.w; ϵ)

# ╔═╡ 0f1573a3-fa40-43e6-97ad-5d0dcdd4690b
function nan_argmax(A::AbstractMatrix)
	aamax = Vector{Int}(undef, size(A,2))
	for j in 1:size(A,2)
		if all(isnan.(ρij[j,:]))
			aamax[j] = -1
		else
			row = copy(A[:,j])
			row[isnan.(row)] .= -1.e9
			aamax[j] = argmax(row)
		end
	end
	return aamax
end

# ╔═╡ a296783d-2bd7-4d1f-a5f6-558c01e8f459
amax_ρij = nan_argmax(ρij)

# ╔═╡ f691b3c4-cfe8-45f2-8624-ebffc8160dd4


# ╔═╡ 240e2cb3-3f12-49e8-a1ad-a6349cf645b5
heatmap(ρij, colormap=:seismic, colorrange=(-1,1))

# ╔═╡ 00895a80-0e57-4d60-b9d8-18f09fee40d4


# ╔═╡ 9f7d47e2-b462-408f-94d4-887457905edc
md"""
### 2.3.4. Lost and gained fraction
"""

# ╔═╡ 011a4562-8ffe-4579-8700-007aae445bfa
gaineds = (abs.(Y_T_Ss) .< ϵ) .& (abs.(rbmTS.w) .> ϵ);

# ╔═╡ be56159c-f0ea-4d52-a8a0-367bb9c7cab2
losts = (abs.(Y_T_Ss) .> ϵ) .& (abs.(rbmTS.w) .< ϵ);

# ╔═╡ b5c82913-7074-40c5-8297-4f5c506c719c
f_losts = sum(losts, dims=1)[1,:] ./ sum((abs.(Y_T_Ss) .> ϵ), dims=1)[1,:]#mean(losts, dims=1)[1,:]

# ╔═╡ 6afeef4c-3cd2-406a-9eec-e04578e27bd1
f_gaineds = sum(gaineds, dims=1)[1,:] ./ sum((abs.(Y_T_Ss) .< ϵ), dims=1)[1,:]#mean(gaineds, dims=1)[1,:]

# ╔═╡ 11b43a52-c31d-4064-bfcc-af96dde208de


# ╔═╡ e806d2db-c3ac-4e24-b715-e629962f657d
T_ons = sum(rbmT.w .> ϵ, dims=1)[1,:]

# ╔═╡ 64e2aaaf-b1d2-46be-81d2-2f92ba9e3ec4
S_ons = sum(rbmTS.w .> ϵ, dims=1)[1,:]

# ╔═╡ 1638fb7d-b2b9-40f3-94a6-1a5439312d9e
idplotter(T_ons, S_ons)

# ╔═╡ 9c922d67-5fe8-4ac1-904b-e1fa44a01878
f_ons = S_ons ./ T_ons

# ╔═╡ 7436ffbc-1f85-4339-a1ce-89c58aae9db5
hist(f_ons, bins=0:0.05:3)

# ╔═╡ 68d22db5-3528-407a-aaf0-e9aecafe728b
size(rbmT.w,1), size(rbmTS.w,1)

# ╔═╡ 68e2cb73-3a56-436a-ad58-5eeb5262b174


# ╔═╡ ac070631-440b-43ed-8f91-487006dedf2e
md"""
# 3. All fish pairs
"""

# ╔═╡ 09ec90be-0bdd-4c1f-b56a-615cae2ff5f2
md"""
!!! warn "Warning"
	The rest of this notebook precomputes distance metrics between all pairs of fish. This computation is a bit long, therefore we provide the reulting file as a DataDeps.

	If you still want to recompute it, enable the cells bellow.
"""

# ╔═╡ 135a0682-7624-4349-9b05-0bef15fff62e
# ╠═╡ disabled = true
#=╠═╡
outpath = joinpath(
	CONV.MISCDATAPATH, "WeightDist_$(length(FISH))fish_$(base_mod[3:end])_sigma$(σ)_epsilon$(ϵ).h5"
)
  ╠═╡ =#

# ╔═╡ d41c15e2-6280-4614-b1bf-c7b884edfa21
# ╠═╡ disabled = true
#=╠═╡
begin
	Ρ_beforeTraining = Array{Float64}(undef, length(FISH), length(FISH), size(rbmT.w,2))
	Ρ_afterTraining = Array{Float64}(undef, length(FISH), length(FISH), size(rbmT.w,2))
	Ρ_afterTraining_pairwise = Array{Float64}(undef, length(FISH), length(FISH), size(rbmT.w,2), size(rbmT.w,2))
	Ρ_shuff = Array{Float64}(undef, length(FISH), length(FISH), size(rbmT.w,2)*N)
	F_lost = Array{Float64}(undef, length(FISH), length(FISH), size(rbmT.w,2))
	F_gained = Array{Float64}(undef, length(FISH), length(FISH), size(rbmT.w,2))
	F_on = Array{Float64}(undef, length(FISH), length(FISH), size(rbmT.w,2))
	
	for (i,teacher) in enumerate(FISH)
		rbmT,_,_,_,_,_ = load_brainRBM( # teacher RBM
			LOAD.load_wbscRBM(
				"bRBMs", 
				"bRBM_".*teacher.*base_mod
			)
		)
		dataT = load_data(LOAD.load_dataWBSC(teacher))
		for (j,student) in enumerate(FISH)
			if student == teacher
				rbmTS = rbmSu = rbmT
				dataS = dataT
			else
				dataS = load_data(LOAD.load_dataWBSC(student))
				student_path = LOAD.load_wbscRBM(
						"biRBMs", 
						"biRBM_$(student)_FROM_$(teacher)$(base_mod)"
					)
				student_before_training_path = LOAD.load_wbscRBM(
						"biRBMs_before_training", 
						"biRBM_$(student)_FROM_$(teacher)$(base_mod)"
					)
				rbmTS,_,_,_,_,_ = load_brainRBM(student_path) # bitrained student RBM	
				rbmSu,_,_,_,_,_ = load_brainRBM(student_before_training_path) # untrained student RBM
			end
	
			# interpolation of weights
			W_TS = KernelInterpolation.gaussian_kernel_interpolate(
				dataT.coords,
				rbmT.w,
				dataS.coords,
				σ=σ
			);
	
			# correlations
			Ρ_beforeTraining[i,j,:] .= KernelInterpolation.filtered_correlation(W_TS, rbmSu.w; ϵ)
			Ρ_afterTraining[i,j,:] = KernelInterpolation.filtered_correlation(W_TS, rbmTS.w; ϵ)
			Ρ_shuff[i,j,:] = KernelInterpolation.filtered_correlation_shuffled(W_TS, rbmTS.w; ϵ, N)
			Ρ_afterTraining_pairwise[i,j,:,:] .= KernelInterpolation.filtered_correlation_pairwise(W_TS, rbmTS.w; ϵ)
	
			# losts and gains
			losts = (abs.(W_TS) .> ϵ) .& (abs.(rbmTS.w) .< ϵ)
			gaineds = (abs.(W_TS) .< ϵ) .& (abs.(rbmTS.w) .> ϵ)
			F_lost[i,j,:] = sum(losts, dims=1)[1,:] ./ sum((abs.(W_TS) .> ϵ), dims=1)[1,:]
			F_gained[i,j,:] = sum(gaineds, dims=1)[1,:] ./ sum((abs.(W_TS) .< ϵ), dims=1)[1,:]
			#F_lost[i,j,:] = mean(losts, dims=1)[1,:]
			#F_gained[i,j,:] = mean(gaineds, dims=1)[1,:]
			T_on = sum(rbmT.w .> ϵ, dims=1)[1,:]
			S_on = sum(rbmTS.w .> ϵ, dims=1)[1,:]
			F_on[i,j,:] = S_on ./ T_on
		end
	end
end
  ╠═╡ =#

# ╔═╡ 5e9938b4-c61a-4ecf-bdb2-deb3e3a8146f
# ╠═╡ disabled = true
#=╠═╡
begin
	outfile = h5open(outpath, "cw")
	outfile["fish_list"] = FISH
	outfile["rho_beforeTraining"] = Ρ_beforeTraining
	outfile["rho_afterTraining"] = Ρ_afterTraining
	outfile["rho_shuff"] = Ρ_shuff
	outfile["f_lost"] = F_lost
	outfile["f_gained"] = F_gained
	outfile["f_on"] = F_on
	outfile["rho_afterTraining_pairwise"] = Ρ_afterTraining_pairwise
	close(outfile)
end
  ╠═╡ =#

# ╔═╡ f3ce43aa-378d-4d65-b336-edce39e4389a


# ╔═╡ Cell order:
# ╠═c3cd9cc4-b7d7-11ef-0315-5d2b6196f61f
# ╠═d51f120c-baf8-4342-8d8d-2581e77b7a5a
# ╠═f30ff77c-b87b-43bf-acb4-a6ab8024a9de
# ╠═592a59db-71d6-46b6-a831-5df1f101d309
# ╠═bf5f89a3-f195-44d9-808f-6e6101c3fa03
# ╟─e3fb7712-5014-43d3-9139-d6ecf3025490
# ╠═628b1130-e86b-4426-beb6-fe0c2dc01f73
# ╠═f7eead4c-76f4-4f58-abf3-f0e06d3a6f77
# ╟─55ca2891-d668-4bc2-a376-b1fa6be5a550
# ╟─879d712e-9a85-40a0-afe1-3d692251df34
# ╟─ce5971e6-3258-44f5-829a-6aca2008ce0e
# ╠═1d717fbd-e0a3-4797-84f3-9719d39df458
# ╠═03da6d76-9f93-476a-ae93-ddbd389e36dc
# ╠═b4f32a66-f817-440f-91b6-b43e7fef4158
# ╠═6ca5c1c3-be58-4a89-b4c0-0287871eb888
# ╟─b7dd2822-efa7-4d6e-aa48-fabac0092bc8
# ╟─630ea536-1842-4758-af8f-6b7070864133
# ╠═988d4bd2-9dd9-4147-ad5a-16006630074f
# ╠═2ec52b04-1643-43f5-a355-5874a38c04a9
# ╠═fe6d64d9-80e9-48c9-8395-e2f2ae16c697
# ╟─ae64923b-0384-4bd4-84e1-0925e0bb6d93
# ╟─64c6d7f7-e476-4da6-8a01-293d7cbec1b7
# ╟─351afed0-a4fc-4fea-8222-54f43c6b7c36
# ╠═7ab42a25-00aa-4b4e-861d-d4722bcd44da
# ╠═85083f58-0cc7-48d3-bfb4-db795876bb6a
# ╟─665b15a1-1a51-41cf-a807-7e94b8089eb4
# ╠═74ad6a2c-2146-4787-9619-c22a9a677767
# ╟─2354b78c-4aec-4c40-94f7-eccd2070c6f8
# ╠═c1d16162-0dad-4c03-b8f5-8a921f0794ec
# ╠═0962d174-a824-47ae-802a-ddedf4f1c379
# ╠═edc77a38-d182-4093-aa25-9ef68e009a0a
# ╠═ff6ff8cf-4e4b-4e92-a2bf-364dcee4359a
# ╠═4e4ae65e-3828-4701-9e32-def65d1b517d
# ╟─e8154de1-9608-44f7-b38c-f395a01b1997
# ╠═7257296b-724a-48f2-9d6d-88636ef29955
# ╠═bd9274a1-b341-4b3b-939f-916928c6ab63
# ╠═6e6e8009-0767-4cca-ab4c-f9025867557c
# ╠═3ce08f79-fe49-436a-beb0-e9bcd820f66a
# ╠═ee9cf060-b090-481b-b5a1-68dcc0b6a7b2
# ╠═24b0e9b9-c27d-4a1f-b54f-0444fcf56cc9
# ╠═92c9b2ec-c055-4d1a-b55d-38987fda4d2a
# ╠═771c459e-2b12-407a-a82e-512963f0a63f
# ╠═8979518b-923a-46f7-9fc2-8e7d48502077
# ╠═6e60b5e9-ea61-44f1-92e1-3d0d44fe2778
# ╠═a62e8db9-2781-4d0b-9a69-269cac890d0b
# ╟─da9de94d-4b8c-4401-92b5-c75ecd93bd10
# ╟─63990b14-f1fa-49f0-8745-99648088c337
# ╠═77e24749-f3dc-4876-864e-0af4d2bdc09e
# ╠═0a01214d-c978-47af-b7d1-7cc456c072a5
# ╟─af402543-97e0-4652-900b-1b5cce9e4d58
# ╠═abea8382-4373-4b2b-8df6-0f7546025b76
# ╠═63720567-9838-4b89-9c21-1edd02936fb0
# ╠═649d750f-434d-4a1f-a29e-6fa5cc57a796
# ╠═3ad52c1b-aaf7-4291-952f-72e5f3cd8435
# ╠═19498a21-c95d-40fc-86f1-40e4f1cf126d
# ╟─3156b645-88a8-487e-b411-d0e896661c8e
# ╠═f0666fa0-aa1c-4ddd-a2e7-a428177b5a34
# ╠═0f1573a3-fa40-43e6-97ad-5d0dcdd4690b
# ╠═a296783d-2bd7-4d1f-a5f6-558c01e8f459
# ╠═f691b3c4-cfe8-45f2-8624-ebffc8160dd4
# ╠═240e2cb3-3f12-49e8-a1ad-a6349cf645b5
# ╠═00895a80-0e57-4d60-b9d8-18f09fee40d4
# ╟─9f7d47e2-b462-408f-94d4-887457905edc
# ╠═011a4562-8ffe-4579-8700-007aae445bfa
# ╠═be56159c-f0ea-4d52-a8a0-367bb9c7cab2
# ╠═b5c82913-7074-40c5-8297-4f5c506c719c
# ╠═6afeef4c-3cd2-406a-9eec-e04578e27bd1
# ╠═11b43a52-c31d-4064-bfcc-af96dde208de
# ╠═e806d2db-c3ac-4e24-b715-e629962f657d
# ╠═64e2aaaf-b1d2-46be-81d2-2f92ba9e3ec4
# ╠═1638fb7d-b2b9-40f3-94a6-1a5439312d9e
# ╠═9c922d67-5fe8-4ac1-904b-e1fa44a01878
# ╠═7436ffbc-1f85-4339-a1ce-89c58aae9db5
# ╠═68d22db5-3528-407a-aaf0-e9aecafe728b
# ╠═68e2cb73-3a56-436a-ad58-5eeb5262b174
# ╟─ac070631-440b-43ed-8f91-487006dedf2e
# ╟─09ec90be-0bdd-4c1f-b56a-615cae2ff5f2
# ╠═da4e4084-ccf4-46e1-a923-aeeb305f5117
# ╠═135a0682-7624-4349-9b05-0bef15fff62e
# ╠═d41c15e2-6280-4614-b1bf-c7b884edfa21
# ╠═5e9938b4-c61a-4ecf-bdb2-deb3e3a8146f
# ╠═f3ce43aa-378d-4d65-b336-edce39e4389a
