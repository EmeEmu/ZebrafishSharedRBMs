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

# ╔═╡ 115cae49-ac04-472b-9c76-d9dce839fb49
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ dfce61ac-6cc8-4687-83ec-36edf5b6a8e5
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

# ╔═╡ 4f7946ee-c220-4282-99d0-009a28371309
begin
	using Assignment
	using LinearAlgebra: tr, diag
end

# ╔═╡ 153a4546-1d2b-4b06-995c-d6d727fa1487
begin
	using FLoops
	function bin_idx(x::AbstractVector, xmin::Real, xmax::Real, dx::Real)
		xspan = xmax - xmin
		y = (x.-xmin)./xspan
		nbins = xspan./dx
		return Int64.(round.(clamp.(y.*nbins, 1, nbins)))
	end
	function bin2D(x::AbstractVector, y::AbstractVector, xmin::Real, xmax::Real, dx::Real)
		@assert length(x) == length(y)
		xspan = xmax - xmin
		nbins = Int64(round(xspan./dx))
		mat = zeros(Int64, nbins, nbins)
		X = bin_idx(x, xmin, xmax, dx)
		Y = bin_idx(y, xmin, xmax, dx)
		for i in 1:length(X)
			mat[X[i],Y[i]] += 1
		end
		return return mat
	end
	function bin2D(x::HDF5.Dataset, y::HDF5.Dataset, xmin::Real, xmax::Real, dx::Real; chunksize=100)
		@assert size(x) == size(y)
		@assert size(x,1) == size(x,2)
		xspan = xmax - xmin
		nbins = Int64(round(xspan./dx))

		chunks = collect(1:100:size(x,1))
		nchunks = length(chunks)
		push!(chunks, size(x,1))
		
		mat = zeros(Int64, nchunks, nchunks, nbins, nbins)

		@floop for i in 1:length(chunks)-1
			for j in 1:length(chunks)-1
				# println(chunks[i],"->",chunks[i+1],"  |  ", chunks[j],"->",chunks[j+1])
				# println(
				# 	size(x[chunks[i]:chunks[i+1], chunks[j]:chunks[j+1]]), 
				# 	" ", 
				# 	size(y[chunks[i]:chunks[i+1], chunks[j]:chunks[j+1]])
				# )
				a = x[chunks[i]:chunks[i+1], chunks[j]:chunks[j+1]]
				b = y[chunks[i]:chunks[i+1], chunks[j]:chunks[j+1]]
				mat[i,j,:,:] .= bin2D(vec(a), vec(b), xmin, xmax, dx)
			end
		end
		
		return sum(mat, dims=(1,2))[1,1,:,:]
	end
end

# ╔═╡ 0fa74f88-533b-11f0-0a8e-efc5fc5a1f8e
md"""
# Imports + Notebook Preparation
"""

# ╔═╡ 4bdfb278-88ed-47cd-8eb5-9d36f6d31414
TableOfContents()

# ╔═╡ 5f4d5e80-c69e-45cf-9331-7207dafb3b07
set_theme!(CONV.style_publication)

# ╔═╡ bbeb54f3-e53c-4b02-96d5-fa900e52352d
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ fe351289-6593-4d7d-a254-a06003f44b3d


# ╔═╡ a078a9db-92e8-409c-8a86-49f79201cb88
md"""
!!! warn "Warning"
	This notebook requires the precomputations from `Analysis/Degenerecense.jl`. These precomputations are very long and require a lot of memory.
"""

# ╔═╡ f27e7177-5a1e-4b14-bcc8-3e7c6b4ec179


# ╔═╡ f3bd0d3d-326e-42d3-b9bf-51d59420c50f
md"""
# 0. Data
"""

# ╔═╡ 828d8b14-0c02-4039-af82-95a122247a57
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ dfe6df10-d105-474e-b7e7-8a1d812852ae
md"example fish : $(@bind fish Select(FISH, default=FISH[1]))"

# ╔═╡ 22ff3835-23c1-4c4a-9ac3-1aeda255eca9
base_mod = "*_WBSC_M100_l10.02_l2l10";

# ╔═╡ 7d9c48f9-0a5e-4b1f-b8fc-dbd428a4ec9c
in_path = joinpath(LOAD.MISC, "Degen_$(fish)_$(base_mod[2:end]).h5")

# ╔═╡ 40e498e3-f65c-4abd-b05a-84fa0abf7267
in_file = h5open(in_path, "r")

# ╔═╡ 6bd30412-1864-4c27-affb-7f4be6e569e2


# ╔═╡ 86337c06-63fc-43a7-9ec0-9ea22801e5f5
begin
	Jij_id_precompute_path = joinpath(LOAD.MISC, "NOGIT_Jij_id_precompute.h5")
	if isfile(Jij_id_precompute_path)
		println("Jij identity precompute found")
	else
		println("Precomputing Jij identity")
		Jij_id_precompute_file = h5open(Jij_id_precompute_path, "cw")
		Jij_id_precompute_file["H"] = bin2D(
			in_file["J1"],
			in_file["J2"], 
			-0.4, +0.8, 0.01,
		)
		Jij_id_precompute_file["bins"] = collect(-0.4:0.01:+0.8)[begin:end-1]
		close(Jij_id_precompute_file)
	end
	Jij_id_H = h5read(Jij_id_precompute_path, "H")
	Jij_id_D = (Jij_id_H ./ sum(Jij_id_H)) 
	Jij_id_D[Jij_id_D.==0] .= NaN
	Jij_id_bins = h5read(Jij_id_precompute_path, "bins")
end;

# ╔═╡ ad5a78ff-e4af-4532-a60b-30fbf782e239
begin
	Rij_Jij_precompute_path = joinpath(LOAD.MISC, "NOGIT_Rij_Jij_precompute.h5")
	if isfile(Rij_Jij_precompute_path)
		println("Rij vs Jij precompute found")
	else
		println("Precomputing Rij vs Jij identity")
		Rij_Jij_precompute_file = h5open(Rij_Jij_precompute_path, "cw")
		Rij_Jij_precompute_file["H"] = bin2D(
			in_file["corr"],
			in_file["J1"], 
			-0.4, +0.8, 0.01,
		)
		Rij_Jij_precompute_file["bins"] = collect(-0.4:0.01:+0.8)[begin:end-1]
		Rij_Jij_precompute_file["nrmse"] = nRMSE(in_file["corr"][:,:], in_file["J1"][:,:])
		close(Rij_Jij_precompute_file)
	end
	Rij_Jij_H = h5read(Rij_Jij_precompute_path, "H")
	Rij_Jij_D = (Rij_Jij_H ./ sum(Rij_Jij_H)) 
	Rij_Jij_D[Rij_Jij_D.==0] .= NaN
	Rij_Jij_bins = h5read(Rij_Jij_precompute_path, "bins")
	Rij_Jij_nrmse = h5read(Rij_Jij_precompute_path, "nrmse")
end;

# ╔═╡ da1e17fd-61c6-45fb-96b6-cad5241d569e
md"""
# 1. Fig
"""

# ╔═╡ 8066905a-4c4a-437f-86a4-4d6ec5f5d772
begin
	fig = Figure(size=(53, 49).*(2.7,4).*(4/3/0.35))

	g_diag = fig[1,1] = GridLayout()
	g_rj = fig[2,1] = GridLayout()
	g_rjC = fig[2,2] = GridLayout()
	g_jj = fig[2,3] = GridLayout()
	g_jjC = fig[2,4] = GridLayout()
	g_jv = fig[3,1:2] = GridLayout()
	g_jjv = fig[3,3] = GridLayout()
	g_jjvC = fig[3,4] = GridLayout()
	
	
	g_wex = fig[4,1] = GridLayout()
	g_wexC = fig[4,2] = GridLayout()
	g_w = fig[4,3:4] = GridLayout()
	# g_hex = fig[4,1] = GridLayout()
	# g_hexC = fig[4,2] = GridLayout()
	# g_h = fig[4,3:4] = GridLayout()
	
	
	for (label, layout) in zip(
		["A", "B", "C", "D", "E", "F", "G"], 
		[g_diag, g_rj, g_jj, g_jv, g_jjv, g_wex, g_w]
	)
		Label(layout[1, 1, TopLeft()], label,
			fontsize = Makie.current_default_theme().Axis.titlesize.val,
			font = :bold,
			padding = (0, 5, 5, 0),
			halign = :right)
	end

	fig_supp = Figure(size=(53, 49).*(2.7,1).*(4/3/0.35))

	g_hex = fig_supp[1,1] = GridLayout()
	g_hexC = fig_supp[1,2] = GridLayout()
	g_h = fig_supp[1,3:4] = GridLayout()
	
	
	for (label, layout) in zip(
		["A", "B"], 
		[g_hex, g_h]
	)
		Label(layout[1, 1, TopLeft()], label,
			fontsize = Makie.current_default_theme().Axis.titlesize.val,
			font = :bold,
			padding = (0, 5, 5, 0),
			halign = :right)
	end
end

# ╔═╡ f6305738-929b-48db-bb73-713e4a447681
fig

# ╔═╡ a7267733-5e28-4dde-a070-664d8d0035e7
fig_supp

# ╔═╡ 94fb7e3b-c267-4ced-ada5-8b7e00b4ff96
md"""
## Jij nRMSE voxel size
"""

# ╔═╡ ee872b66-1936-4b53-a8cf-098218ceef79
begin
	voxs = sort([parse(Float64, split(k, "_")[2]) for k in keys(in_file) if occursin("Vox", k)])
	nrmses_vox = hcat([in_file["Vox_$(v)/nRMSEJ123"][:] for v in voxs]...)
end;

# ╔═╡ e98bf09b-7441-45f0-99f3-02ff7634e28e
begin
	ex_v_idx = 4
	ex_nrmses_vox = nrmses_vox[1,ex_v_idx]
	ex_v_size = voxs[ex_v_idx]
end

# ╔═╡ e3247ca0-c537-4795-a65b-1a938f9daa49
begin
	vox_Jij_precompute_path = joinpath(LOAD.MISC, "NOGIT_vox_Jij_precompute_$(ex_v_size).h5")
	if isfile(vox_Jij_precompute_path)
		println("vox Jij precompute found")
	else
		println("Precomputing Rij vs Jij identity")
		vox_Jij_precompute_file = h5open(vox_Jij_precompute_path, "cw")
		vox_Jij_precompute_file["H"] = bin2D(
			in_file["Vox_$(ex_v_size)/J1"],
			in_file["Vox_$(ex_v_size)/J2"], 
			-0.1, +0.4, 0.01,
		)
		vox_Jij_precompute_file["bins"] = collect(-0.1:0.01:+0.4)[begin:end-1]
		vox_Jij_precompute_file["nrmse"] = nRMSE(in_file["Vox_$(ex_v_size)/J1"][:,:], in_file["Vox_$(ex_v_size)/J2"][:,:])
		close(vox_Jij_precompute_file)
	end
	vox_Jij_H = h5read(vox_Jij_precompute_path, "H")
	vox_Jij_D = (vox_Jij_H ./ sum(vox_Jij_H)) 
	vox_Jij_D[vox_Jij_D.==0] .= NaN
	vox_Jij_bins = h5read(vox_Jij_precompute_path, "bins")
	vox_Jij_nrmse = h5read(vox_Jij_precompute_path, "nrmse")
end;

# ╔═╡ 7d9f6537-7186-4fbb-9f31-b5c672a34218
nrmses_vox

# ╔═╡ edfdcde6-5570-4f78-8872-6d769413aeea


# ╔═╡ 051af313-d4a1-405b-abf8-e4cd6ff72f3e


# ╔═╡ d2cbfa90-aaaf-47e1-b8c2-adc0967b0f6d
begin
	# fig_Jijid_vox = Figure()
	ax_Jijid_vox = Axis(
		g_jjv[1,1],
		xlabel="Jᵢⱼ training 1",
		ylabel="Jᵢⱼ training 2",
		aspect=1,
	)
	
	# idplotter!(
	# 	ax_Jijid_vox, 
	# 	in_file["Vox_$(ex_v_size)/J1"][:,:], 
	# 	in_file["Vox_$(ex_v_size)/J2"][:,:]
	# )
	lines!(ax_Jijid_vox, [minimum(vox_Jij_bins), maximum(vox_Jij_bins)], [minimum(vox_Jij_bins), maximum(vox_Jij_bins)], color=:grey)

	h_Jijvox = heatmap!(
		ax_Jijid_vox,
		vox_Jij_bins, 
		vox_Jij_bins, 
		vox_Jij_D, 
		colormap=:plasma, 
		colorscale=log10
	)

	text!(
	    ax_Jijid_vox,
	    maximum(vox_Jij_bins),
	    minimum(vox_Jij_bins),
	    text="nRMSE = $(round(vox_Jij_nrmse, sigdigits=3))",
	    align=(:right, :bottom),
	    space=:data,
	  )

	Colorbar(g_jjvC[1,1], h_Jijvox, label="Density")
	# fig_Jijid_vox
end

# ╔═╡ f3cc8cbd-857e-4f2f-ac5a-ff80c564fc89


# ╔═╡ 763606ed-74e4-4f0e-923d-d322298f5303
begin
	# fig_Jijid_id = Figure()
	ax_Jijid_id = Axis(
		g_jj[1,1],
		xlabel="Jᵢⱼ training 1",
		ylabel="Jᵢⱼ training 2",
		aspect=1,
	)
	
	lines!(ax_Jijid_id, [minimum(Jij_id_bins), maximum(Jij_id_bins)], [minimum(Jij_id_bins), maximum(Jij_id_bins)], color=:grey)
	
	h_Jijid_id = heatmap!(
		ax_Jijid_id,
		Jij_id_bins, 
		Jij_id_bins, 
		Jij_id_D, 
		colormap=:plasma, 
		colorscale=log10
	)
	
	text!(
	    ax_Jijid_id,
	    maximum(Jij_id_bins),
	    minimum(Jij_id_bins),
	    text="nRMSE = $(round(in_file["nRMSE_J123"][1], sigdigits=3))",
	    align=(:right, :bottom),
	    space=:data,
	  )

	Colorbar(g_jjC[1,1], h_Jijid_id, label="Density")
	
	# fig_Jijid_id
end

# ╔═╡ ce4f9518-6adc-46e3-8076-c29ecab35edb
begin
	# fig_RJijid = Figure()
	ax_RJijid = Axis(
		g_rj[1,1],
		xlabel="ρ(vᵢ, vⱼ) ½ (⟨vᵢ⟩ + ⟨vⱼ⟩)",
		ylabel="Jᵢⱼ",
		aspect=1,
	)
	
	lines!(ax_RJijid, [minimum(Rij_Jij_bins), maximum(Rij_Jij_bins)], [minimum(Rij_Jij_bins), maximum(Rij_Jij_bins)], color=:grey)
	
	h_RJijid = heatmap!(
		ax_RJijid,
		Rij_Jij_bins, 
		Rij_Jij_bins, 
		Rij_Jij_D, 
		colormap=:plasma, 
		colorscale=log10
	)
	
	text!(
	    ax_RJijid,
	    maximum(Rij_Jij_bins),
	    minimum(Rij_Jij_bins),
	    text="nRMSE = $(round(Rij_Jij_nrmse, sigdigits=3))",
	    align=(:right, :bottom),
	    space=:data,
	  )
	
	Colorbar(g_rjC[1,1], h_RJijid, label="Density")
	
	# fig_RJijid
end

# ╔═╡ aadcb490-4a69-4283-aa1e-7f632e0282d0


# ╔═╡ 7a74e828-7503-4936-93da-eeb63fdac1d3


# ╔═╡ 583fd0d7-cd7e-4208-a0fc-95df450ffa44


# ╔═╡ 9826112c-2025-4d37-904b-1c8bc836e185
md"""
# HU Alignment
"""

# ╔═╡ df7a1ddd-7dca-4cca-a160-da32eb7d0301
md"""
## Weights
"""

# ╔═╡ fc3cbc0d-b259-4c84-8d7e-ab4daf86a56c
Corr_w = in_file["Corr_w"][:,:,:,:];

# ╔═╡ d016c025-8f93-4173-942e-0ddbd582a8b7
begin
	Corr_w_alinged = Array{typeof(Corr_w[1])}(undef, size(Corr_w)...)
	for i in 1:size(Corr_w,1)
		for j in 1:size(Corr_w,2)
			if i == j
				Corr_w_alinged[i,j,:,:] .= Corr_w[i,j,:,:]
			else
				sol = find_best_assignment(Corr_w[i,j,:,:], true)
				Corr_w_alinged[i,j,:,:] .= Corr_w[i,j,:,sol.row4col]
			end
		end
	end
end

# ╔═╡ 4f8b8de3-5500-4fe2-8cc1-66e1a760c5ff
begin
	# fig2 = Figure()
	ax_corrW_ex = Axis(
		g_wex[1,1],#fig2[1,1],
		xlabel="HU μ - training 1",
		ylabel="HU ν - training 2",
		aspect=1
	)
	h_corrW_ex = corrplotter!(
		ax_corrW_ex,
		Corr_w_alinged[1,3,:,:],
	)
	Colorbar(g_wexC[1,1], h_corrW_ex, label="Corr(wiμ,wiν)")
	# fig2
end

# ╔═╡ d87a279e-5924-4a94-a6a0-9f1e3dc8b26a


# ╔═╡ 2eb6f50d-2bf2-4d69-a771-427d9382f001
Cw_diff_diag = vcat([diag(Corr_w_alinged[i,j,:,:]) for i in 1:size(Corr_w_alinged, 1) for j in 1:size(Corr_w_alinged, 1) if i!=j]...);

# ╔═╡ 64b40204-b543-41f8-8ac7-72ef4e81c986


# ╔═╡ 7b313e83-d2e9-4272-861e-06441561fb1e
md"""
## Activity
"""

# ╔═╡ c6f3ab26-3275-41cc-94cd-95fd14132c3b
Corr_h = in_file["Corr_h"][:,:,:,:];

# ╔═╡ bcfba4c4-a56b-4468-8c2e-ac986fd5c157
begin
	Corr_h_alinged = Array{typeof(Corr_h[1])}(undef, size(Corr_h)...)
	for i in 1:size(Corr_h,1)
		for j in 1:size(Corr_h,2)
			if i == j
				Corr_h_alinged[i,j,:,:] .= Corr_h[i,j,:,:]
			else
				sol = find_best_assignment(Corr_h[i,j,:,:], true)
				Corr_h_alinged[i,j,:,:] .= Corr_h[i,j,:,sol.row4col]
			end
		end
	end
end

# ╔═╡ 89bc5566-db42-42a9-aec6-f9d3887829fc
begin
	# fig3 = Figure()
	ax_corrH_ex = Axis(
		g_hex[1,1],#fig3[1,1],
		xlabel="HU μ - training 1",
		ylabel="HU ν - training 2",
		aspect=1,
	)
	h_corrH_ex = corrplotter!(
		ax_corrH_ex,
		Corr_h_alinged[1,3,:,:],
	)
	Colorbar(g_hexC[1,1], h_corrH_ex, label="Corr(hμ(t),hν(t))")
	# fig3
end

# ╔═╡ a04812d4-2ca0-410e-8228-c3c1e51bbd50
Ch_diff_diag = vcat([diag(Corr_h_alinged[i,j,:,:]) for i in 1:size(Corr_h_alinged, 1) for j in 1:size(Corr_h_alinged, 1) if i!=j]...);

# ╔═╡ 4f6c213b-1b01-4cce-9d64-2fa6aef459c7


# ╔═╡ ab94dd59-d7c2-45e1-938d-06b35228007a
# ╠═╡ disabled = true
#=╠═╡
fig_algn
  ╠═╡ =#

# ╔═╡ 96d0184b-2934-4ffb-90d7-3e0b7de484f3


# ╔═╡ c543caf4-a0f5-4683-93a8-6b95d1e7b676
md"""
# Adjustments
"""

# ╔═╡ 6a6dbd34-9e58-44f7-8d2b-3745f2794166
all_axes = [ax for ax in fig.content if typeof(ax)==Axis];

# ╔═╡ fbc281f7-a241-4930-8797-7e36faaf1910
all_axes_supp = [ax for ax in fig_supp.content if typeof(ax)==Axis];

# ╔═╡ 1f753260-2853-4a79-9c5a-ffa45de4786c
for ax in all_axes
	ax.alignmode = Mixed(left=0)
end

# ╔═╡ e5a0d431-ee3e-4bb2-bafc-8dd9259a76db
for ax in all_axes_supp
	ax.alignmode = Mixed(left=0)
end

# ╔═╡ 3fab5259-9f38-48ed-b1c1-1f93d774b211
begin
	colgap!(fig.layout, 1, 5)
	colgap!(fig.layout, 3, 5)
end

# ╔═╡ 664101ce-c5f0-459d-b6dc-4c4d599d85cd
begin
	colgap!(fig_supp.layout, 1, 5)
	colgap!(fig_supp.layout, 3, 5)
	colsize!(fig_supp.layout, 3, 17)
end

# ╔═╡ 679c40a8-5f2d-40b1-b92f-213c87533eab
fig

# ╔═╡ 5656d3bc-7a6d-443a-a129-5c855a2a3c48
# ╠═╡ disabled = true
#=╠═╡
save(@figpath("rbm_degen"), fig)
  ╠═╡ =#

# ╔═╡ 0306f3f3-4917-42ec-8156-080290bc5486
fig_supp

# ╔═╡ ebc4bdaa-f23d-4ea8-8027-d0983a0f8778
# ╠═╡ disabled = true
#=╠═╡
save(@figpath("rbm_degen_supp"), fig_supp)
  ╠═╡ =#

# ╔═╡ f217f2d8-149d-4426-bd14-014cbc851ed5


# ╔═╡ 91a89157-8395-4443-9f01-c7061e19e542


# ╔═╡ 150d85e3-a17d-4d30-89cd-04d21108855f
md"""
# Tools
"""

# ╔═╡ 1cc5cb82-6a57-4130-85b4-841b3628817f
function upper_flat(A::AbstractMatrix)
	    n = size(A,1)
	    @assert n == size(A,2) "A must be square"
	    flat = Vector{eltype(A)}(undef, n*(n-1) ÷ 2)
	
	    k = 1
	    @inbounds for i in 1:n-1, j in i+1:n
	        flat[k] = A[i,j]
	        k += 1
	    end
	    return flat
	end

# ╔═╡ 7f30dd60-7bee-4987-b8f4-2cb94f657cc9
Cw_same_triu = vcat([upper_flat(Corr_w_alinged[i,i,:,:]) for i in 1:size(Corr_w_alinged, 1)]...);

# ╔═╡ d69f3048-9827-403d-833f-fc940edadd69
Cw_diff_triu = vcat([upper_flat(Corr_w_alinged[i,j,:,:]) for i in 1:size(Corr_w_alinged, 1) for j in 1:size(Corr_w_alinged, 1) if i!=j]...);

# ╔═╡ bc2de9a0-19da-4689-bf79-fe8839920465
begin
	# fig = Figure()
	ax_corrW_hist = Axis(
		g_w[1,1],#fig[1,1], 
		yscale=log10,
		ylabel="Density",
		xlabel="Corr(w_iμ^1, w_iμ^2)",
	)
	stephist!(
		ax_corrW_hist,
		vcat(Cw_same_triu, Cw_diff_triu),
		bins=0:0.05:1,
		normalization=:pdf,
		color=:grey,
	)
	stephist!(
		ax_corrW_hist,
		Cw_diff_diag,
		normalization=:pdf,
		bins=0:0.05:1,
		color=:green,
	)
	xlims!(ax_corrW_hist, 0,1.05)
	ylims!(ax_corrW_hist, 1.e-3, 2.e2)

	# lines!([0.9,1],[5.e1, 5.e1], color=:orange)
	# lines!([0,0.1],[5.e1, 5.e1], color=:orange)
	bracket!(
		ax_corrW_hist,
		0.9, 3.e1, 1, 3.e1,
		style=:square, width=5,
		color=:green,
		text="$(Int(round(mean(Cw_diff_diag .> 0.9) .* 100; digits=0)))%", textcolor=:green,
	)
	bracket!(
		ax_corrW_hist,
		0, 3.e1, 0.2, 3.e1,
		style=:square, width=5,
		color=:green,
		text="$(Int(round(mean(Cw_diff_diag .< 0.2) .* 100; digits=0)))%", textcolor=:green,
	)
	
	# fig
end

# ╔═╡ 7fc9d710-62e6-4a75-9d06-f312bbd15b58
Ch_same_triu = vcat([upper_flat(Corr_h_alinged[i,i,:,:]) for i in 1:size(Corr_h_alinged, 1)]...);

# ╔═╡ 82a310dc-3b40-4bc5-baa5-9fef3ec32993
Ch_diff_triu = vcat([upper_flat(Corr_h_alinged[i,j,:,:]) for i in 1:size(Corr_h_alinged, 1) for j in 1:size(Corr_h_alinged, 1) if i!=j]...);

# ╔═╡ 824ec178-2349-4f9d-a98d-9fa16c2b8dd2
begin
	# fig4 = Figure()
	ax_corrH_hist = Axis(
		g_h[1,1],#fig4[1,1], 
		yscale=log10,
		ylabel="Density",
		xlabel="Corr(h_μ^1, h_μ^2)",
	)
	stephist!(
		ax_corrH_hist,
		vcat(Ch_same_triu, Ch_diff_triu),
		bins=-0:0.05:1,
		normalization=:pdf,
		color=:grey,
	)
	stephist!(
		ax_corrH_hist,
		Ch_diff_diag,
		normalization=:pdf,
		bins=-1:0.05:1,
		color=:green,
	)
	xlims!(ax_corrH_hist, 0,1.05)
	ylims!(ax_corrH_hist, 1.e-3, 3.e2)

	bracket!(
		ax_corrH_hist,
		0.9, 3.e1, 1, 3.e1,
		style=:square, width=5,
		color=:green,
		text="$(Int(round(mean(Ch_diff_diag .> 0.9) .* 100; digits=0)))%", textcolor=:green,
	)
	bracket!(
		ax_corrH_hist,
		0, 3.e1, 0.2, 3.e1,
		style=:square, width=5,
		color=:green,
		text="$(Int(round(mean(Ch_diff_diag .< 0.2) .* 100; digits=0)))%", textcolor=:green,
	)
	
	# fig4
end

# ╔═╡ 30f52065-38a9-4354-b7c8-3925b8a21094
function axis_intercepts_full!(ax::Makie.Axis, x::Real, y::Real; kwargs...)
	# make sure the limits are up to date
	autolimits!(ax)

	xmin, xmax = ax.xaxis.attributes.limits[]
	ymin, ymax = ax.yaxis.attributes.limits[]

	h = lines!(ax, [xmin, x], fill(y, 2); kwargs...)  # horizontal line
	v = lines!(ax, fill(x, 2), [ymin, y]; kwargs...)  # vertical   line

	xlims!(ax, xmin, xmax)
	ylims!(ax, ymin, ymax)
	return h, v
end

# ╔═╡ 2a149797-2aaa-439a-8f11-10372b6dbcf2
begin
	# fig_nrmsevox = Figure()
	ax_nrmsevox = Axis(
		g_jv[1,1], 
		xticks=voxs[begin:2:end],
		xlabel="Voxel size (μm)",
		ylabel="Cross training nRMSE Jᵢⱼ"
	)
	lines!(ax_nrmsevox, voxs, nrmses_vox[1,:], color=:black)
	lines!(ax_nrmsevox, voxs, nrmses_vox[2,:], color=:black)
	lines!(ax_nrmsevox, voxs, nrmses_vox[3,:], color=:black)
	scatter!(ax_nrmsevox, in_file["nRMSE_J123"][:].*0, in_file["nRMSE_J123"][:], color=:black)
	
	ylims!(ax_nrmsevox, 0,1)
	
	axis_intercepts_full!(
		ax_nrmsevox,
		ex_v_size, ex_nrmses_vox, 
		color=(:grey, 0.5), linestyle=:dash,
	)

	ylims!(ax_nrmsevox, 0,1)
	
	# fig_nrmsevox
end

# ╔═╡ Cell order:
# ╟─0fa74f88-533b-11f0-0a8e-efc5fc5a1f8e
# ╠═115cae49-ac04-472b-9c76-d9dce839fb49
# ╠═dfce61ac-6cc8-4687-83ec-36edf5b6a8e5
# ╠═4bdfb278-88ed-47cd-8eb5-9d36f6d31414
# ╠═5f4d5e80-c69e-45cf-9331-7207dafb3b07
# ╠═bbeb54f3-e53c-4b02-96d5-fa900e52352d
# ╠═fe351289-6593-4d7d-a254-a06003f44b3d
# ╠═a078a9db-92e8-409c-8a86-49f79201cb88
# ╠═f27e7177-5a1e-4b14-bcc8-3e7c6b4ec179
# ╟─f3bd0d3d-326e-42d3-b9bf-51d59420c50f
# ╠═828d8b14-0c02-4039-af82-95a122247a57
# ╟─dfe6df10-d105-474e-b7e7-8a1d812852ae
# ╠═22ff3835-23c1-4c4a-9ac3-1aeda255eca9
# ╠═7d9c48f9-0a5e-4b1f-b8fc-dbd428a4ec9c
# ╠═40e498e3-f65c-4abd-b05a-84fa0abf7267
# ╠═e98bf09b-7441-45f0-99f3-02ff7634e28e
# ╠═6bd30412-1864-4c27-affb-7f4be6e569e2
# ╠═86337c06-63fc-43a7-9ec0-9ea22801e5f5
# ╠═ad5a78ff-e4af-4532-a60b-30fbf782e239
# ╠═e3247ca0-c537-4795-a65b-1a938f9daa49
# ╟─da1e17fd-61c6-45fb-96b6-cad5241d569e
# ╠═8066905a-4c4a-437f-86a4-4d6ec5f5d772
# ╠═f6305738-929b-48db-bb73-713e4a447681
# ╠═a7267733-5e28-4dde-a070-664d8d0035e7
# ╟─94fb7e3b-c267-4ced-ada5-8b7e00b4ff96
# ╠═ee872b66-1936-4b53-a8cf-098218ceef79
# ╠═7d9f6537-7186-4fbb-9f31-b5c672a34218
# ╠═2a149797-2aaa-439a-8f11-10372b6dbcf2
# ╠═edfdcde6-5570-4f78-8872-6d769413aeea
# ╠═051af313-d4a1-405b-abf8-e4cd6ff72f3e
# ╠═d2cbfa90-aaaf-47e1-b8c2-adc0967b0f6d
# ╠═f3cc8cbd-857e-4f2f-ac5a-ff80c564fc89
# ╠═763606ed-74e4-4f0e-923d-d322298f5303
# ╠═ce4f9518-6adc-46e3-8076-c29ecab35edb
# ╠═aadcb490-4a69-4283-aa1e-7f632e0282d0
# ╠═7a74e828-7503-4936-93da-eeb63fdac1d3
# ╠═583fd0d7-cd7e-4208-a0fc-95df450ffa44
# ╟─9826112c-2025-4d37-904b-1c8bc836e185
# ╠═4f7946ee-c220-4282-99d0-009a28371309
# ╟─df7a1ddd-7dca-4cca-a160-da32eb7d0301
# ╠═fc3cbc0d-b259-4c84-8d7e-ab4daf86a56c
# ╠═d016c025-8f93-4173-942e-0ddbd582a8b7
# ╠═4f8b8de3-5500-4fe2-8cc1-66e1a760c5ff
# ╠═d87a279e-5924-4a94-a6a0-9f1e3dc8b26a
# ╠═7f30dd60-7bee-4987-b8f4-2cb94f657cc9
# ╠═d69f3048-9827-403d-833f-fc940edadd69
# ╠═2eb6f50d-2bf2-4d69-a771-427d9382f001
# ╠═bc2de9a0-19da-4689-bf79-fe8839920465
# ╠═64b40204-b543-41f8-8ac7-72ef4e81c986
# ╟─7b313e83-d2e9-4272-861e-06441561fb1e
# ╠═c6f3ab26-3275-41cc-94cd-95fd14132c3b
# ╠═bcfba4c4-a56b-4468-8c2e-ac986fd5c157
# ╠═89bc5566-db42-42a9-aec6-f9d3887829fc
# ╠═7fc9d710-62e6-4a75-9d06-f312bbd15b58
# ╠═82a310dc-3b40-4bc5-baa5-9fef3ec32993
# ╠═a04812d4-2ca0-410e-8228-c3c1e51bbd50
# ╠═824ec178-2349-4f9d-a98d-9fa16c2b8dd2
# ╠═4f6c213b-1b01-4cce-9d64-2fa6aef459c7
# ╠═ab94dd59-d7c2-45e1-938d-06b35228007a
# ╠═96d0184b-2934-4ffb-90d7-3e0b7de484f3
# ╟─c543caf4-a0f5-4683-93a8-6b95d1e7b676
# ╠═6a6dbd34-9e58-44f7-8d2b-3745f2794166
# ╠═fbc281f7-a241-4930-8797-7e36faaf1910
# ╠═1f753260-2853-4a79-9c5a-ffa45de4786c
# ╠═e5a0d431-ee3e-4bb2-bafc-8dd9259a76db
# ╠═3fab5259-9f38-48ed-b1c1-1f93d774b211
# ╠═664101ce-c5f0-459d-b6dc-4c4d599d85cd
# ╠═679c40a8-5f2d-40b1-b92f-213c87533eab
# ╠═5656d3bc-7a6d-443a-a129-5c855a2a3c48
# ╠═0306f3f3-4917-42ec-8156-080290bc5486
# ╠═ebc4bdaa-f23d-4ea8-8027-d0983a0f8778
# ╠═f217f2d8-149d-4426-bd14-014cbc851ed5
# ╠═91a89157-8395-4443-9f01-c7061e19e542
# ╟─150d85e3-a17d-4d30-89cd-04d21108855f
# ╠═153a4546-1d2b-4b06-995c-d6d727fa1487
# ╠═1cc5cb82-6a57-4130-85b4-841b3628817f
# ╠═30f52065-38a9-4354-b7c8-3925b8a21094
