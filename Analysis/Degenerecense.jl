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

# ╔═╡ 7802e794-4e89-11f0-1f0f-3b4bcf55fe61
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ 3c269232-f97f-4d5c-935e-24c8f17ea648
begin
	# loading modules
	using BrainRBMjulia
	using CairoMakie
	using HDF5
	using FLoops

	using BrainRBMjulia: idplotter, couplingplotter, corrplotter
end

# ╔═╡ a80ee4f7-d82e-4bd7-9891-a3f69d2429bf
using Assignment

# ╔═╡ 2b16e61d-a7fc-4fc8-a76d-3a53832e3967
using LinearAlgebra: tr, diag

# ╔═╡ daf51889-0150-425d-9eb6-d3b50916da11
begin
	using BrainRBMjulia: grid_from_points, project_to_grid, populate_voxels, select_voxels
	function grid(coords::Vector{Matrix{Float64}}, voxsize::Float64, pad=0.1)
		size = [voxsize,voxsize,voxsize]
		orgs, ends = grid_from_points(coords; size, pad)
		Ns = [length(orgs[i]:size[i]:ends[i]) for i in 1:length(orgs)] # number of voxels in each direction

		indss = project_to_grid(coords, size, orgs)
		voxs, capacity = populate_voxels(indss, Ns) # place points in voxels + count nb of point in each voxel
		goods = select_voxels(capacity, thresh=1) # voxel selection criteria

	  	vox_composition = voxs[:, goods]

		return vox_composition
	end
end

# ╔═╡ d9ebbbf2-3b8d-420e-a618-ac56289bccb6
TableOfContents()

# ╔═╡ b8a71f61-8b60-4e9b-9507-91e09a30b4fa
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ 9ebb8023-b3a4-47bd-9822-81ac32707448


# ╔═╡ 7b083a2d-2b46-46d7-9c4d-117bebd203ae
md"""
!!! warn "Warning"
	This notebook performes precomputation for `Fig_Degen.jl`. This computation is VERY long and requires ~50GB of disk space and 60GB of RAM.

	If you still want to compute it, enable the cells bellow.
"""

# ╔═╡ 5d4f9a24-77a6-4c12-91be-fba0c56c5bb4
md"""
# 1. Fish and data
"""

# ╔═╡ 1f7a7a51-f852-418a-a8e9-7dd5f0097ee6
# ╠═╡ disabled = true
#=╠═╡
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];
  ╠═╡ =#

# ╔═╡ 0b17eec8-fbbb-4824-8cd2-0c93fa265602
md"example fish : $(@bind fish Select(FISH, default=FISH[1]))"

# ╔═╡ b8503f2f-88cc-426a-b724-8fe04647acde
base_mod = "*_WBSC_M100_l10.02_l2l10";

# ╔═╡ c86c3f02-6d76-4855-a082-f13871d1c65f


# ╔═╡ 07a0477f-5ff2-488b-b8ce-7065b1bda615
md"""
# Preparing output file
"""

# ╔═╡ 6962ec42-e99b-4e96-9cb3-bf7e440f1bfd
begin
	out_path = joinpath(LOAD.MISC, "Degen_$(fish)_$(base_mod[2:end]).h5")
	out_file = h5open(out_path, "cw");
end

# ╔═╡ 703acf1f-f471-4765-b767-a7d1fc252e24


# ╔═╡ a025c5d4-d35f-4943-96a7-71334efaef51
md"""
# Input RBMs
"""

# ╔═╡ b2429260-ceb1-4d9c-82ab-303381b7e6e8
inpaths = LOAD.load_wbscRBMs("bRBMs", "bRBM_$(fish)*$(base_mod)*");

# ╔═╡ 966c82bd-ea14-4926-baa9-20a245f3668a
inpaths_sorted = inpaths[sortperm(nRMSEs_L4(load_brainRBM_eval(inpaths, ignore="1-nLLH")))];

# ╔═╡ 1e7a79e6-2eb7-44bd-92be-0a82fc046c1d
begin
	rbm_path1 = inpaths_sorted[1]
	rbm_path2 = inpaths_sorted[2]
	rbm_path3 = inpaths_sorted[3]
end

# ╔═╡ 578b00e5-48cf-4c85-80a4-1541d79db9b8
#=╠═╡
begin
	grp_inRBMS = create_group(out_file, "RBMs")
	grp_inRBMS["rbm1"] = rbm_path1
	grp_inRBMS["rbm2"] = rbm_path2
	grp_inRBMS["rbm3"] = rbm_path3
end
  ╠═╡ =#

# ╔═╡ 1ecf2d1a-f875-4d08-b248-d6c7d384d5ab


# ╔═╡ fd3fd583-e6ce-4ee4-8c6c-fd36daf18474
md"""
# Compute Correlations
"""

# ╔═╡ 346b665d-b274-44b0-ad0e-4a4478f63944


# ╔═╡ 523af3fb-ce84-4e70-972b-75676aeb5b9a
md"""
# Compute Couplings
"""

# ╔═╡ 0b653390-764e-4b82-8740-ca26f61f906f


# ╔═╡ 29cbaaa3-0d66-4bc2-ab10-3721310e7a40


# ╔═╡ 009bb0e9-e922-46d7-a1d4-6ce29686382b


# ╔═╡ f0253f9c-83b9-4543-9eb4-7e1195d2299f
md"""
# Voxelized Couplings
"""

# ╔═╡ 9d6a0761-02ef-4364-ba80-5f7a5114e206
coords = load_data(LOAD.load_dataWBSC(fish)).coords

# ╔═╡ 6288eb14-e754-48a4-a00e-9cee5ec33890
vs = Float64.(5:5:50)

# ╔═╡ 35973b3b-108e-4255-98ef-95291d69deca


# ╔═╡ 7d812cd6-a2c6-4fcd-bcd2-e807b615dc47
nrmses = hcat([out_file["Vox_$(v)/nRMSEJ123"][:] for v in vs]...)

# ╔═╡ 1533a7b4-ccc6-42e4-a87d-7515a8182a49
begin
	fig = Figure()
	Axis(fig[1,1], xticks=vs[begin:2:end])
	lines!(vs, nrmses[1,:], color=:black)
	lines!(vs, nrmses[2,:], color=:black)
	lines!(vs, nrmses[3,:], color=:black)
	scatter!(out_file["nRMSE_J123"][:].*0, out_file["nRMSE_J123"][:], color=:black)
	ylims!(0,1)
	fig
end

# ╔═╡ 4d02d530-157e-4fe4-baf7-ee80e892e00d


# ╔═╡ 5ff74912-9c9e-49ed-94a3-ca1e6d273934
md"""
# Hidden Correlations
"""

# ╔═╡ b0da8c61-bbe6-4f3b-9fcf-de6ecdebc963
begin
	rbm1,_,_,_,_,trans1 = load_brainRBM(rbm_path1)
	rbm2,_,_,_,_,trans2 = load_brainRBM(rbm_path2)
	rbm3,_,_,_,_,trans3 = load_brainRBM(rbm_path3)
	trans = [trans1, trans2, trans3]
	ws = [rbm1.w, rbm2.w, rbm3.w]
end

# ╔═╡ dc375860-0578-4d62-bcec-9b24c3390ce2
begin
	Chs = Array{Float32}(undef, length(trans), length(trans), size(trans[1],1), size(trans[1],1));
	Cws = Array{Float32}(undef, length(trans), length(trans), size(trans[1],1), size(trans[1],1));
	for i in 1:length(trans)
		for j in 1:length(trans)
			Chs[i,j,:,:] .= cor(trans[i]', trans[j]', dims=1)
			Cws[i,j,:,:] .= cor(ws[i], ws[j], dims=1)
		end
	end
	out_file["Corr_h"] = Chs
	out_file["Corr_w"] = Cws
end

# ╔═╡ a2c9ac4b-318c-469c-8635-987ac3f09d30


# ╔═╡ 88f7ee4b-8fd5-48f3-bcb9-7d53b35b4c90
corrplotter(Cws[1,1,:,:])

# ╔═╡ 17f67f0e-5033-4de5-9498-034777737438
corrplotter(Cws[2,3,:,:])

# ╔═╡ d207ea52-84ad-4a5a-abda-fc296cac168e


# ╔═╡ 2dcba5d4-7044-49f4-bb78-394c64ae3e4a
C = Cws[1,3,:,:];

# ╔═╡ c772305b-bd84-45b5-b3ce-5f84718b3932
corrplotter(C)

# ╔═╡ e0e79017-6e15-4560-91e4-5138b3f414b9
sol = find_best_assignment(C, true)

# ╔═╡ c43c1b02-11b9-41c2-9693-0c272176e730
sol.row4col

# ╔═╡ 8c259cc8-e388-471e-8d9d-90cef0afc390
sols = find_kbest_assignments(C, 100000, true);

# ╔═╡ b580ca5d-cf2b-41df-9e3d-615b25b36031
corrplotter(C[:,sols[end].row4col])

# ╔═╡ 2b6d1777-dbea-41b1-a2ae-79b2713bfe85
begin
	μ = 16
	fig_trans = Figure()
	Axis(fig_trans[1,1])
	lines!(trans[1][μ,1:500])
	lines!(trans[3][sols[end].row4col[μ],1:500])
	fig_trans
end

# ╔═╡ 473a6426-b98b-4744-b279-3aa76ee02d2b
lines([sol.cost for sol in sols]./size(C,1))

# ╔═╡ 7865aa97-d104-4506-93d8-f6f50fae6e2e
[sol.cost for sol in sols]./size(C,1)

# ╔═╡ de1d6b7d-0bf4-4029-8348-b3215dc67712
idplotter(trans[1], trans[3])

# ╔═╡ 833695f3-f4da-43a0-9850-67d272fb9d56
idplotter(trans[1], trans[3][sols[begin].row4col,:])

# ╔═╡ ffbf54ef-798a-45ea-8355-1e98692bddd1
idplotter(trans[1], trans[3][sols[end].row4col,:])

# ╔═╡ 517d459b-8cc4-4189-af17-9b74c959ba2b


# ╔═╡ e1c5c887-342f-47b1-bf7d-0492257ef903
C[1:100,sols[begin].row4col]

# ╔═╡ e813cdca-4561-4105-be28-751e68c8878b
[C[i,sols[begin].row4col[i]] for i in 1:100]

# ╔═╡ 8d6228d8-a778-4d0e-9e7f-33f58573ef06
hist(diag(C[:, sols[end].row4col]))

# ╔═╡ 5f85ea1c-13f5-4151-9f69-736cea59655a
hist(diag(C[:, sols[end].row4col]), bins=100)

# ╔═╡ dcf630dc-a70a-4d03-af89-3177f02a3495
begin
	match = zeros(Float64, 100,100)
	total_cost = 0
	for sol in sols
		for i in 1:100
			match[sol.row4col[i],i] += sol.cost
			total_cost += sol.cost
		end
	end
	match = match ./ total_cost
end

# ╔═╡ 1da6660b-156f-4248-996d-ed5a8a8c2c51
sol.cost

# ╔═╡ c6b10791-d482-4048-a645-26158fe9e508
heatmap(match[sol.row4col,:])#, colorrange=(0,1), colormap=:inferno)

# ╔═╡ 8e03de85-e8bc-4256-95a7-9cf079d21e1a


# ╔═╡ f4bcd884-6156-49ef-b55e-a1055821389c
out_file

# ╔═╡ 82d98e34-f44c-4d52-8566-425a921d8b1d
close(out_file)

# ╔═╡ 96be5061-2562-4a18-a31b-c21f502c4f1e


# ╔═╡ 0bb32c3c-f17a-4eab-bbad-cbe8b88e637d


# ╔═╡ f796088e-57d8-415a-8cdf-ba8dc6a38e65


# ╔═╡ 6450adf2-d627-4624-b4f4-1388dd7dd72e


# ╔═╡ 49855990-2a45-492d-a5ff-7eeceba4745b
md"""
# 2. Is coupling the same as correlation ?
"""

# ╔═╡ f7a50419-bc8f-42c1-95c1-9a61e9e070f3
md"""
## Compute Couplings
"""

# ╔═╡ a2f708d0-2e6b-4a5d-a1a8-a0821497c1e7
md"""
## Compute Correlations
"""

# ╔═╡ 9e59c60e-169c-4085-91f6-21e0639af203


# ╔═╡ 021d67ec-975b-4e3d-8571-0dca3f66bff4
md"""
## Saving to file
"""

# ╔═╡ 751fc037-ca41-4419-b098-9c481d83722f
#=╠═╡
out_RvsJ_path = joinpath(LOAD.MISC, "RvsJij_$(fish).h5")
  ╠═╡ =#

# ╔═╡ 3ee21eec-7f4a-4823-82a8-eabac29fe537
#=╠═╡
out_RvsJ_file = h5open(out_RvsJ_path, "cw");
  ╠═╡ =#

# ╔═╡ cf9c57e8-5983-413e-87c2-b1d3b31856c0
#=╠═╡
close(out_RvsJ_file)
  ╠═╡ =#

# ╔═╡ a2e0670e-ca21-45b9-a4ef-68c2f7124c8c


# ╔═╡ 9c686cf1-80f7-4e15-9ebd-c6a93d9f6623


# ╔═╡ 8e0cf7bd-70b8-4c8e-953b-9d348d647b35
md"""
# Is Jij stable accross trainings ?
"""

# ╔═╡ c4b117bc-ffe1-4f8f-883b-aecab136377d
#=╠═╡
out_JvsJ_path = joinpath(LOAD.MISC, "JvsJij_$(fish).h5")
  ╠═╡ =#

# ╔═╡ ad6b4c6a-a1fd-4391-a3db-bd49886e294c
#=╠═╡
out_JvsJ_file = h5open(out_JvsJ_path, "cw");
  ╠═╡ =#

# ╔═╡ 72ea9330-0fbf-4930-a0a9-7b46f667b016
#=╠═╡
close(out_JvsJ_file)
  ╠═╡ =#

# ╔═╡ c0a97c84-7d6b-4e2d-8128-38532bc33f9f


# ╔═╡ 9e11204b-8986-4ad6-a9f4-65190f0e4088
md"""
# Jij large structures
"""

# ╔═╡ 755fbefb-4ceb-4221-8817-43b0f4f2c865
#=╠═╡
begin
	coords1 = load_data(LOAD.load_dataWBSC(fish)).coords
	coords2 = load_data(LOAD.load_dataWBSC(fish2)).coords
end;
  ╠═╡ =#

# ╔═╡ 3e63d8de-bcb0-4d01-b76a-0d2dd498d72e
#=╠═╡
begin
	out_JvsJvox_path = joinpath(LOAD.MISC, "JvsJij_vox_$(fish).h5")
	out_JvsJvox_file = h5open(out_JvsJvox_path, "cw");
end
  ╠═╡ =#

# ╔═╡ aad064ed-018b-4cd9-9521-885cda3fa2a7
#=╠═╡
out_JvsJvox_file
  ╠═╡ =#

# ╔═╡ 88e78be7-925b-4c76-9d56-e1b4d574ed28
#=╠═╡
in_JvsJvox_file = h5open(out_JvsJvox_path, "r");
  ╠═╡ =#

# ╔═╡ da7afa77-6916-49a5-b091-88774ab736e4
#=╠═╡
lines([nRMSE(in_JvsJvox_file["Vox_$(v)/J1"][:,:],in_JvsJvox_file["Vox_$(v)/J2"][:,:]) for v in vs])
  ╠═╡ =#

# ╔═╡ eb85b833-482e-4c98-9812-254798732dfd
#=╠═╡
size(in_JvsJvox_file["Vox_5.0/J1"])
  ╠═╡ =#

# ╔═╡ 8f62f233-d1b8-4798-88e4-7f65c106aa9c
#=╠═╡
lines([nRMSE(in_JvsJvox_file["Vox_$(v)/J1"][:,:],in_JvsJvox_file["Vox_$(v)/J3"][:,:]) for v in vs])
  ╠═╡ =#

# ╔═╡ 0996945a-5bbd-4328-8be2-d4527808fc13
#=╠═╡
lines([nRMSE(in_JvsJvox_file["Vox_$(v)/J2"][:,:],in_JvsJvox_file["Vox_$(v)/J3"][:,:]) for v in vs])
  ╠═╡ =#

# ╔═╡ 64e8d7fe-52dd-4819-802b-0fed2658c7b7


# ╔═╡ 386f3958-ea80-4c09-893d-73d126208f34


# ╔═╡ a6176fa3-d821-4535-82f8-a142394f878d
#=╠═╡
G = grid([coords1, coords2], 50.)
  ╠═╡ =#

# ╔═╡ 06e08bf4-3240-46e9-bc7b-d0e42ca510cb
#=╠═╡
NG = size(G,2)
  ╠═╡ =#

# ╔═╡ f549c797-c0d9-4981-ae54-020bdf19337b


# ╔═╡ 4965e057-7183-4bf7-bea5-91f001b16230


# ╔═╡ 1fe93259-fe07-424a-b188-64ad0232282c


# ╔═╡ e56755b0-010b-4637-baf3-a621b091b304


# ╔═╡ dca3529c-6140-4761-9690-1099c61f57bc
md"""
# Tools
"""

# ╔═╡ 354d53bb-a1ef-4122-addc-7e34abbb0c48
begin
	"""
	    upper_flat(A) -> Vector
	
	Return a 1-D vector that stores the strict upper-triangle of the
	*square, symmetric* matrix `A`, row by row, **excluding** the diagonal.
	
	The length of the returned vector is `n*(n-1) ÷ 2`, where `n = size(A,1)`.
	The element order is
	
	"""
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
	
	
	"""
	    get_from_flat(flat, i, j) -> A_ij
	
	Return `A[i,j]` when the strict-upper-triangle of `A`
	was previously stored in `flat = upper_flat(A)`.
	
	Neither the diagonal nor the strictly lower-triangle are stored,
	but because `A` is symmetric you may pass any pair `(i,j)`;
	the function will swap them if `i > j`.
	
	If `flat`’s length is not of the form n*(n-1)/2, a `DomainError` is
	raised.
	"""
	function get_from_flat(flat::AbstractVector, i::Integer, j::Integer)
	    m = length(flat)                     # = n*(n-1)/2
	    # Solve n² – n – 2m = 0  ⇒  n = (1 + √(1+8m))/2
	    n = (1 + isqrt(1 + 8m)) ÷ 2          # integer floor
	
	    n*(n-1) ÷ 2 == m || throw(DomainError(
	        flat, "length(flat) = $m is not a triangular number"))
	
	    @boundscheck begin
	        1 ≤ i ≤ n || throw(BoundsError())
	        1 ≤ j ≤ n || throw(BoundsError())
	        i ≠ j      || error("Diagonal elements are not stored")
	    end
	
	    i, j = (i ≤ j) ? (i, j) : (j, i)     # ensure i < j
	
	    idx = (i-1)*(2n - i) ÷ 2 + (j - i)   # 1-based index, explained earlier
	    @inbounds return flat[idx]
	end

	
end

# ╔═╡ 74a9341e-b6be-488a-82e7-1858c5333f61
"""
    flat_indices(Is, Js, ℓ) -> Vector{Int}

Given two index collections `Is`, `Js` and the length `ℓ` of a vector that
stores `A`’s strict upper-triangle (the output of `upper_flat(A)`),
return the 1-based flat indices corresponding to **all Cartesian pairs**
(i,j) ∈ Is × Js.

For (i,j) below the diagonal we silently swap the indices, because
`upper_flat` only keeps the upper half.  
Passing i == j raises an error (diagonal is not stored).

The output is row-major: i varies slowest, j fastest.
"""
function flat_indices(Is, Js, ℓ::Integer)
    # --- infer original size n from ℓ = n(n-1)/2 ---------------------------
    n = (1 + isqrt(1 + 8ℓ)) ÷ 2
    n*(n-1) ÷ 2 == ℓ || throw(DomainError(ℓ,
        "ℓ is not a triangular number → no valid n"))

    # --- bounds checks -----------------------------------------------------
    @assert all(1 .≤ Is .≤ n) "Some i in Is is out of 1:$n"
    @assert all(1 .≤ Js .≤ n) "Some j in Js is out of 1:$n"

    idxs = Vector{Int}(undef, length(Is)*length(Js))
    k = 1
    @inbounds for i in Is, j in Js
        i == j && error("Diagonal element ($i,$j) is not stored")

        # always map to the stored (i<j) half
        ii, jj = i < j ? (i, j) : (j, i)

        # flat position (same formula used in `get_from_flat`)
        idxs[k] = (ii-1)*(2n - ii) ÷ 2 + (jj - ii)
        k += 1
    end
    return CartesianIndices(idxs)
end


# ╔═╡ 86c89cfe-1f46-44ce-a37d-7a343cdb0dc4
function compute_couplings(rbmpath, spikepath)
	rbm,_,_,_,_,_ = load_brainRBM(rbmpath)
	data = load_data(spikepath)
	
	Jij = coupling_approx(rbm, data.spikes)
	Jij_sym = (Jij .+ Jij') ./ 2

	# return upper_flat(Jij_sym)
	return Jij_sym
end

# ╔═╡ 98d2be2a-e6ae-455e-8876-083895caf7ff
#=╠═╡
begin
	out_file["J1"] = compute_couplings(
		rbm_path1,
		LOAD.load_dataWBSC(fish)
	)
end;
  ╠═╡ =#

# ╔═╡ 1389a0da-e6b5-457e-ae09-2bb9b5d7cb80
#=╠═╡
begin
	out_file["J2"] = compute_couplings(
		rbm_path2,
		LOAD.load_dataWBSC(fish)
	)
end;
  ╠═╡ =#

# ╔═╡ c2180147-9e32-4231-a283-fb1a777abc12
#=╠═╡
begin
	out_file["J3"] = compute_couplings(
		rbm_path3,
		LOAD.load_dataWBSC(fish)
	)
end;
  ╠═╡ =#

# ╔═╡ 2a5c7ed1-789f-4480-9ce5-e58ff88b6151
#=╠═╡
Jij_sym_flat = compute_couplings(
	LOAD.load_wbscRBM("bRBMs", "bRBM_".*fish.*base_mod),
	LOAD.load_dataWBSC(fish)
);
  ╠═╡ =#

# ╔═╡ 547c73de-b87c-4050-b433-af9adc2bb5cd
#=╠═╡
out_RvsJ_file["Jij"] = Jij_sym_flat
  ╠═╡ =#

# ╔═╡ f2a524df-8dbf-4e5a-86a4-f6b80bf7a70e
#=╠═╡
out_file["nRMSE_J123"] = [nRMSE(J1[:,:], J2[:,:]), nRMSE(J1[:,:], J3[:,:]), nRMSE(J2[:,:], J3[:,:])]
  ╠═╡ =#

# ╔═╡ d4ebaae7-1f88-4c00-a2d5-0c6cf30542d4
#=╠═╡
begin	
	J1lodded = J1[:,:]
	J2lodded = J2[:,:]
	J3lodded = J3[:,:]
	for v in vs
		grp_name = "Vox_$(v)"
		if grp_name ∈ keys(out_file)
			grp = out_file[grp_name]
		else
			grp = create_group(out_file, grp_name)
		end
		G = grid([coords], v)
		NG = size(G,2)
		Jmn1 = Matrix{Float32}(undef, NG, NG)
		Jmn2 = Matrix{Float32}(undef, NG, NG)
		Jmn3 = Matrix{Float32}(undef, NG, NG)
		@floop for m in 1:NG
			for n in m:NG
				Jmn1[m,n] = Jmn1[n,m] = mean(J1lodded[G[1,m], G[1,n]])
				Jmn2[m,n] = Jmn2[n,m] = mean(J2lodded[G[1,m], G[1,n]])
				Jmn3[m,n] = Jmn3[n,m] = mean(J3lodded[G[1,m], G[1,n]])
				# Jmn1[m,n] = Jmn1[n,m] = mean([J1[i,j] for i∈G[1,m] , j∈G[1,n]])
				# Jmn2[m,n] = Jmn2[n,m] = mean([J2[i,j] for i∈G[1,m] , j∈G[1,n]])
				# Jmn3[m,n] = Jmn3[n,m] = mean([J3[i,j] for i∈G[1,m] , j∈G[1,n]])
			end
		end
		grp["J1"] = Jmn1
		grp["J2"] = Jmn2
		grp["J3"] = Jmn3
		grp["nRMSEJ123"] = [nRMSE(Jmn1, Jmn2), nRMSE(Jmn1, Jmn3), nRMSE(Jmn2, Jmn3)]
	end
end
  ╠═╡ =#

# ╔═╡ 2ef51e35-18b9-4fb9-b3e3-36724e38a640
#=╠═╡
begin
	Jmn1 = Matrix{Float32}(undef, NG, NG)
	Jmn2 = Matrix{Float32}(undef, NG, NG)
	Jmn3 = Matrix{Float32}(undef, NG, NG)
	for m in 1:NG
		for n in m:NG
			Jmn1[m,n] = Jmn1[n,m] = mean(J1[G[1,m], G[1,n]])
			Jmn2[m,n] = Jmn2[n,m] = mean(J2[G[1,m], G[1,n]])
			Jmn3[m,n] = Jmn3[n,m] = mean(J3[G[2,m], G[2,n]])
		end
	end
end
  ╠═╡ =#

# ╔═╡ bb1749cf-ae51-4d53-8ed2-32dd7e8282e0
#=╠═╡
for (i,path) in enumerate(inpaths_sorted[1:5])
	out_JvsJ_file["J$i"] = compute_couplings(
		path,
		LOAD.load_dataWBSC(fish)
	);
end
  ╠═╡ =#

# ╔═╡ a450c7e3-8eb9-4962-88d9-8557223b46d1
function compute_correlations(spikepath)
	data = load_data(spikepath)
	Rij = Float32.(cor(data.spikes'))
	Rij[isnan.(Rij)] .= 0
	mi = Float32.(mean(data.spikes, dims=2))
	Rmij = [Rij[i,j]*mi[i] for i∈1:length(mi) , j∈1:length(mi)]
	Rmij_sym = (Rmij .+ Rmij') ./ 2
	#return upper_flat(Rmij_sym)
	return Rmij_sym
end

# ╔═╡ 8f504cc0-d64c-47dd-83dd-644c5316977c
#=╠═╡
out_file["corr"] = compute_correlations(LOAD.load_dataWBSC(fish));
  ╠═╡ =#

# ╔═╡ 9a3d436b-ac97-41fc-beea-8a64805ea53e
#=╠═╡
Rmij_sym_flat = compute_correlations(LOAD.load_dataWBSC(fish));
  ╠═╡ =#

# ╔═╡ 93dcc2ad-c75e-4898-87ed-ab67f4a01138
#=╠═╡
out_RvsJ_file["Rmij"] = Rmij_sym_flat
  ╠═╡ =#

# ╔═╡ 882a61cb-270e-463e-86cf-34555e576b15
#=╠═╡
J3 = compute_couplings(
	inpaths_sorted[3],
	LOAD.load_dataWBSC(fish)
);
  ╠═╡ =#

# ╔═╡ 43d63593-308e-441f-a971-f488f36025cb
#=╠═╡
begin
	J1 = out_file["J1"]
	J2 = out_file["J2"]
	J3 = out_file["J3"]
end
  ╠═╡ =#

# ╔═╡ 442a0844-b8bc-4f95-b57a-a368a032e1b6
#=╠═╡
J2 = compute_couplings(
	inpaths_sorted[2],
	LOAD.load_dataWBSC(fish)
);
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═7802e794-4e89-11f0-1f0f-3b4bcf55fe61
# ╠═3c269232-f97f-4d5c-935e-24c8f17ea648
# ╠═d9ebbbf2-3b8d-420e-a618-ac56289bccb6
# ╠═b8a71f61-8b60-4e9b-9507-91e09a30b4fa
# ╠═9ebb8023-b3a4-47bd-9822-81ac32707448
# ╟─7b083a2d-2b46-46d7-9c4d-117bebd203ae
# ╟─5d4f9a24-77a6-4c12-91be-fba0c56c5bb4
# ╠═1f7a7a51-f852-418a-a8e9-7dd5f0097ee6
# ╟─0b17eec8-fbbb-4824-8cd2-0c93fa265602
# ╠═b8503f2f-88cc-426a-b724-8fe04647acde
# ╠═c86c3f02-6d76-4855-a082-f13871d1c65f
# ╟─07a0477f-5ff2-488b-b8ce-7065b1bda615
# ╠═6962ec42-e99b-4e96-9cb3-bf7e440f1bfd
# ╠═703acf1f-f471-4765-b767-a7d1fc252e24
# ╟─a025c5d4-d35f-4943-96a7-71334efaef51
# ╠═b2429260-ceb1-4d9c-82ab-303381b7e6e8
# ╠═966c82bd-ea14-4926-baa9-20a245f3668a
# ╠═1e7a79e6-2eb7-44bd-92be-0a82fc046c1d
# ╠═578b00e5-48cf-4c85-80a4-1541d79db9b8
# ╠═1ecf2d1a-f875-4d08-b248-d6c7d384d5ab
# ╟─fd3fd583-e6ce-4ee4-8c6c-fd36daf18474
# ╠═8f504cc0-d64c-47dd-83dd-644c5316977c
# ╠═346b665d-b274-44b0-ad0e-4a4478f63944
# ╟─523af3fb-ce84-4e70-972b-75676aeb5b9a
# ╠═98d2be2a-e6ae-455e-8876-083895caf7ff
# ╠═1389a0da-e6b5-457e-ae09-2bb9b5d7cb80
# ╠═c2180147-9e32-4231-a283-fb1a777abc12
# ╠═0b653390-764e-4b82-8740-ca26f61f906f
# ╠═43d63593-308e-441f-a971-f488f36025cb
# ╠═29cbaaa3-0d66-4bc2-ab10-3721310e7a40
# ╠═f2a524df-8dbf-4e5a-86a4-f6b80bf7a70e
# ╠═009bb0e9-e922-46d7-a1d4-6ce29686382b
# ╟─f0253f9c-83b9-4543-9eb4-7e1195d2299f
# ╠═9d6a0761-02ef-4364-ba80-5f7a5114e206
# ╠═6288eb14-e754-48a4-a00e-9cee5ec33890
# ╠═d4ebaae7-1f88-4c00-a2d5-0c6cf30542d4
# ╠═35973b3b-108e-4255-98ef-95291d69deca
# ╠═7d812cd6-a2c6-4fcd-bcd2-e807b615dc47
# ╠═1533a7b4-ccc6-42e4-a87d-7515a8182a49
# ╠═4d02d530-157e-4fe4-baf7-ee80e892e00d
# ╟─5ff74912-9c9e-49ed-94a3-ca1e6d273934
# ╠═a80ee4f7-d82e-4bd7-9891-a3f69d2429bf
# ╠═2b16e61d-a7fc-4fc8-a76d-3a53832e3967
# ╠═b0da8c61-bbe6-4f3b-9fcf-de6ecdebc963
# ╠═dc375860-0578-4d62-bcec-9b24c3390ce2
# ╠═a2c9ac4b-318c-469c-8635-987ac3f09d30
# ╠═88f7ee4b-8fd5-48f3-bcb9-7d53b35b4c90
# ╠═17f67f0e-5033-4de5-9498-034777737438
# ╠═d207ea52-84ad-4a5a-abda-fc296cac168e
# ╠═2dcba5d4-7044-49f4-bb78-394c64ae3e4a
# ╠═c772305b-bd84-45b5-b3ce-5f84718b3932
# ╠═e0e79017-6e15-4560-91e4-5138b3f414b9
# ╠═b580ca5d-cf2b-41df-9e3d-615b25b36031
# ╠═c43c1b02-11b9-41c2-9693-0c272176e730
# ╠═8c259cc8-e388-471e-8d9d-90cef0afc390
# ╠═2b6d1777-dbea-41b1-a2ae-79b2713bfe85
# ╠═473a6426-b98b-4744-b279-3aa76ee02d2b
# ╠═7865aa97-d104-4506-93d8-f6f50fae6e2e
# ╠═de1d6b7d-0bf4-4029-8348-b3215dc67712
# ╠═833695f3-f4da-43a0-9850-67d272fb9d56
# ╠═ffbf54ef-798a-45ea-8355-1e98692bddd1
# ╠═517d459b-8cc4-4189-af17-9b74c959ba2b
# ╠═e1c5c887-342f-47b1-bf7d-0492257ef903
# ╠═e813cdca-4561-4105-be28-751e68c8878b
# ╠═8d6228d8-a778-4d0e-9e7f-33f58573ef06
# ╠═5f85ea1c-13f5-4151-9f69-736cea59655a
# ╠═dcf630dc-a70a-4d03-af89-3177f02a3495
# ╠═1da6660b-156f-4248-996d-ed5a8a8c2c51
# ╠═c6b10791-d482-4048-a645-26158fe9e508
# ╠═8e03de85-e8bc-4256-95a7-9cf079d21e1a
# ╠═f4bcd884-6156-49ef-b55e-a1055821389c
# ╠═82d98e34-f44c-4d52-8566-425a921d8b1d
# ╠═96be5061-2562-4a18-a31b-c21f502c4f1e
# ╠═0bb32c3c-f17a-4eab-bbad-cbe8b88e637d
# ╠═f796088e-57d8-415a-8cdf-ba8dc6a38e65
# ╠═6450adf2-d627-4624-b4f4-1388dd7dd72e
# ╟─49855990-2a45-492d-a5ff-7eeceba4745b
# ╟─f7a50419-bc8f-42c1-95c1-9a61e9e070f3
# ╠═2a5c7ed1-789f-4480-9ce5-e58ff88b6151
# ╟─a2f708d0-2e6b-4a5d-a1a8-a0821497c1e7
# ╠═9a3d436b-ac97-41fc-beea-8a64805ea53e
# ╠═9e59c60e-169c-4085-91f6-21e0639af203
# ╟─021d67ec-975b-4e3d-8571-0dca3f66bff4
# ╠═751fc037-ca41-4419-b098-9c481d83722f
# ╠═3ee21eec-7f4a-4823-82a8-eabac29fe537
# ╠═93dcc2ad-c75e-4898-87ed-ab67f4a01138
# ╠═547c73de-b87c-4050-b433-af9adc2bb5cd
# ╠═cf9c57e8-5983-413e-87c2-b1d3b31856c0
# ╠═a2e0670e-ca21-45b9-a4ef-68c2f7124c8c
# ╠═9c686cf1-80f7-4e15-9ebd-c6a93d9f6623
# ╟─8e0cf7bd-70b8-4c8e-953b-9d348d647b35
# ╠═442a0844-b8bc-4f95-b57a-a368a032e1b6
# ╠═882a61cb-270e-463e-86cf-34555e576b15
# ╠═c4b117bc-ffe1-4f8f-883b-aecab136377d
# ╠═ad6b4c6a-a1fd-4391-a3db-bd49886e294c
# ╠═bb1749cf-ae51-4d53-8ed2-32dd7e8282e0
# ╠═72ea9330-0fbf-4930-a0a9-7b46f667b016
# ╠═c0a97c84-7d6b-4e2d-8128-38532bc33f9f
# ╟─9e11204b-8986-4ad6-a9f4-65190f0e4088
# ╠═755fbefb-4ceb-4221-8817-43b0f4f2c865
# ╠═3e63d8de-bcb0-4d01-b76a-0d2dd498d72e
# ╠═aad064ed-018b-4cd9-9521-885cda3fa2a7
# ╠═88e78be7-925b-4c76-9d56-e1b4d574ed28
# ╠═da7afa77-6916-49a5-b091-88774ab736e4
# ╠═eb85b833-482e-4c98-9812-254798732dfd
# ╠═8f62f233-d1b8-4798-88e4-7f65c106aa9c
# ╠═0996945a-5bbd-4328-8be2-d4527808fc13
# ╠═64e8d7fe-52dd-4819-802b-0fed2658c7b7
# ╠═386f3958-ea80-4c09-893d-73d126208f34
# ╠═a6176fa3-d821-4535-82f8-a142394f878d
# ╠═06e08bf4-3240-46e9-bc7b-d0e42ca510cb
# ╠═2ef51e35-18b9-4fb9-b3e3-36724e38a640
# ╠═f549c797-c0d9-4981-ae54-020bdf19337b
# ╠═4965e057-7183-4bf7-bea5-91f001b16230
# ╠═1fe93259-fe07-424a-b188-64ad0232282c
# ╠═e56755b0-010b-4637-baf3-a621b091b304
# ╟─dca3529c-6140-4761-9690-1099c61f57bc
# ╠═354d53bb-a1ef-4122-addc-7e34abbb0c48
# ╠═74a9341e-b6be-488a-82e7-1858c5333f61
# ╠═86c89cfe-1f46-44ce-a37d-7a343cdb0dc4
# ╠═a450c7e3-8eb9-4962-88d9-8557223b46d1
# ╠═daf51889-0150-425d-9eb6-d3b50916da11
