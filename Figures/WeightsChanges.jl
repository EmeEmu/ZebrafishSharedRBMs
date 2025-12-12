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

# ╔═╡ 8f45eef0-9852-11f0-36cb-315b468ca212
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	# loading usefules
	using PlutoLinks
	using PlutoUI
end

# ╔═╡ 20dc1a3d-909d-437f-ac61-d9d7a5dc88d0
begin
	using BrainRBMjulia
	using CairoMakie
	using BrainRBMjulia: idplotter, idplotter!, neuron2dscatter!, cmap_aseismic, quantile_range, dfsize

	using Statistics
	using LinearAlgebra
	using Random
	using Arpack
	
	CONV = @ingredients("conventions.jl")
	include(joinpath(dirname(Base.current_project()), "Misc_Code", "fig_saving.jl"))
end

# ╔═╡ 3d77062d-e99c-4aac-8079-818f340682a6
using StatsBase

# ╔═╡ f6cd1e27-d457-41ff-a965-4ee4accf078b
TableOfContents()

# ╔═╡ 0949d5da-cec1-4471-8314-f7c89dbf90f2
set_theme!(CONV.style_publication)

# ╔═╡ 9023c534-cbe6-4c74-9752-1b44f800809c
LOAD = @ingredients(joinpath(dirname(Base.current_project()), "Misc_Code", "loaders.jl"))

# ╔═╡ 7803099c-55d2-4c35-81e2-d04ea4808f03


# ╔═╡ 3291de56-c6cb-4b34-969c-2d7f2fd35110
md"""
# 1. Fish and Training Base
"""

# ╔═╡ 7dc9ad36-59cc-4a59-a77b-e2b9e5b2271a
FISH = [
	"Marianne",
	"Eglantine",
	"Silvestre",
	"Carolinne",
	"Hector",
	"Michel",
];

# ╔═╡ c4e6187b-e1ed-4556-9486-2ae3f25c8447
base_mod = "*_WBSC_M100_l10.02_l2l10";

# ╔═╡ 93363e03-0867-4ab5-ab63-83f9b7ad937e


# ╔═╡ a6e03dda-3384-4f94-a764-17995ef0e6a0
md"test teacher : $(@bind testteacher Select(FISH, default=FISH[3]))"

# ╔═╡ 9effc2a3-e508-4886-93c6-10d2b9989d45
md"test student : $(@bind teststudent Select(FISH, default=FISH[1]))"

# ╔═╡ 5456b909-7ae9-4c89-815b-abee9833eb7c


# ╔═╡ 9c1f046d-4c54-4ac1-96e5-7a1018bfdac2


# ╔═╡ 42d1ab6f-f057-4cb2-b677-1968165b30dd
rbmT,_,_,dsplitT,genT,_  = load_brainRBM(
	LOAD.load_wbscRBM(
		"bRBMs",
		"bRBM_".*testteacher.*base_mod
	)
)

# ╔═╡ cfa2bf1b-3afe-4262-8514-e7113944191b
rbmS,_,_,dsplitS,_,_  = load_brainRBM(
	LOAD.load_wbscRBM(
		"biRBMs",
		"biRBM_$(teststudent)_FROM_$(testteacher)$(base_mod)"
	)
)

# ╔═╡ e97f6aa3-c929-4df6-a3f2-6068781faec4
rbmSb,_,_,_,_,_  = load_brainRBM(
	LOAD.load_wbscRBM(
			"biRBMs_before_training", 
			"biRBM_$(teststudent)_FROM_$(testteacher)$(base_mod)"
		)
)

# ╔═╡ c8aac77f-8a54-42ae-bdc0-b080c2c4f1a8


# ╔═╡ 50720da9-589c-4b5b-b72f-89e536d1cb48
begin
	fig_wdistrib = Figure()
	Axis(fig_wdistrib[1,1], yscale=log10)
	hist!(vec(rbmT.w), bins=1000)
	hist!(vec(rbmS.w), bins=1000)
	hist!(vec(rbmSb.w), bins=1000)
	fig_wdistrib
end

# ╔═╡ 7c691fe0-44d3-42fc-aab3-da22b19b7356


# ╔═╡ f79bc683-ab87-451c-980f-905a9c437805
begin
	qs = LinRange(0,1,100)
	qs_T = quantile(vec(abs.(rbmT.w)), qs)
	qs_S = quantile(vec(abs.(rbmS.w)), qs)
	qs_Sb = quantile(vec(abs.(rbmSb.w)), qs)
	qs
end

# ╔═╡ 5d78d63d-d88e-41c1-b928-687c3f6db23c
begin
	fig_quantiles = Figure()
	Axis(fig_quantiles[1,1], xscale=log10, yscale=log10)
	lines!(qs_T, qs_S, color=:red)
	lines!(qs_T, qs_Sb, color=:green)
	lines!(qs_Sb, qs_S, color=:blue)
	# lines!(qs_T, qs)
	# lines!(qs_S, qs)
	# lines!(qs_Sb, qs)
	fig_quantiles
end

# ╔═╡ aed91fdb-8722-4bc1-b8f2-a265e64bfa6d


# ╔═╡ 0117b477-3ad4-43fe-93d3-d6d0b62b2157


# ╔═╡ 36d1efcb-4754-4f46-846b-0d9c3b881350


# ╔═╡ 7ef7efed-3489-4d6a-b429-5cd7433d9ed6


# ╔═╡ ea456cb4-5d35-4411-a901-24a0648317a4
cov_vS = cor(dsplitS.train')#cov(dsplitS.train')

# ╔═╡ ef3c2c37-f60f-4fc8-86e6-8fc07f38b140
cov_hT = cor(genT.h')#cov(genT.h')

# ╔═╡ 7dc1c101-8ba3-4e68-965d-5fdfb37e5f0e
cov_vS[isnan.(cov_vS)] .= 0

# ╔═╡ 54bdfedd-4c6a-4048-ba0b-c78b2841c803
cov_hT[isnan.(cov_hT)] .= 0

# ╔═╡ 6997b69b-da73-4950-864b-0f7b59157475
# ╠═╡ disabled = true
#=╠═╡
rinds = randperm(size(cov_vS,1))[1:10000]
  ╠═╡ =#

# ╔═╡ 02800a9c-26f7-41e7-b3e0-b905f270428c


# ╔═╡ 61540537-e454-4012-a521-4568aa9097e2
eig_hT = eigs(cov_hT, nev=99)

# ╔═╡ 59e31b14-ea6f-4029-963b-ef098ea0e5a2
eig_vS = eigs(cov_vS, nev=99)

# ╔═╡ e4566ba1-9610-4d2b-878c-a249d8385789
τ = [0.5*eig_vS[1][v] + 0.5*eig_hT[1][h] for v∈1:length(eig_vS[1]), h∈1:length(eig_hT[1])]

# ╔═╡ 733cdce9-63ff-4d80-bcab-d62d5e4937f2
begin
	bins =  LinRange(1.e-9, 1.e2, 50)
	H_vS = fit(Histogram, eig_vS[1], bins)
	H_hT = fit(Histogram, eig_hT[1], bins)
	H_τ = fit(Histogram, vec(τ), bins)
end

# ╔═╡ 223c5cf8-b475-4b7f-bd25-f518dd17d807


# ╔═╡ 02791b88-9cfc-4095-bb10-638435e0735e
begin
	fig = Figure()
	Axis(fig[1,1], yscale=log10, xscale=log10)
	lines!(bins[2:end], (H_vS.weights ./ sum(H_vS.weights)).+1.e-10, color=:red)
	lines!(bins[2:end], (H_τ.weights ./ sum(H_τ.weights)).+1.e-10, color=:green)
	fig
end

# ╔═╡ a20b03e9-bc5a-4329-a38a-728715562304
begin
	Qs = LinRange(0,1,100)
	fig_id = Figure()
	Axis(
		fig_id[1,1], 
		xscale=log10, yscale=log10,
		xlabel="τᵥ", ylabel="(1-λ)τᵥ + λτₕ"
	)
	idplotter!(
		quantile(eig_vS[1], Qs),
		quantile(vec(τ), Qs)
	)
	save(@figpath("id"), fig_id)
	fig_id
end

# ╔═╡ 5f30d9e6-0a46-4045-b271-a4297a59b56a


# ╔═╡ b2a17647-b13a-44cf-bd79-b4557d269237


# ╔═╡ 5a9b3a39-c74e-410d-84dd-2344930a613d


# ╔═╡ bec19a20-2e7f-4be7-bdb3-03b87a2e119a


# ╔═╡ 3c1e8f62-3cf0-4e69-944f-bb99fdcbc94e
begin
	fig_joint = Figure(size=dfsize().*2)
	ax_joint = Axis(
		fig_joint[1,1], 
		xlabel="τᵥ",
		ylabel="τₕ",
		xscale=log10, yscale=log10,
	)
	ax_v = Axis(
		fig_joint[0,1], 
		ylabel="PDF",
		xticklabelsvisible=false,
		xscale=log10,
	)
	ax_h = Axis(
		fig_joint[1,2], 
		xlabel="PDF",
		yticklabelsvisible=false,
		yscale=log10
	)


	lines!(ax_joint, [1.e-1, 70], [1.e-1,70], color=:grey)
	scatter!(ax_joint, eig_vS[1], eig_hT[1], color=(:black, 0.4))
	hist!(ax_v, eig_vS[1], bins=50, normalization=:pdf, color=:black)
	hist!(ax_h, eig_hT[1], bins=50, direction=:x, normalization=:pdf, color=:black)

	linkxaxes!(ax_joint, ax_v)
	linkyaxes!(ax_joint, ax_h)
	ylims!(ax_joint, (1.8e-2,1.5e1))
	ylims!(ax_h, (1.8e-2,1.5e1))
	xlims!(ax_joint, (1.e0,1.e3))
	xlims!(ax_v, (1.e0,1.e3))

	
	save(@figpath("main"), fig_joint)
	fig_joint
end

# ╔═╡ 0ca4bd51-7838-4bae-a1b5-088340d21cb2
bins[2:end]

# ╔═╡ 265a07c9-452f-4366-a811-1f0e18c53874
hist(eig_vS[1])

# ╔═╡ 37403022-4bf0-4357-9133-4ec1842f93f8
hist(0.5.*eig_vS[1] .+ 0.5.*eig_hT[1])

# ╔═╡ 342cdb2b-4a97-4d50-8117-145ee20327b7
lines(H_vS.edges[1][2:end], H_vS.weights)

# ╔═╡ 9da6208b-a293-401e-8f64-8c21fe05a22b
H_vS.edges[1][2:end]

# ╔═╡ 5f41116d-1d54-45bd-a8f7-a401252c962a
H_vS.weights

# ╔═╡ Cell order:
# ╠═8f45eef0-9852-11f0-36cb-315b468ca212
# ╠═20dc1a3d-909d-437f-ac61-d9d7a5dc88d0
# ╠═3d77062d-e99c-4aac-8079-818f340682a6
# ╠═f6cd1e27-d457-41ff-a965-4ee4accf078b
# ╠═0949d5da-cec1-4471-8314-f7c89dbf90f2
# ╠═9023c534-cbe6-4c74-9752-1b44f800809c
# ╠═7803099c-55d2-4c35-81e2-d04ea4808f03
# ╟─3291de56-c6cb-4b34-969c-2d7f2fd35110
# ╠═7dc9ad36-59cc-4a59-a77b-e2b9e5b2271a
# ╠═c4e6187b-e1ed-4556-9486-2ae3f25c8447
# ╠═93363e03-0867-4ab5-ab63-83f9b7ad937e
# ╟─a6e03dda-3384-4f94-a764-17995ef0e6a0
# ╟─9effc2a3-e508-4886-93c6-10d2b9989d45
# ╠═5456b909-7ae9-4c89-815b-abee9833eb7c
# ╠═9c1f046d-4c54-4ac1-96e5-7a1018bfdac2
# ╠═42d1ab6f-f057-4cb2-b677-1968165b30dd
# ╠═cfa2bf1b-3afe-4262-8514-e7113944191b
# ╠═e97f6aa3-c929-4df6-a3f2-6068781faec4
# ╠═c8aac77f-8a54-42ae-bdc0-b080c2c4f1a8
# ╠═50720da9-589c-4b5b-b72f-89e536d1cb48
# ╠═7c691fe0-44d3-42fc-aab3-da22b19b7356
# ╠═f79bc683-ab87-451c-980f-905a9c437805
# ╠═5d78d63d-d88e-41c1-b928-687c3f6db23c
# ╠═aed91fdb-8722-4bc1-b8f2-a265e64bfa6d
# ╠═0117b477-3ad4-43fe-93d3-d6d0b62b2157
# ╠═36d1efcb-4754-4f46-846b-0d9c3b881350
# ╠═7ef7efed-3489-4d6a-b429-5cd7433d9ed6
# ╠═ea456cb4-5d35-4411-a901-24a0648317a4
# ╠═ef3c2c37-f60f-4fc8-86e6-8fc07f38b140
# ╠═7dc1c101-8ba3-4e68-965d-5fdfb37e5f0e
# ╠═54bdfedd-4c6a-4048-ba0b-c78b2841c803
# ╠═6997b69b-da73-4950-864b-0f7b59157475
# ╠═02800a9c-26f7-41e7-b3e0-b905f270428c
# ╠═61540537-e454-4012-a521-4568aa9097e2
# ╠═59e31b14-ea6f-4029-963b-ef098ea0e5a2
# ╠═e4566ba1-9610-4d2b-878c-a249d8385789
# ╠═733cdce9-63ff-4d80-bcab-d62d5e4937f2
# ╠═223c5cf8-b475-4b7f-bd25-f518dd17d807
# ╠═02791b88-9cfc-4095-bb10-638435e0735e
# ╠═a20b03e9-bc5a-4329-a38a-728715562304
# ╠═5f30d9e6-0a46-4045-b271-a4297a59b56a
# ╠═b2a17647-b13a-44cf-bd79-b4557d269237
# ╠═5a9b3a39-c74e-410d-84dd-2344930a613d
# ╠═bec19a20-2e7f-4be7-bdb3-03b87a2e119a
# ╠═3c1e8f62-3cf0-4e69-944f-bb99fdcbc94e
# ╠═0ca4bd51-7838-4bae-a1b5-088340d21cb2
# ╠═265a07c9-452f-4366-a811-1f0e18c53874
# ╠═37403022-4bf0-4357-9133-4ec1842f93f8
# ╠═342cdb2b-4a97-4d50-8117-145ee20327b7
# ╠═9da6208b-a293-401e-8f64-8c21fe05a22b
# ╠═5f41116d-1d54-45bd-a8f7-a401252c962a
