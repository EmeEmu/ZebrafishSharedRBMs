module KernelInterpolation

using NearestNeighbors
using Distances
using LinearAlgebra       # dot, mul!
using Statistics          # cor
using Random

export gaussian_kernel_interpolate, filtered_correlation

# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

"""Return a KD‑tree with point coordinates as columns."""
_build_tree(X::AbstractMatrix) = KDTree(permutedims(X))

"""Return `Y` reshaped to a `n×m` matrix (m=1 for vectors)."""
_as_matrix(Y::AbstractVector) = reshape(Y, :, 1)
_as_matrix(Y::AbstractMatrix) = Y

# ---------------------------------------------------------------------
# Gaussian kernel interpolation — vector observations (existing method)
# ---------------------------------------------------------------------
function gaussian_kernel_interpolate(X1::AbstractMatrix{<:Real},
  Y1::AbstractVector{<:Real},
  X2::AbstractMatrix{<:Real};
  σ::Real,
  k::Real=3.0)
  Ymat = _as_matrix(Y1)                                # make (n₁×1)
  Y2mat = _kernel_interp_matrix(X1, Ymat, X2; σ=σ, k=k) # (n₂×1)
  return vec(Y2mat)                                    # return Vector
end

# ---------------------------------------------------------------------
# Gaussian kernel interpolation — matrix observations (new method)
# ---------------------------------------------------------------------
function gaussian_kernel_interpolate(X1::AbstractMatrix{<:Real},
  Y1::AbstractMatrix{<:Real},
  X2::AbstractMatrix{<:Real};
  σ::Real,
  k::Real=3.0)
  return _kernel_interp_matrix(X1, Y1, X2; σ=σ, k=k)   # (n₂×m)
end

# === shared implementation (works on matrices) =========================
function _kernel_interp_matrix(X1::AbstractMatrix{<:Real},
  Y1::AbstractMatrix{<:Real},
  X2::AbstractMatrix{<:Real};
  σ::Real, k::Real)
  @assert size(X1, 1) == size(Y1, 1) "rows(X1) must equal rows(Y1)"
  @assert size(X1, 2) == 3 && size(X2, 2) == 3 "coordinates must be 3‑D"
  @assert σ > 0 && k > 0 "σ and k must be positive"

  tree = _build_tree(X1)
  n₂ = size(X2, 1)
  m = size(Y1, 2)                   # #observations per point
  Y2 = Matrix{Float64}(undef, n₂, m)
  inv2s2 = 1 / (2 * σ^2)
  radius = k * σ

  w = Vector{Float64}()             # reusable buffer

  for i in 1:n₂
    q = @view X2[i, :]
    idxs = inrange(tree, q, radius)

    if isempty(idxs)
      @views Y2[i, :] .= NaN
      continue
    end

    # ensure weight buffer has right length
    resize!(w, length(idxs))

    # compute weights
    qx, qy, qz = q
    for (t, idx) in enumerate(idxs)
      dx = X1[idx, 1] - qx
      dy = X1[idx, 2] - qy
      dz = X1[idx, 3] - qz
      w[t] = exp(-(dx * dx + dy * dy + dz * dz) * inv2s2)
    end
    wsum = sum(w)

    # weighted sum across observations:   Y2[i,:] = (w' * Y1[idxs,:]) / wsum
    @views Y2[i, :] = (w' * Y1[idxs, :]) ./ wsum
  end

  return Y2
end

# ---------------------------------------------------------------------
# Correlation with NaN / ε filtering — vector method (existing)
# ---------------------------------------------------------------------
function filtered_correlation(y1::AbstractVector{<:Real},
  y2::AbstractVector{<:Real};
  ϵ::Real=0.0)
  @assert length(y1) == length(y2) "Vectors must be the same length"
  @assert ϵ ≥ 0 "ε must be non‑negative"

  mask = map((a, b) -> isfinite(a) && isfinite(b) && abs(a) ≥ ϵ && abs(b) ≥ ϵ,
    y1, y2)
  if !any(mask)
    return NaN
  end
  return cor(y1[mask], y2[mask])
end

# ---------------------------------------------------------------------
# Correlation with NaN / ε filtering — matrix method (new)
# ---------------------------------------------------------------------
function filtered_correlation(Y1::AbstractMatrix{<:Real},
  Y2::AbstractMatrix{<:Real};
  ϵ::Real=0.0)
  @assert size(Y1) == size(Y2) "Matrices must have identical shape"
  m = size(Y1, 2)
  r = Vector{Float64}(undef, m)
  for j in 1:m
    r[j] = filtered_correlation(@view(Y1[:, j]), @view(Y2[:, j]); ϵ=ϵ)
  end
  return r
end

# ---------------------------------------------------------------------
# Correlation with shuffled columns — matrix + permutation bootstrap
# ---------------------------------------------------------------------
"""
    filtered_correlation_shuffled(Y1, Y2; ε = 0.0, N = 100) -> Matrix

Compute `filtered_correlation(Y1, Y2[:, shuffle(cols)])` *N* times, each time
randomly permuting the **columns** of `Y2`.  Returns an `N × obs` matrix where
`obs = size(Y1,2)`; each row contains the correlations obtained for one random
shuffle.  If you prefer a single long vector, simply `vec(result)`.
"""
function filtered_correlation_shuffled(Y1::AbstractMatrix{<:Real},
  Y2::AbstractMatrix{<:Real};
  ϵ::Real=0.0,
  N::Int=100)
  @assert size(Y1) == size(Y2) "Matrices must have identical shape"
  # m = size(Y1, 2)  # observations / columns
  # results = Matrix{Float64}(undef, N, m)
  # cols = collect(1:m)

  # for n in 1:N
  #     shuffle!(cols)
  #     @views results[n, :] = filtered_correlation(Y1, Y2[:, cols]; ϵ=ϵ)
  # end

  # return vec(results)

  return reduce(
    vcat,
    [filtered_correlation(
      Y1,
      Y2[:, shuffle(1:size(Y2, 2))];
      ϵ=ϵ
    ) for _ in 1:N]
  )
end

# ---------------------------------------------------------------------
# Pairwise correlation — full matrix of filtered correlations
# ---------------------------------------------------------------------
"""
    filtered_correlation_pairwise(Y1, Y2; ε = 0.0) -> Matrix

Compute a full `m₁ × m₂` matrix `R` where
`R[i,j] = filtered_correlation(Y1[:, i], Y2[:, j]; ε = ε)`.

`Y1` and `Y2` must have the same number of rows (points), but may have
different numbers of columns (`m₁`, `m₂`).  The same NaN / ε filtering is
applied to every pair.
"""
function filtered_correlation_pairwise(Y1::AbstractMatrix{<:Real},
  Y2::AbstractMatrix{<:Real};
  ϵ::Real=0.0)
  @assert size(Y1, 1) == size(Y2, 1) "Matrices must have identical row count"
  m1 = size(Y1, 2)
  m2 = size(Y2, 2)
  R = Matrix{Float64}(undef, m1, m2)

  for i in 1:m1
    col1 = @view Y1[:, i]
    for j in 1:m2
      R[i, j] = filtered_correlation(col1, @view Y2[:, j]; ϵ=ϵ)
    end
  end
  for i in 1:m1
    if all(isnan.(R[i, :]))
      R[:, i] .= NaN
    end
  end

  return R
end

end # module
