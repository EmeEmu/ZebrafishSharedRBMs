using Pkg
Pkg.activate(Base.current_project())
using HDF5

"""
    strip_weight_maps!(dir::AbstractString; recursive=true, dry_run=false, keep_backup=false, verbose=true)

Rewrite every .h5/.hdf5 in `dir` **without** the top-level group "Weight Maps".
This actually frees disk space because files are rebuilt without that data.

Arguments
- `recursive`: search subfolders
- `dry_run`: just report which files would change; don't modify anything
- `keep_backup`: if true, keeps `file.h5.bak` (safer but uses space during the run)
- `verbose`: print per-file status

Returns a NamedTuple with a summary and per-file details.
"""
function strip_weight_maps!(dir::AbstractString; recursive::Bool=true, dry_run::Bool=false,
  keep_backup::Bool=false, verbose::Bool=true)

  # collect HDF5 files
  files = String[]
  if recursive
    for (root, _, fns) in walkdir(dir)
      append!(files, [joinpath(root, f) for f in fns
                      if isfile(joinpath(root, f)) &&
        (endswith(lowercase(f), ".h5") || endswith(lowercase(f), ".hdf5"))])
    end
  else
    append!(files, [joinpath(dir, f) for f in readdir(dir; join=false)
                    if isfile(joinpath(dir, f)) &&
      (endswith(lowercase(f), ".h5") || endswith(lowercase(f), ".hdf5"))])
  end

  details = Vector{Dict{Symbol,Any}}()
  total_saved = 0
  changed = 0

  for file in files
    before = filesize(file)
    entry = Dict{Symbol,Any}(:file => file, :before => before, :changed => false, :saved => 0)
    try
      h5open(file, "r") do src
        if haskey(src, "Weight Maps")
          entry[:changed] = true
          if dry_run
            verbose && println("Would strip: $file")
          else
            # create temp output in same folder
            tmp, io = mktemp(dirname(file))
            close(io)
            # build new file without "Weight Maps"
            h5open(tmp, "w") do dest
              # copy file-level attributes
              for nm in keys(attrs(src))
                attrs(dest)[nm] = attrs(src)[nm]
              end
              # copy all top-level objects except "Weight Maps"
              for nm in keys(src)
                nm == "Weight Maps" && continue
                copy_object(src, nm, dest, nm)  # copies groups/datasets recursively
              end
            end
            # swap files (minimizes risk and space; backup optional)
            if keep_backup
              bak = file * ".bak"
              isfile(bak) && rm(bak; force=true)
              mv(file, bak)
              mv(tmp, file)
            else
              rm(file; force=true)
              mv(tmp, file)
            end
            after = filesize(file)
            saved = max(0, before - after)
            entry[:after] = after
            entry[:saved] = saved
            total_saved += saved
            changed += 1
            verbose && println("Stripped: $file  (saved $(Base.format_bytes(saved)))")
          end
        else
          entry[:after] = before
          verbose && println("Skip (no Weight Maps): $file")
        end
      end
    catch e
      entry[:error] = string(e)
      verbose && @warn "Failed on $file: $e"
    end
    push!(details, entry)
  end

  return (summary=(n_files=length(files), n_changed=changed,
      bytes_saved=total_saved, saved_pretty=Base.format_bytes(total_saved)),
    details=details)
end

strip_weight_maps!(
  joinpath(
    dirname(Base.current_project()),
    "DataAndModels/RBMs_WBSC/bRBMs/"
  )
)
