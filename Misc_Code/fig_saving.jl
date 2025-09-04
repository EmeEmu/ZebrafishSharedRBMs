notebookpath() = replace(@__FILE__, r"#==#.*" => "")

# addapted from https://discourse.julialang.org/t/pluto-get-current-notebook-file-path/60543/8
# macro figsave(fig, name)
#   notebook = __source__.file
#   return quote
#     println(notebook)
#   end
# end
macro toJupiter(b)
  toJupiter(String(split(String(__source__.file), "#==#")[1]), b)
end
function toJupiter(a, b)
  println(a, b)
end

using CairoMakie: save, Figure

macro figpath(name::AbstractString)
  notebookpath = String(split(String(__source__.file), "#==#")[1])
  notebookname = splitext(basename(notebookpath))[1]
  notebookdir = dirname(notebookpath)
  dir = joinpath([notebookdir, "Panels", notebookname])
  if !isdir(dir)
    mkdir(dir)
  end
  path = joinpath([dir, name]) .* ".svg"
end

export save
