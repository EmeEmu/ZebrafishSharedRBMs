using Makie
pt_px = 4 / 3
ticklabel = 8
axelabel = 10
title = 12
ticklenght = 3
rasterize = 5


style_publication = Theme(
  size=(200, 200),
  font="Arial", # ? 
  fontsize=7 * pt_px,
  Axis=(
    rightspinevisible=false,
    topspinevisible=false,
    spinewidth=1 * pt_px,
    titlefont=:bold,
    titlesize=title * pt_px,
    subtitlesize=(title - 1) * pt_px,
    xgridvisible=false,
    ygridvisible=false,
    xlabelsize=axelabel * pt_px,
    ylabelsize=axelabel * pt_px,
    xtrimspine=false,
    ytrimspine=false,
    xticksize=ticklenght * pt_px,
    yticksize=ticklenght * pt_px,
    xticklabelsize=ticklabel * pt_px,
    yticklabelsize=ticklabel * pt_px,
  ),
  Colorbar=(
    ticksize=ticklenght * pt_px,
    size=6,
    labelsize=axelabel * pt_px,
    ticklabelsize=ticklabel * pt_px
  ),
  Scatter=(
    markersize=5,
    rasterize=rasterize,
  ),
  Heatmap=(
    rasterize=rasterize,
  ),
  Hexbin=(
    colormap=:plasma,
    bins=100,
    rasterize=rasterize,
  ),
  CairoMakie=(
    px_per_unit=4.0,
    type="svg",
  ),
  Text=(
    fontsize=ticklabel * pt_px,
  ),
)
