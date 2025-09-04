# COLORS
using ColorSchemes
using BrainRBMjulia: cmap_aseismic
COLORS_FISH = ColorSchemes.Set1_6#[0:0.25:1]
COLOR_TEACHER = "#bf0bb6" # violet eggplant
COLOR_STUDENT = "#0bbf14" # malachite
COLOR_STUDENTb = "#85df8a" # pastel green

CMAP_WEIGHTS = :seismic
CMAP_WEIGHTS_2d = cmap_aseismic()
CMAP_DENSITY = :inferno
CMAP_GOODNESS = ColorSchemes.reverse(ColorSchemes.RdYlGn_9)

# PLOT STYLES
include("styles.jl")
