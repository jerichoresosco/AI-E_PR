import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import inspect
import model.utils
import scipy.stats

import logging
logger = logging.getLogger(__name__)

# Folder that receives EVERY figure (plus the optional per-figure subfolder).
# Set by model.runner.plot (main.py modes "plot" and "plot_thesis"); None =
# the plots/ folder of the project.
PLOT_OUTPUT_DIR = None

# =============================================================================
# THESIS FIGURE STYLE
# =============================================================================
# Everything below is applied to EVERY figure by _style_figure(), called from
# filepath_save_plot_or_show() just before saving, so the single plot
# functions do not need to care about sizes, fonts or spines.
#
# Grid colour map ("inferno" = dark for low values, bright for high values;
# "inferno_r" = the same palette reversed).  Empty cells (no data) are drawn in
# GRID_BAD_COLOR[cmap].
GRID_CMAP = "inferno"            # "inferno" | "inferno_r"
GRID_BAD_COLOR = {"inferno": "white", "inferno_r": "0.75"}

# Figure title: only the name of the figure (no parameters, no "multi").
# SHOW_TITLE = False removes the figure title altogether (the panel titles of
# multi-panel figures are kept).
SHOW_TITLE = True

# Fixed size of ONE panel (inches).  Square panels for the plots with the same
# variable on x and y (grids, rank vs rank, r_g vs r_g, x vs y), rectangular
# otherwise.  Same height, so that figures placed side by side line up.  A
# figure with several panels is (ncols x nrows) panels; a panel with a
# colourbar gets COLORBAR_EXTRA more inches in width.
PANEL_RECT = (6.4, 4.8)
PANEL_SQUARE = (4.8, 4.8)
COLORBAR_EXTRA = 1.3
SQUARE_PLOTS = ("grid", "rank_origin", "two_radius", "mobility")

# Font sizes (pt), large because the figures are scaled down in LaTeX.
FONT_TITLE = 20          # figure title
FONT_PANEL_TITLE = 18    # panel titles of multi-panel figures
FONT_LABEL = 18          # axis labels, colourbar labels
FONT_TICKS = 16          # tick labels
FONT_LEGEND = 14
FONT_TEXT = 13           # annotations inside the axes
FONT_T0 = 17             # label of the t0 marker (end of the baseline)

# Resolution of the raster parts of the PDFs (the grids); lines and text are
# vector graphics and have no resolution limit.
SAVE_DPI = 300   # 600 resamples the grids to very large images (GBs of RAM)

# Time axis of the time-resolved plots: None = automatic (days when the axis
# goes beyond 10 days, i.e. the new parameters; seconds otherwise), True =
# always days, False = always seconds.
TIME_IN_DAYS = None
_DAY = 86400.0

# Series shown by grid_final_comparison (labels, in panel order); None = all
# the series of the multiplot (2 to 6).  Set by model.campaigns (campaign "f").
GRID_FINAL_SERIES = None
# Grid figures are drawn once per colour map in GRID_CMAPS (file name suffix
# _<cmap>, e.g. _inferno and _inferno_r) and only for the popularity metrics
# in GRID_METRICS ("explorers" = n^E; add "visits" for n^V).
GRID_CMAPS = ("inferno", "inferno_r")
GRID_METRICS = ("explorers",)
# Kendall tau_b, top-10 turnover and top-10 membership: the values are
# computed between consecutive (daily) snapshots as usual, but only one point
# every TIME_THIN_EVERY snapshots is DRAWN.  With 2: the snapshots at an odd
# distance from t0 (t0 + 1 day, t0 + 3 days, ...; before t0 likewise), so the
# first snapshot after t0 is always shown, plus the last snapshot (t1), and
# for the membership also t0 (its starting point).  1 = draw every snapshot.
TIME_THIN_EVERY = 2
_NAME_TAG = ""        # appended to the file names while looping over GRID_CMAPS
# Tag appended to the title of every saved figure (e.g. "$k$ = 5"), used by the
# sensitivity campaigns (model.campaigns) to label each panel.  "" = none.
PANEL_TAG = ""
# Layout (rows, columns) of the final grids with 5 or 6 panels.
GRID_LAYOUT_6 = (3, 2)

# Accumulation velocity of a middle block of the ranking (plot_venue_velocity_fixed,
# which="mid" frozen at t0, which="mid_final" frozen at t1): ranks from
# MID_RANKS[0] to MID_RANKS[1], both included, among the visited venues.
MID_RANKS = (100, 200)

# Extra velocity plots drawn together with the usual ones (no change to the
# plot list of main.py): with "venue_velocity_fixed_top" also the middle block
# at t0; with "venue_velocity_fixed_final" also the bottom set and the middle
# block at t1.  False = only the usual three.
VELOCITY_EXTRA = True

# Velocity sets whose legend goes in two columns ABOVE the curves, with
# VELOCITY_LEGEND_HEADROOM (fraction of the y-range) of empty space added on
# top for it, instead of matplotlib's "best" spot (which covered the curves).
VELOCITY_LEGEND_ABOVE = ("worst",)
VELOCITY_LEGEND_HEADROOM = 0.35

# Zoomed time plots (plot_kendall_over_time_window, plot_top10_turnover_window):
# the same curves as plot_kendall_over_time / plot_top10_turnover restricted to
# a time window in days, with every snapshot drawn (no thinning).  (0, 16) =
# the first 16 days of the simulation; (4, 20) = the first 16 days after t0.
# Drawn for the exploration popularity together with the full plots.
# None = not drawn.
ZOOM_WINDOW = (0, 16)

# Names of the recommenders in legends and panel titles (None = raw labels).
LABEL_NAMES = {"ge": "g-EPR", "uk": "User-kNN", "ik": "Item-kNN", "mf": "MF",
               "ra": "Random", "ce": "EPR", "de": "d-EPR",
               "mf3": "MF, $f$=3", "mf5": "MF, $f$=5", "mf10": "MF, $f$=10",
               "mf15": "MF, $f$=15", "mf20": "MF, $f$=20", "mf30": "MF, $f$=30"}
# Sensitivity labels: "<alg>_k<K>" = k (terms in the score) and
# "<alg>_ktop<K>" = k_top (length of the list).
for _alg, _name in (("uk", "User-kNN"), ("ik", "Item-kNN"), ("mf", "MF")):
    for _k in (5, 10, 15, 25, 50):
        LABEL_NAMES[f"{_alg}_k{_k}"] = f"{_name}, $k$={_k}"
        LABEL_NAMES[f"{_alg}_ktop{_k}"] = f"{_name}, $k_{{\\mathrm{{top}}}}$={_k}"

plt.rcParams.update({
    "savefig.dpi": SAVE_DPI,
    "image.interpolation": "nearest",      # crisp cells in the grids
    "axes.titlesize": FONT_TITLE,
    "axes.labelsize": FONT_LABEL,
    "xtick.labelsize": FONT_TICKS,
    "ytick.labelsize": FONT_TICKS,
    "legend.fontsize": FONT_LEGEND,
    "figure.titlesize": FONT_TITLE,
    "axes.linewidth": 1.2,
    "lines.linewidth": 1.8,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.minor.size": 3, "ytick.minor.size": 3,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.minor.width": 0.9, "ytick.minor.width": 0.9,
    "mathtext.fontset": "dejavusans",
})


def _grid_cmap():
    """Colour map of the grids, with the colour of the empty cells."""
    cmap = plt.get_cmap(GRID_CMAP).copy()
    cmap.set_bad(color=GRID_BAD_COLOR.get(GRID_CMAP, "white"))
    return cmap


def _each_grid_cmap(fn):
    """Run a grid-plotting function once per colour map of GRID_CMAPS; every
    file gets the suffix _<cmap> (no suffix when there is a single map)."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global GRID_CMAP, _NAME_TAG
        old_cmap, old_tag = GRID_CMAP, _NAME_TAG
        cmaps = tuple(GRID_CMAPS) or (GRID_CMAP,)
        try:
            for c in cmaps:
                GRID_CMAP = c
                _NAME_TAG = f"_{c}" if len(cmaps) > 1 else ""
                fn(*args, **kwargs)
        finally:
            GRID_CMAP, _NAME_TAG = old_cmap, old_tag
    return wrapper


def _bare_map_axes(ax):
    """Grid maps: no cell indices on the axes (the frame is kept)."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def _use_days(xmax):
    """True when a time axis reaching xmax (clock units = seconds) is shown in days."""
    if TIME_IN_DAYS is not None:
        return bool(TIME_IN_DAYS)
    try:
        return float(xmax) > 10 * _DAY
    except (TypeError, ValueError):
        return False


def _time_axis(ax):
    """Label (and, for long runs, tick in days) a linear time axis whose data
    are clock values in seconds.  Call it after the curves are drawn."""
    lo, hi = ax.get_xlim()
    if _use_days(hi):
        from matplotlib.ticker import FuncFormatter, MultipleLocator
        span = (hi - max(lo, 0.0)) / _DAY
        step = next((st for st in (1, 2, 4, 8, 16, 32, 64, 128) if span / st <= 8), 256)
        ax.xaxis.set_major_locator(MultipleLocator(step * _DAY))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / _DAY:g}"))
        ax.set_xlabel(r"$t$ [days]")
    elif hi > 1e4:
        # seconds, ticks in thousands (no overlapping long numbers)
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 4, 5, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / 1e3:g}"))
        ax.set_xlabel(r"$t$ [$10^3$ s]")
    else:
        ax.set_xlabel(r"$t$ [s]")


def _pretty_label(text):
    """Display name of a recommender label (exact match, or 'label (...)')."""
    if not LABEL_NAMES or not text:
        return text
    if text in LABEL_NAMES:
        return LABEL_NAMES[text]
    head, sep, tail = text.partition(" ")
    if sep and head in LABEL_NAMES:
        return f"{LABEL_NAMES[head]} {tail}"
    return text


# Markers: the points are drawn as circles ("o"); in _style_figure, just
# before saving, those of g-EPR (recognised by its colour) become GE_MARKER and
# those of every other series OTHER_MARKER (same size).  None = keep "o".
GE_MARKER = "o"
OTHER_MARKER = "s"
# Layer of g-EPR (points, lines and error bars) with respect to the other
# dynamics: "below" = drawn under them, "above" = drawn over them, None = in
# the order of the series.
GE_LAYER = "above"
_GE_Z = {"below": 2.5, "above": 10}   # other curves: 3; bands: 2


def _is_ge_color(c):
    from matplotlib.colors import to_rgba
    try:
        r, g, b, _a = to_rgba(c)
        r0, g0, b0, _a0 = to_rgba(SERIES_COLORS.get("ge", "#000000"))
    except (TypeError, ValueError):
        return False
    return abs(r - r0) + abs(g - g0) + abs(b - b0) < 1e-6


def _is_series_color(c):
    """True for the colour of a series (SERIES_COLORS or the palette)."""
    from matplotlib.colors import to_rgba
    try:
        rgb = to_rgba(c)[:3]
    except (TypeError, ValueError):
        return False
    for ref in list(SERIES_COLORS.values()) + list(_MP_COLORS):
        r = to_rgba(ref)[:3]
        if sum(abs(x - y) for x, y in zip(rgb, r)) < 1e-6:
            return True
    return False


def _ge_squares(fig):
    """Circles drawn in the colour of g-EPR become GE_MARKER, those of the
    other series OTHER_MARKER (lines, error bars, scatter points and the
    handles of the legends); g-EPR is moved to the layer GE_LAYER."""
    from matplotlib.lines import Line2D
    from matplotlib.collections import PathCollection
    from matplotlib.markers import MarkerStyle

    from matplotlib.collections import LineCollection

    def new_marker(color):
        if _is_ge_color(color):
            return GE_MARKER or "o"
        if _is_series_color(color):
            return OTHER_MARKER or "o"
        return "o"

    def fix_line(ln):
        if isinstance(ln, Line2D) and ln.get_marker() == "o":
            ln.set_marker(new_marker(ln.get_color()))

    def path_of(m):
        ms = MarkerStyle(m)
        return ms.get_path().transformed(ms.get_transform())
    for a in fig.axes:
        for ln in a.get_lines():
            fix_line(ln)
            # g-EPR drawn above all the other curves
            if GE_LAYER in _GE_Z and _is_ge_color(ln.get_color()):
                ln.set_zorder(_GE_Z[GE_LAYER]
                              + (0.2 if ln.get_marker() not in (None, "", "None", "none") else 0))
        for coll in a.collections:
            if isinstance(coll, PathCollection):
                fc = coll.get_facecolors()
                if (len(fc) and OTHER_MARKER and _is_series_color(fc[0])
                        and not _is_ge_color(fc[0])
                        and all(tuple(c) == tuple(fc[0]) for c in fc)):
                    coll.set_paths([path_of(OTHER_MARKER)])
                if len(fc) and all(_is_ge_color(c) for c in fc):
                    coll.set_paths([path_of(GE_MARKER or "o")])
                    if GE_LAYER in _GE_Z:
                        coll.set_zorder(_GE_Z[GE_LAYER] + 0.2)
            elif isinstance(coll, LineCollection) and GE_LAYER in _GE_Z:   # error bars
                ec = coll.get_colors()
                if len(ec) and all(_is_ge_color(c) for c in ec):
                    coll.set_zorder(_GE_Z[GE_LAYER])
        leg = a.get_legend()
        if leg is not None:
            leg.set_zorder(20)                               # legend above the curves
    legends = [a.get_legend() for a in fig.axes] + list(fig.legends)
    for leg in legends:
        if leg is None:
            continue
        handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])
        for h in handles:
            fix_line(h)
            if isinstance(h, PathCollection):
                fc = h.get_facecolors()
                if len(fc):
                    h.set_paths([path_of(new_marker(fc[0]))])


def _legend_markers(fig):
    """Legend entries drawn as a bare line (curves whose points are drawn by a
    separate errorbar call) get the marker of the points of the same colour,
    so that every legend shows the marker of its series (squares for g-EPR)."""
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    for a in fig.axes:
        leg = a.get_legend()
        if leg is None:
            continue
        by_color = {}
        for ln in a.get_lines():
            m = ln.get_marker()
            if m not in (None, "", "None", "none", " ", "x"):
                try:
                    by_color.setdefault(to_rgba(ln.get_color())[:3], (m, ln.get_markersize()))
                except (TypeError, ValueError):
                    pass
        handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])
        for h in handles:
            if not isinstance(h, Line2D):
                continue
            no_marker = h.get_marker() in (None, "", "None", "none", " ")
            no_line = h.get_linestyle() in ("None", "none", "", " ")
            if no_marker:
                try:
                    mk = by_color.get(to_rgba(h.get_color())[:3])
                except (TypeError, ValueError):
                    mk = None
                if mk:
                    h.set_marker(mk[0])
                    h.set_markersize(mk[1])
                    no_marker = False
            # points drawn without a line: the legend shows line + marker too
            if no_line and not no_marker:
                h.set_linestyle("-")
                h.set_linewidth(plt.rcParams.get("lines.linewidth", 1.8))


def _style_figure(fig, name=""):
    """Thesis style applied to a finished figure: fixed panel size, large
    fonts, only the left/bottom spines on line plots, clean titles."""
    from matplotlib.collections import QuadMesh

    _ge_squares(fig)
    _legend_markers(fig)

    all_axes = list(fig.axes)
    cbars = [a for a in all_axes if hasattr(a, "_colorbar")]
    main = [a for a in all_axes if a not in cbars]
    if not main:
        return

    # ---- size: fixed panel, figure = grid of panels ----
    nrows, ncols = 1, 1
    for a in main:
        ss = a.get_subplotspec() if hasattr(a, "get_subplotspec") else None
        if ss is not None:
            # topmost: a colourbar splits its panel into a sub-gridspec
            nrows, ncols = ss.get_topmost_subplotspec().get_gridspec().get_geometry()
            break
    square = any(key in (name or "") for key in SQUARE_PLOTS)
    w, h = PANEL_SQUARE if square else PANEL_RECT
    shared = getattr(fig, "_thesis_shared_cbar", None)   # (nrows, ncols) of the maps
    if shared:
        nrows, ncols = shared
        fig.set_size_inches(ncols * w + COLORBAR_EXTRA, nrows * h)
    else:
        if cbars:
            w += COLORBAR_EXTRA
        fig.set_size_inches(ncols * w, nrows * h)
    multi_panel = sum(1 for a in main if a.axison) > 1

    # ---- figure title (suptitle) ----
    sup = getattr(fig, "_suptitle", None)
    if sup is not None:
        sup.set_text(build_plot_title(sup.get_text()) if SHOW_TITLE else "")
        sup.set_fontsize(FONT_TITLE)

    for a in all_axes:
        is_cbar = a in cbars
        a.xaxis.label.set_fontsize(FONT_LABEL)
        a.yaxis.label.set_fontsize(FONT_LABEL)
        a.tick_params(which="major", labelsize=FONT_TICKS)
        a.xaxis.get_offset_text().set_fontsize(FONT_TICKS - 2)
        a.yaxis.get_offset_text().set_fontsize(FONT_TICKS - 2)
        if is_cbar:
            continue
        # title: panel title in multi-panel figures, figure title otherwise
        t = a.title.get_text()
        if t:
            if multi_panel:
                a.title.set_text(_pretty_label(t))
                a.title.set_fontsize(FONT_PANEL_TITLE)
            else:
                a.title.set_text(build_plot_title(t) if SHOW_TITLE else "")
                a.title.set_fontsize(FONT_TITLE)
        # annotations inside the axes
        for txt in a.texts:
            if txt.get_gid() == "t0_label":
                txt.set_fontsize(FONT_T0)
                continue
            txt.set_fontsize(FONT_TEXT)
        # legend
        leg = a.get_legend()
        if leg is not None:
            for lt in leg.get_texts():
                lt.set_text(_pretty_label(lt.get_text()))
                lt.set_fontsize(FONT_LEGEND)
            if leg.get_title() is not None:
                leg.get_title().set_fontsize(FONT_LEGEND)
        # spines: drop top/right on line/scatter plots, keep the frame of maps
        is_map = bool(a.images) or any(isinstance(c, QuadMesh) for c in a.collections)
        if not is_map and a.axison:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            a.tick_params(top=False, right=False, which="both")

    for leg in fig.legends:
        for lt in leg.get_texts():
            lt.set_text(_pretty_label(lt.get_text()))
            lt.set_fontsize(FONT_LEGEND)
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(FONT_LEGEND)

def build_plot_title(base_title, filters=None):
    """Title of a figure: the base title only, without "(multi)"."""
    return re.sub(r"\s*\((?i:multi)\)", "", str(base_title)).strip()


def filepath_save_plot_or_show(name=None, show=False, subfolder=None):
    """Style the current figure and save it as <name>.pdf in the plot folder
    (_resolve_plot_dir), or show it on screen when show is True.  Without a
    name, the name of the calling function is used."""
    # if name is not given, the calling function name is used instead
    if name is None:
        name = inspect.stack()[1].function

    _style_figure(plt.gcf(), name)
    if PANEL_TAG:
        _apply_panel_tag(plt.gcf(), PANEL_TAG)

    if show:
        plt.tight_layout()
        plt.show()
        return

    plots_dir = _resolve_plot_dir(subfolder)
    filename = f"{name}{_NAME_TAG}.pdf"
    filepath = os.path.join(plots_dir, filename)
    model.utils.check_path_length(filepath)

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight", dpi=SAVE_DPI)
    plt.close()
    logger.debug(f"Saved plot: {filepath}")


def _apply_panel_tag(fig, tag):
    """Append the panel tag to the figure title: to the suptitle if there is
    one, otherwise to the title of every axes that has one (colourbars have
    none).  A figure without any title gets the tag as title of its first axes."""
    sup = getattr(fig, "_suptitle", None)
    if sup is not None and sup.get_text():
        sup.set_text(f"{sup.get_text()} ({tag})")
        return
    titled = [ax for ax in fig.axes if ax.get_title()]
    for ax in titled:
        ax.set_title(f"{ax.get_title()} ({tag})")
    if not titled and fig.axes:
        fig.axes[0].set_title(tag)


def _resolve_plot_dir(subfolder=None):
    """Create and return the folder of a figure: PLOT_OUTPUT_DIR (set by
    model.runner.plot), plus the optional per-figure subfolder.  Shared by
    filepath_save_plot_or_show and by the writers of the per-plot tables, so
    they all land in the same folder."""
    root = PLOT_OUTPUT_DIR or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
    plots_dir = os.path.join(root, subfolder) if subfolder else root
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def dotted_histogram(numpy_array, bins_per_decade=4):
    """
    Compute a logarithmic histogram with density normalization.

    This function creates logarithmically spaced bins for the input array,
    computes frequencies, and normalizes to density. It filters out zero values
    and returns only bins with positive density.

    Parameters
    ----------
    numpy_array : array-like
        Input array of values to histogram (must be positive).
    bins_per_decade : int, optional (default=4)
        Number of bins per decade for logarithmic spacing.

    Returns
    -------
    centers_filtered : np.ndarray
        Geometric centers of the bins with positive density.
    density_filtered : np.ndarray
        Normalized density values for the bins.
    """
    array = numpy_array[numpy_array > 0]   # avoiding zeros

    min_array = array.min()
    max_array = array.max()

    log_min = np.log10(min_array)
    log_max = np.log10(max_array)

    #Avoiding boundary problems using a margin
    margin = 0.05
    log_min -= margin
    log_max += margin

    num_decades = log_max - log_min
    num_bins = int(np.ceil(num_decades * bins_per_decade))

    # Creation of logarithmic bins
    bins = np.logspace(log_min, log_max, num_bins + 1)

    # frequencies
    counts, edges = np.histogram(array, bins=bins)

    # bin logarithmic center
    centers = np.sqrt(edges[:-1] * edges[1:])

    # normailization
    widths = np.diff(edges)
    density = counts / widths / len(array)

    mask = density > 0 #Avoiding zero density bins 
    centers_filtered = centers[mask]
    density_filtered = density[mask]

    return centers_filtered, density_filtered


# ==========================================================================
# CROSS-SIMULATION AGGREGATION (error bands)
# ==========================================================================
#
# The statistical model behind every band drawn below:
#
#   * Within ONE simulation the N users are NOT independent (shared world,
#     shared recommender state, shared venues). They are correlated draws.
#   * Across simulations (different seeds) the runs ARE independent.
#
# Therefore the unit of independent replication is the SIMULATION, not the
# user. To quantify how reliable a curve is, we:
#   1. fix a common coordinate grid (bin edges / sample times / rank index);
#   2. compute, for EACH simulation, its own curve on that grid (this already
#      averages out the within-simulation correlation);
#   3. summarise the spread of those per-simulation curves, coordinate by
#      coordinate.
#
# Pooling all users into one flat list of N*M points and treating them as
# independent would UNDER-estimate the uncertainty (it ignores the within-run
# correlation and pretends the effective sample size is N*M instead of M).
#
# DEFAULT_BAND controls which spread is reported:
#   "ci"  : 95% confidence interval on the MEAN curve, mean ± z * SEM,
#           SEM = std_across_sims / sqrt(n_sims).  Answers "how well do I know
#           the average curve?"  Shrinks as you add simulations — this is the
#           band that visually demonstrates the result has stabilised.
#   "std" : mean ± 1 standard deviation across simulations.  Answers "how much
#           does a single simulation typically deviate?"  Does NOT shrink with
#           more simulations.
#   "pct" : median with a [16, 84] percentile band (robust, distribution-free
#           analogue of ±1σ).  Also a single-run spread; does not shrink.
# ==========================================================================

DEFAULT_BAND = "ci"
_CI_Z = 1          # 1 sigma
_PCT_LO, _PCT_HI = 16, 84

# How the uncertainty is RENDERED (orthogonal to DEFAULT_BAND, which decides
# how it is COMPUTED):
#   "errorbar" : classic vertical error bars with caps on each point
#   "fill"     : translucent shaded ribbon between the lower/upper edges
BAND_STYLE = "errorbar"


def aggregate_over_sims(per_sim_curves, band=DEFAULT_BAND):
    """
    Aggregate a stack of per-simulation curves into a central line and a band.

    Parameters
    ----------
    per_sim_curves : array-like, shape (n_sims, n_x)
        Row i is the curve produced by simulation i, evaluated on the shared
        coordinate grid.  Missing points must be encoded as NaN (e.g. a rank
        that exists only in some simulations).
    band : {"ci", "std", "pct"}
        See the module note above.

    Returns
    -------
    center, lo, hi : np.ndarray, each shape (n_x,)
        Central line and lower/upper band edges.  Columns with no valid data
        are NaN in all three.
    """
    M = np.asarray(per_sim_curves, dtype=float)
    if M.ndim == 1:
        M = M[None, :]

    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        n_valid = np.sum(np.isfinite(M), axis=0)

        if band == "pct":
            center = np.nanmedian(M, axis=0)
            lo = np.nanpercentile(M, _PCT_LO, axis=0)
            hi = np.nanpercentile(M, _PCT_HI, axis=0)
        else:
            center = np.nanmean(M, axis=0)
            std = np.nanstd(M, axis=0, ddof=1)
            if band == "std":
                lo, hi = center - std, center + std
            else:  # "ci"
                sem = std / np.sqrt(np.maximum(n_valid, 1))
                lo, hi = center - _CI_Z * sem, center + _CI_Z * sem

    # Columns with <2 valid simulations cannot support a band; collapse it.
    only_one = n_valid < 2
    lo = np.where(only_one, center, lo)
    hi = np.where(only_one, center, hi)
    # Columns with no data at all -> NaN everywhere.
    empty = n_valid < 1
    center = np.where(empty, np.nan, center)
    lo = np.where(empty, np.nan, lo)
    hi = np.where(empty, np.nan, hi)
    return center, lo, hi


def _shared_log_edges(all_positive_values, bins_per_decade, margin=0.05):
    """Build log-spaced bin edges and geometric centers shared across sims.

    The edges are derived ONCE from the pooled positive values so that every
    simulation is histogrammed on the SAME grid — the prerequisite for being
    able to aggregate the per-bin densities across simulations.
    """
    arr = np.asarray(all_positive_values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return None, None
    log_min = np.log10(arr.min()) - margin
    log_max = np.log10(arr.max()) + margin
    n_bins = max(1, int(np.ceil((log_max - log_min) * bins_per_decade)))
    edges = np.logspace(log_min, log_max, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers


def _shared_int_log_edges(all_positive_values, bins_per_decade):
    """Log-spaced bin edges for INTEGER data (counts), on half-integers.

    Plain log edges (_shared_log_edges) cut an integer axis into bins that
    hold 1, 2, 3, ... integers in an irregular sequence while their width
    grows smoothly, so below a few tens the density jumps up and down by up
    to a factor 2 (a comb, like the one _shared_linear_edges removes for
    linear bins).  Here every edge sits on a half-integer, so each bin holds
    a whole number >= 1 of consecutive integers and its width IS that number:
    density = counts / width is then the mean probability per integer, with
    no comb.  Small values get one bin per integer; the bins widen
    log-uniformly above ~bins_per_decade / ln(10).

    centers: geometric mean of the first and last integer of the bin (the
    integer itself for single-integer bins).
    """
    arr = np.asarray(all_positive_values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return None, None
    lo, hi = int(round(arr.min())), int(round(arr.max()))
    n = max(1, int(np.ceil(np.log10((hi + 1) / lo) * bins_per_decade)))
    starts = np.unique(np.ceil(np.logspace(np.log10(lo), np.log10(hi + 1), n + 1)).astype(int))
    starts = starts[(starts >= lo) & (starts <= hi + 1)]
    if starts[0] != lo:
        starts = np.insert(starts, 0, lo)
    if starts[-1] != hi + 1:
        starts = np.append(starts, hi + 1)
    edges = starts.astype(float) - 0.5
    first, last = starts[:-1], starts[1:] - 1
    centers = np.sqrt(first * last.astype(float))
    return edges, centers


def _is_integer_data(values):
    """True when every finite value is a whole number (counts)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr.size > 0 and bool(np.all(arr == np.round(arr)))


def _shared_linear_edges(all_values, n_bins=30, margin=0.0):
    """Build linear bin edges and centers shared across sims (entropy plots).

    Integer-valued data (counts such as S = number of distinct venues explored
    per user) is binned on INTEGER-ALIGNED edges. Splitting an integer range
    into a fixed number of equal-width bins whose width is not an integer makes
    some bins straddle two integers while others cover only one, so the counts
    alternate high/low and the density plot grows a spurious "comb" of spikes
    and dips on top of the true (smooth) shape. Because the observed [min, max]
    shifts from run to run, the phase of that comb shifts too, which is why the
    artefact looks different on each plot. Aligning the edges to half-integers
    guarantees every bin covers exactly the same number of consecutive integers
    and removes the artefact. Continuous data (entropy, predictability, the
    2rg/rg ratio, ...) is left on the original linspace path unchanged.
    """
    arr = np.asarray(all_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    lo, hi = arr.min(), arr.max()
    if lo == hi:
        hi = lo + 1.0

    # Integer-valued counts: bin on integer-aligned (half-integer) edges so
    # every bin holds the same number of consecutive integers (no comb).
    if np.allclose(arr, np.round(arr)):
        width = max(1, int(round((hi - lo) / n_bins)))   # ~n_bins bins, integer width
        n = int(np.ceil((hi - lo + 1) / width))
        edges = (lo - 0.5) + width * np.arange(n + 1)     # edges sit on half-integers
        centers = 0.5 * (edges[:-1] + edges[1:])
        return edges, centers

    span = hi - lo
    lo -= margin * span
    hi += margin * span
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _density_on_edges(values, edges):
    """Density-normalised histogram of `values` on fixed `edges`.

    Returns a curve aligned with the shared bin centers (length = len(edges)-1).
    Bins the simulation never reached get density 0 (a legitimate observation),
    which is what we want when averaging densities across simulations.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.zeros(len(edges) - 1)
    counts, _ = np.histogram(arr, bins=edges)
    widths = np.diff(edges)
    return counts / widths / arr.size


def _stack_density_per_sim(per_sim_values, edges):
    """Stack per-sim density curves on shared edges -> (n_sims, n_bins)."""
    return np.vstack([_density_on_edges(v, edges) for v in per_sim_values]) \
        if per_sim_values else np.empty((0, len(edges) - 1))


def _density_from_counts(counts, edges):
    """Density curve from a pre-computed COUNT vector on fixed `edges`.

    Numerically identical to _density_on_edges(raw_values, edges): the only
    difference is that the histogram counts were accumulated earlier (at load
    time) instead of here, so the raw values never had to be kept in memory.
    density = counts / bin_width / total_count.
    """
    counts = np.asarray(counts, dtype=float)
    widths = np.diff(edges)
    total = counts.sum()
    if total <= 0:
        return np.zeros(len(edges) - 1)
    return counts / widths / total


def _stack_density_from_counts(per_sim_counts, edges):
    """Stack per-sim density curves from pre-binned COUNT vectors -> (n_sims, n_bins)."""
    rows = [_density_from_counts(c, edges) for c in per_sim_counts
            if c is not None and np.asarray(c).size]
    return np.vstack(rows) if rows else np.empty((0, len(edges) - 1))


# =============================================================================
# JENSEN-SHANNON DIVERGENCE between distributions (per run, vs a reference)
# =============================================================================
# Applied to every DISTRIBUTION plot (per-user and per-venue histograms) in
# multiplot mode, on the SAME shared bins the figure uses.
#
# Per-run logic (runs are independent, never pooled):
#   * each run is reduced to its own probability vector over the bins
#     (mass per bin = counts / total of that run);
#   * series X != reference: for each run i of X, JS(p_i, q_j) is averaged over
#     the reference runs j -> one value per X-run -> mean +/- SEM over X-runs;
#   * reference itself (noise floor): for each reference run i, JS(q_i, q_j)
#     averaged over j != i -> mean +/- SEM.  A series is distinguishable from
#     the reference only if its JS clearly exceeds this floor.
# JS is in bits (log base 2), bounded in [0, 1]; 0 = identical distributions.
# Values outside the plotted bins (e.g. non-positive values on log axes) are
# not part of the comparison, exactly as they are not part of the figure.
# -----------------------------------------------------------------------------

JS_REFERENCE_LABEL = "ge"   # series label every other series is compared to


def _mass_from_density(curves, edges):
    """(n_runs, n_bins) densities -> per-run probability vectors (rows sum 1).
    Runs with no mass are dropped."""
    M = np.asarray(curves, dtype=float) * np.diff(np.asarray(edges, dtype=float))[None, :]
    tot = M.sum(axis=1)
    return M[tot > 0] / tot[tot > 0, None]


def js_divergence_bits(p, q):
    """Jensen-Shannon divergence in bits between two probability vectors."""
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)

    def _kl(a, b):
        nz = a > 0                        # 0 * log 0 = 0; m > 0 wherever a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def js_vs_reference(series_mass, ref_label=None):
    """series_mass : list of (label, (n_runs, n_bins) probability matrix).
    Returns a list of row dicts (reference first), [] if the reference series
    is absent or there is nothing to compare."""
    ref_label = ref_label or JS_REFERENCE_LABEL
    by_label = dict(series_mass)
    Q = by_label.get(ref_label)
    if Q is None or Q.shape[0] == 0 or len(series_mass) < 2:
        return []
    rows = []
    # noise floor: each reference run vs the OTHER reference runs
    if Q.shape[0] >= 2:
        vals = [np.mean([js_divergence_bits(Q[i], Q[j])
                         for j in range(Q.shape[0]) if j != i])
                for i in range(Q.shape[0])]
        m, e = _mean_sem(vals)
    else:
        m, e = np.nan, np.nan
    rows.append({"series": ref_label,
                 "comparison": f"{ref_label} vs {ref_label} (noise floor)",
                 "n_runs": int(Q.shape[0]), "n_ref_runs": int(Q.shape[0]),
                 "js_mean_bits": m, "js_sem_bits": e})
    for label, P in series_mass:
        if label == ref_label or P.shape[0] == 0:
            continue
        vals = [np.mean([js_divergence_bits(P[i], Q[j]) for j in range(Q.shape[0])])
                for i in range(P.shape[0])]
        m, e = _mean_sem(vals)
        rows.append({"series": label, "comparison": f"{label} vs {ref_label}",
                     "n_runs": int(P.shape[0]), "n_ref_runs": int(Q.shape[0]),
                     "js_mean_bits": m, "js_sem_bits": e})
    return rows


def _write_js_table(rows, plot_name, filters):
    """One TSV per distribution plot in <plots folder>/js/; multipleplot()
    merges them into js_summary.tsv (merge_js_tables)."""
    if not rows or not plot_name:
        return
    try:
        plots_dir = _resolve_plot_dir(subfolder="js")
        df = pd.DataFrame(rows)
        df.insert(0, "plot", plot_name)
        df.to_csv(os.path.join(plots_dir, f"{plot_name}.tsv"), sep="\t", index=False)
    except Exception as exc:                  # a table must never break a figure
        logger.warning(f"JS table for '{plot_name}' not written: {exc}")


def _js_for_plot(ax, series_curves, edges, plot_name, filters, show):
    """Compute and (unless show) save JS for one distribution figure.
    series_curves : list of (label, (n_runs, n_bins) density matrix)."""
    if len(series_curves) < 2:
        return
    mass = [(label, _mass_from_density(c, edges)) for label, c in series_curves]
    rows = js_vs_reference(mass)
    if not rows:
        return
    if not show:
        _write_js_table(rows, plot_name, filters)


def merge_js_tables():
    """Merge every <plots folder>/js/*.tsv into <plots folder>/js_summary.tsv
    and log the table.  Called once by multipleplot() after all figures."""
    try:
        plots_dir = _resolve_plot_dir(subfolder="js")
    except Exception:
        return None
    files = sorted(f for f in os.listdir(plots_dir) if f.endswith(".tsv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(os.path.join(plots_dir, f), sep="\t") for f in files],
                   ignore_index=True)
    out = os.path.join(os.path.dirname(plots_dir), "js_summary.tsv")
    df.to_csv(out, sep="\t", index=False)
    logger.debug("Jensen-Shannon divergence vs reference [bits]:\n" +
                df[["plot", "comparison", "n_runs", "js_mean_bits", "js_sem_bits"]]
                .to_string(index=False))
    logger.debug(f"Saved JS summary: {out}")
    return out


def _plot_prebinned_loglog(per_sim_series, edges, xlabel, ylabel,
                           filters=None, title=None, show=False, band=DEFAULT_BAND,
                           name=None):
    """Log-log density distribution drawn from PRE-BINNED per-sim count vectors.

    Mirror image of _plot_loglog_distribution, but every simulation is already
    represented by a count vector on the shared, fixed `edges` (computed at load
    time).  The per-bin density, the cross-simulation aggregation and the drawing
    are byte-for-byte the same code path, so the figure is identical to what the
    raw-values path would have produced on those same edges.

    per_sim_series : list of (label, per_sim_counts)
        per_sim_counts is a list over simulations; each element is a count
        vector of length len(edges)-1 (or None/empty for a sim with no data).
    """
    edges = np.asarray(edges, dtype=float)
    centers = np.sqrt(edges[:-1] * edges[1:])
    multi = len(per_sim_series) > 1

    fig, ax = plt.subplots(figsize=(8, 6))
    _js_curves = []
    for idx, (label, per_sim_counts) in enumerate(per_sim_series):
        curves = _stack_density_from_counts(per_sim_counts, edges)   # (n_sims, n_bins)
        if curves.shape[0] == 0:
            logger.warning(f"_plot_prebinned_loglog: no data for series '{label}'")
            continue
        _js_curves.append((label, curves))
        center, lo, hi = aggregate_over_sims(curves, band=band)
        color = _series_color(idx, multi)
        _draw_curve(ax, centers, center, lo, hi, color, n_sims=curves.shape[0],
                    label=label if multi else None, logy=True)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    _js_for_plot(ax, _js_curves, edges, name, filters, show)
    plt.tight_layout()
    filepath_save_plot_or_show(name=name, show=show)


def _plot_ranking_from_values(per_sim_series, xlabel, ylabel,
                              filters=None, title=None, show=False, band=DEFAULT_BAND,
                              name=None):
    """Rank-ordered curve with cross-simulation bands, from PRE-REDUCED per-sim
    value arrays (one value per venue, already aggregated over that sim's users).

    Identical to _plot_ranking, except each simulation is handed its 1-D array of
    per-venue counts directly instead of a list of venue DataFrames: we sort each
    array descending, align on the rank index, pad short curves with NaN, and
    aggregate at each rank -- exactly what _plot_ranking does after _venue_curve.
    """
    multi = len(per_sim_series) > 1
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (label, per_sim_vals) in enumerate(per_sim_series):
        curves = [np.sort(np.asarray(v, dtype=float))[::-1]
                  for v in per_sim_vals if v is not None and np.asarray(v).size]
        if not curves:
            logger.warning(f"_plot_ranking_from_values: no data for series '{label}'")
            continue
        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan)
        for r, c in enumerate(curves):
            padded[r, :len(c)] = c
        ranks = np.arange(1, max_len + 1)

        center, _, _ = aggregate_over_sims(padded, band=band)
        color = _series_color(idx, multi)
        # Ranking plots: scatter only -- no error bars, no connecting line.
        m = np.isfinite(center) & (center > 0)
        ax.plot(ranks[m], center[m], marker="o", linestyle="none",
                markersize=3, color=color,
                label=label if multi else None, zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(name=name, show=show)


def _draw_band(ax, x, lo, hi, color, logy=False):
    """Shade a [lo, hi] band, clamping for log axes so it stays drawable."""
    x = np.asarray(x, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    finite = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
    if not finite.any():
        return
    if logy:
        # Clamp the lower edge to a small positive floor so fill_between works
        # on a log y-axis even when the CI reaches down toward/below zero.
        pos = hi[finite][hi[finite] > 0]
        floor = (pos.min() * 1e-2) if pos.size else 1e-12
        lo = np.where(lo > 0, lo, floor)
        hi = np.where(hi > 0, hi, floor)
    ax.fill_between(x[finite], lo[finite], hi[finite],
                    color=color, alpha=0.20, linewidth=0, zorder=1)


def _series_color(idx, multi):
    # Fixed colour per recommender label when the labels are known.
    if multi and _ACTIVE_LABELS is not None and idx < len(_ACTIVE_LABELS):
        c = SERIES_COLORS.get(_ACTIVE_LABELS[idx])
        if c:
            return c
    return _MP_COLORS[idx % len(_MP_COLORS)] if multi else "#E73948"


def _draw_curve(ax, x, center, lo, hi, color, n_sims, label=None,
                logy=False, marker="o", linestyle="-", style=None,
                max_err_decades=3.0):
    """Draw a central curve plus its cross-simulation uncertainty.

    Honors the module-level BAND_STYLE unless `style` is given:
      "errorbar" -> vertical caps on each point (yerr = distance to lo/hi)
      "fill"     -> translucent shaded ribbon between lo and hi

    The band is the genuine LINEAR center +/- 1 SEM, so its meaning is
    preserved.  On a log y-axis the lower edge `center - SEM` is problematic in
    sparse-tail bins: it can cross zero (not drawable) or stay positive but tiny
    (e.g. 1e-20 under a 1e-5 point), which draws a whisker plunging many empty
    decades and wrecks the y-scale.  In BOTH cases the point keeps only its upper
    error bar and is marked with an "x": a lower whisker is allowed to reach at
    most `max_err_decades` decades below its own point; past that (zero-crossing
    included) the estimate is treated as consistent with zero and the downward
    whisker is dropped rather than drawn or faked.
    """
    style = style or BAND_STYLE
    x = np.asarray(x, dtype=float)
    center = np.asarray(center, dtype=float)
    mask = np.isfinite(x) & np.isfinite(center)
    if logy:
        mask &= center > 0
    if not mask.any():
        return

    show_band = n_sims >= 2 and lo is not None and hi is not None
    fmt = f"{marker}{linestyle}"

    if style == "errorbar" and show_band:
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        yerr_lo = np.clip(center - lo, 0, None)   # the TRUE lower reach
        yerr_hi = np.clip(hi - center, 0, None)

        # On a log y-axis the lower edge center - SEM is dropped (point marked
        # "x", only the upper bar kept) when it either crosses zero OR plunges
        # more than `max_err_decades` decades below the point itself. In both
        # cases the estimate is effectively consistent with zero and a full
        # whisker would span many empty decades, wrecking the y-scale.
        flagged = np.zeros_like(center, dtype=bool)
        if logy:
            with np.errstate(invalid="ignore"):
                lower_edge = center - yerr_lo
                min_allowed = center * (10.0 ** (-max_err_decades))
                flagged = mask & ((lower_edge <= 0) | (lower_edge < min_allowed))

        # Draw the connecting line once (if requested) so splitting the markers
        # across two errorbar calls does not break the curve. The label lives on
        # whichever artist is drawn first and is cleared afterwards, so the
        # legend never shows duplicates.
        pending_label = label
        if linestyle not in ("", "none", "None"):
            ax.plot(x[mask], center[mask], linestyle=linestyle, color=color,
                    label=pending_label, zorder=3)
            pending_label = None

        point_fmt = marker if marker else "none"

        normal = mask & ~flagged
        if normal.any():
            ax.errorbar(x[normal], center[normal],
                        yerr=[yerr_lo[normal], yerr_hi[normal]],
                        fmt=point_fmt, color=color, ecolor=color, elinewidth=1.0,
                        capsize=3, capthick=1.0, markersize=5, linestyle="none",
                        label=pending_label, zorder=3)
            pending_label = None

        if flagged.any():
            ax.errorbar(x[flagged], center[flagged],
                        yerr=[np.zeros(int(flagged.sum())), yerr_hi[flagged]],
                        fmt="x", color=color, ecolor=color, elinewidth=1.0,
                        capsize=3, capthick=1.0, markersize=7, markeredgewidth=1.5,
                        linestyle="none", label=pending_label, zorder=4)
            pending_label = None
    else:
        ax.plot(x[mask], center[mask], fmt, color=color, markersize=5,
                label=label, zorder=3)
        if show_band and style == "fill":
            _draw_band(ax, x, lo, hi, color, logy=logy)


def _to_km(per_sim_series):
    """[(label, [values per sim])] in metres -> the same in kilometres."""
    return [(label, [None if v is None else np.asarray(v, dtype=float) / 1e3
                     for v in per_sim])
            for label, per_sim in per_sim_series]


def _plot_loglog_distribution(per_sim_series, bins_per_decade, xlabel, ylabel,
                              filters=None, title=None, show=False, band=DEFAULT_BAND,
                              name=None, integer="auto"):
    """Generic log-log density distribution with cross-simulation bands.

    per_sim_series : list of (label, per_sim_values)
        per_sim_values is a list over simulations; each element is a flat
        array-like of per-user (or per-item) values for that one simulation.
    integer : "auto" | bool
        Integer data (counts, e.g. venue popularity) are binned on
        half-integer log edges (_shared_int_log_edges) to avoid the comb of
        plain log bins.  "auto" = whenever every value is a whole number.
    """
    multi = len(per_sim_series) > 1

    # Shared bin edges from ALL values across every series and simulation,
    # so curves are directly comparable and aggregatable.
    pooled = []
    for _, per_sim in per_sim_series:
        for sim_vals in per_sim:
            if sim_vals is not None and len(sim_vals):
                pooled.append(np.asarray(sim_vals, dtype=float))
    pooled = np.concatenate(pooled) if pooled else np.array([])
    if integer == "auto":
        integer = _is_integer_data(pooled)
    if integer:
        edges, centers = _shared_int_log_edges(pooled, bins_per_decade)
    else:
        edges, centers = _shared_log_edges(pooled, bins_per_decade)
    if edges is None:
        logger.warning("_plot_loglog_distribution: no positive data to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    _js_curves = []
    for idx, (label, per_sim) in enumerate(per_sim_series):
        per_sim = [s for s in per_sim if s is not None and len(s)]
        if not per_sim:
            logger.warning(f"_plot_loglog_distribution: no data for series '{label}'")
            continue
        curves = _stack_density_per_sim(per_sim, edges)          # (n_sims, n_bins)
        _js_curves.append((label, curves))
        center, lo, hi = aggregate_over_sims(curves, band=band)
        color = _series_color(idx, multi)

        _draw_curve(ax, centers, center, lo, hi, color, n_sims=len(per_sim),
                    label=label if multi else None, logy=True)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    _js_for_plot(ax, _js_curves, edges, name, filters, show)
    plt.tight_layout()
    filepath_save_plot_or_show(name=name, show=show)


def _plot_linear_distribution(per_sim_series, xlabel, ylabel, n_bins=30,
                              filters=None, title=None, show=False, band=DEFAULT_BAND,
                              name=None, xlim=None):
    """Generic linear-axis density distribution with cross-simulation bands
    (used by the entropy plots)."""
    multi = len(per_sim_series) > 1

    pooled = []
    for _, per_sim in per_sim_series:
        for sim_vals in per_sim:
            if sim_vals is not None and len(sim_vals):
                pooled.append(np.asarray(sim_vals, dtype=float))
    pooled = np.concatenate(pooled) if pooled else np.array([])
    edges, centers = _shared_linear_edges(pooled, n_bins=n_bins)
    if edges is None:
        logger.warning("_plot_linear_distribution: no data to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    _js_curves = []
    for idx, (label, per_sim) in enumerate(per_sim_series):
        per_sim = [s for s in per_sim if s is not None and len(s)]
        if not per_sim:
            logger.warning(f"_plot_linear_distribution: no data for series '{label}'")
            continue
        curves = _stack_density_per_sim(per_sim, edges)
        _js_curves.append((label, curves))
        center, lo, hi = aggregate_over_sims(curves, band=band)
        color = _series_color(idx, multi)
        _draw_curve(ax, centers, center, lo, hi, color, n_sims=len(per_sim),
                    label=label if multi else None, logy=False, linestyle="-")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    # A zoomed view (xlim) re-uses the full-range bins: its JS would duplicate
    # the non-zoomed figure's, so it is computed only there.
    if xlim is None:
        _js_for_plot(ax, _js_curves, edges, name, filters, show)
    plt.tight_layout()
    filepath_save_plot_or_show(name=name, show=show)


def _shared_sample_times(per_sim_series, bins_per_decade):
    """Log-spaced sample times shared across all series/sims for over-time plots."""
    clocks = []
    for _, per_sim in per_sim_series:
        for sim_dfs in per_sim:
            for df in sim_dfs:
                if df is not None and len(df):
                    clocks.append(df["clock"].values)
    if not clocks:
        return None
    allc = np.concatenate(clocks)
    allc = allc[allc > 0]
    if allc.size == 0:
        return None
    min_c, max_c = allc.min(), allc.max()
    num = max(2, int(round((np.log10(max_c) - np.log10(min_c)) * bins_per_decade)))
    st = np.logspace(np.log10(min_c), np.log10(max_c), num)
    st[-1] = max_c
    return st


def _column_mean(M):
    """Mean of each column of M ignoring NaN; NaN where a column has no finite
    value (e.g. the first sample times, before any user has data).  Same result
    as np.nanmean(M, axis=0), without the "Mean of empty slice" warning."""
    finite = np.isfinite(M)
    n = finite.sum(axis=0)
    tot = np.where(finite, M, 0.0).sum(axis=0)
    return np.where(n > 0, tot / np.maximum(n, 1), np.nan)


def _user_curve_S(df, sample_times):
    """S(t) for one user on the shared grid (NaN before it first reaches 1)."""
    clocks = df["clock"].values
    s = df["s"].values
    out = np.full(len(sample_times), np.nan)
    idx_s1 = np.where(s >= 1.0)[0]
    if len(idx_s1) == 0:
        return out
    t_reach_1 = clocks[idx_s1[0]]
    for k, t in enumerate(sample_times):
        if t < t_reach_1:
            continue
        mask = clocks <= t
        if mask.any():
            out[k] = s[int(np.where(mask)[0][-1])]
    return out


def _user_curve_rg(df, sample_times):
    """rg(t)/rg_final [%] for one user on the shared grid (NaN before rg>0)."""
    clocks = df["clock"].values
    rg = df["rg"].values
    out = np.full(len(sample_times), np.nan)
    idx_start = np.where(rg > 0)[0]
    if len(idx_start) == 0:
        return out
    t_start = clocks[idx_start[0]]
    final_rg = rg[-1]
    if final_rg == 0:
        return out
    for k, t in enumerate(sample_times):
        if t < t_start:
            continue
        mask = clocks <= t
        if mask.any():
            out[k] = rg[int(np.where(mask)[0][-1])] / final_rg * 100.0
    return out


def _plot_over_time(per_sim_series, user_curve_fn, sample_times, ylabel, ylog,
                    filters=None, title=None, show=False, band=DEFAULT_BAND, name=None,
                    min_coverage=0.5):
    """Generic over-time plot.

    For each simulation we first average its users at each sample time (this
    removes within-simulation correlation), giving one mean curve per sim;
    then we aggregate those per-sim mean curves to get the band.
    """
    multi = len(per_sim_series) > 1
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (label, per_sim) in enumerate(per_sim_series):
        per_sim = [dfs for dfs in per_sim if dfs]
        if not per_sim:
            logger.warning(f"_plot_over_time: no data for series '{label}'")
            continue

        per_sim_means = []
        n_valid = np.zeros(len(sample_times))   # users with data at each time (pooled)
        n_total = 0
        for sim_dfs in per_sim:
            user_curves = np.vstack([user_curve_fn(df, sample_times) for df in sim_dfs])
            n_valid += np.isfinite(user_curves).sum(axis=0)
            n_total += user_curves.shape[0]
            per_sim_means.append(_column_mean(user_curves))
        per_sim_means = np.vstack(per_sim_means)               # (n_sims, n_t)

        center, lo, hi = aggregate_over_sims(per_sim_means, band=band)

        # Drop the unreliable early samples: at the first sample times only a
        # handful of (atypical, early-mover) users have started, so the mean is
        # dominated by noise. Hide any time where fewer than `min_coverage` of
        # the population has data yet.
        coverage = n_valid / max(n_total, 1)
        weak = coverage < min_coverage
        center = center.copy(); center[weak] = np.nan
        lo = lo.copy();         lo[weak]     = np.nan
        hi = hi.copy();         hi[weak]     = np.nan

        color = _series_color(idx, multi)
        _draw_curve(ax, sample_times, center, lo, hi, color, n_sims=len(per_sim),
                    label=label if multi else None, logy=ylog)

    ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(name=name, show=show)


def _venue_curve(vh_list, mode):
    """Sorted descending venue curve for ONE simulation.

    mode="visits"     -> total visits per venue (summed over the sim's users)
    mode="popularity" -> distinct users per venue
    Returns a 1-D array sorted descending (length = number of venues in the sim).
    """
    if mode == "visits":
        totals = {}
        for vh in vh_list:
            id_col = vh.columns[0]
            for _, row in vh.iterrows():
                vid = row[id_col]
                totals[vid] = totals.get(vid, 0) + int(row["Number of visits"])
        y = np.array(sorted(totals.values(), reverse=True), dtype=float)
    else:  # popularity
        user_count = {}
        for u_idx, vh in enumerate(vh_list):
            id_col = vh.columns[0]
            for _, row in vh.iterrows():
                vid = row[id_col]
                user_count.setdefault(vid, set()).add(u_idx)
        y = np.array(sorted((len(s) for s in user_count.values()), reverse=True),
                     dtype=float)
    return y


def _venue_counts(vh_list, mode):
    """Per-venue count values for ONE simulation (for the count distribution).

    mode="visits"    -> total visits per venue
    mode="explorers" -> distinct users per venue
    Returns a 1-D array of counts (one entry per venue, unsorted).
    """
    if mode == "visits":
        totals = {}
        for vh in vh_list:
            id_col = vh.columns[0]
            for _, row in vh.iterrows():
                vid = row[id_col]
                totals[vid] = totals.get(vid, 0) + int(row["Number of visits"])
        counts = np.array(list(totals.values()), dtype=float)
    else:  # explorers
        user_count = {}
        for u_idx, vh in enumerate(vh_list):
            id_col = vh.columns[0]
            for _, row in vh.iterrows():
                vid = row[id_col]
                user_count.setdefault(vid, set()).add(u_idx)
        counts = np.array([len(s) for s in user_count.values()], dtype=float)
    return counts[counts > 0]


def _plot_ranking(per_sim_series, mode, xlabel, ylabel,
                  filters=None, title=None, show=False, band=DEFAULT_BAND, name=None):
    """Generic rank-ordered venue curve with cross-simulation bands.

    Each simulation yields its own descending curve y_sim(rank).  Pooling
    venues across simulations would be wrong here (a venue in run A is not the
    same entity as in run B), so we align the per-sim curves on the rank index
    and aggregate at each rank, padding shorter curves with NaN.
    """
    multi = len(per_sim_series) > 1
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (label, per_sim) in enumerate(per_sim_series):
        per_sim = [vh for vh in per_sim if vh]
        if not per_sim:
            logger.warning(f"_plot_ranking({mode}): no data for series '{label}'")
            continue
        curves = [_venue_curve(vh, mode) for vh in per_sim]
        curves = [c for c in curves if c.size]
        if not curves:
            continue
        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan)
        for r, c in enumerate(curves):
            padded[r, :len(c)] = c
        ranks = np.arange(1, max_len + 1)

        center, _, _ = aggregate_over_sims(padded, band=band)
        color = _series_color(idx, multi)
        # Ranking plots: scatter only -- no error bars, no connecting line.
        m = np.isfinite(center) & (center > 0)
        ax.plot(ranks[m], center[m], marker="o", linestyle="none",
                markersize=3, color=color,
                label=label if multi else None, zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(name=name, show=show)

def plot_mobility_visualization(user_history, venue_history, stats = None, filters = None, title = None, show = False):
    logger.debug("Plotting mobility visualization")
    """
    Visualizes the user mobility trajectory and visited venues.

    Parameters
    ----------
    user_history : pandas.DataFrame
        DataFrame containing the user's trajectory with columns including 'Destination_x_coord',
        'Destination_y_coord', and 'Clock'.
    venue_history : pandas.DataFrame
        DataFrame containing visited venues with columns 'Venue_x_coord', 'Venue_y_coord',
        and 'Number of visits'.
    filters : dict, optional
        Dictionary of filters to be included in the plot title.
        If provided, the center of mass and radius of gyration are plotted.
    title : str, optional
        Title for the plot.
    
    Returns
    -------
    None
    """


    x = venue_history["Venue_x_coord"].to_numpy()
    y = venue_history["Venue_y_coord"].to_numpy()
    num_visits = venue_history["Number of visits"].to_numpy()
    x_traj = user_history["Destination_x_coord"].to_numpy()
    y_traj = user_history["Destination_y_coord"].to_numpy()
    # Adding starting position
    x_traj, y_traj = np.insert(x_traj, 0, 0), np.insert(y_traj, 0, 0)

    _, ax = plt.subplots()

    # Normalize number of visits for colormap
    norm = plt.Normalize(num_visits.min(), num_visits.max())
    cmap = plt.cm.viridis  # Most used color scale
    sc = ax.scatter(x, y, s=10, c=num_visits, cmap=cmap, norm=norm, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Number of visits')
    ax.plot(x_traj, y_traj, color='red', linewidth=1, alpha=0.7)

    if stats is not None:
        r_gyr = stats.query('metric == "radius of gyration"')['value'].iloc[0]
        cm_x = stats.query('metric == "x_center of mass"')['value'].iloc[0]
        cm_y = stats.query('metric == "y_center of mass"')['value'].iloc[0]

        ax.scatter(cm_x,cm_y, marker='x', c="purple", alpha=0.6, s=50, zorder=3)
        circle = plt.Circle(
            (cm_x, cm_y),
            r_gyr,
            facecolor=(0.6, 0, 0.8, 0.2),
            edgecolor=(0.4, 0, 0.6, 0.8), 
            linewidth=2
        )
        ax.add_patch(circle) 

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title,fontsize=7)

    filepath_save_plot_or_show(show=show)


_MP_COLORS = [
    "#000000", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#FF5722", "#8BC34A", "#673AB7", "#009688",
]

# Fixed colour per recommender, so that the same recommender has the same
# colour in every multiplot whatever the order/number of series.
# _ACTIVE_LABELS is set by the caller (model.runner.plot, the time-resolved
# dispatcher in simulation.py) to the labels of the series being drawn, in
# order; _series_color() then returns SERIES_COLORS[label].
SERIES_COLORS = {"ge": "#000000", "uk": "#2196F3", "ik": "#4CAF50",
                 "mf": "#FF9800", "ra": "#9C27B0",
                 # MF with f latent factors (campaign "f"): light -> dark
                 "mf3": "#FDD49E", "mf5": "#FDAE6B", "mf10": "#FD8D3C",
                 "mf15": "#F16913", "mf20": "#D94801", "mf30": "#8C2D04"}
# Sensitivity to k and k_top (campaigns "k" and "k_top"):
# labels "<alg>_k<K>", light -> dark with K, in the colour family of <alg>.
TOP_K_VALUES = (5, 10, 15, 25, 50)
for _alg, _shades in (("uk", ("#C6DBEF", "#9ECAE1", "#6BAED6", "#2171B5", "#08306B")),
                      ("ik", ("#C7E9C0", "#A1D99B", "#74C476", "#238B45", "#00441B")),
                      ("mf", ("#FDD49E", "#FDAE6B", "#FD8D3C", "#D94801", "#8C2D04"))):
    for _k, _c in zip(TOP_K_VALUES, _shades):
        SERIES_COLORS[f"{_alg}_k{_k}"] = _c
        SERIES_COLORS[f"{_alg}_ktop{_k}"] = _c
_ACTIVE_LABELS = None

def _single_as_series(data, label=""):
    """Wrap a single-series call into the list-of-(label, data) format."""
    return [(label, data)]


def plot_rg_distribution(rg_list, filters=None, title=None, show=False, series_list=None,
                         per_sim_series=None, band=DEFAULT_BAND):
    """
    Plot the distribution of radius of gyration values on a log-log scale.

    Preferred mode (with error bands): pass ``per_sim_series`` as
    ``[(label, per_sim_values), ...]`` where ``per_sim_values`` is a list over
    simulations and each element is that simulation's flat array of per-user rg
    values.  When a series has >=2 simulations a cross-simulation band is drawn.

    Legacy modes (no band): ``series_list`` of flat lists, or a single flat
    ``rg_list``.  Behaviour unchanged.

    Parameters
    ----------
    rg_list : list or array-like
        Per-user rg values (used only when series_list/per_sim_series is None).
    filters : dict, optional
    title : str, optional
    show : bool, optional
    series_list : list of (str, list), optional
        Multi-series override: [(label, rg_list), ...].
    per_sim_series : list of (str, list-of-list), optional
        Per-simulation override enabling error bands.
    band : {"ci", "std", "pct"}
    """
    logger.debug("Plotting rg distribution")

    if per_sim_series is not None:
        per_sim_series = _to_km(per_sim_series)
        _plot_loglog_distribution(per_sim_series, bins_per_decade=16,
                                  xlabel=r"$r_g$ [km]", ylabel=r"$P(r_g)$",
                                  filters=filters, title=title, show=show, band=band,
                                  name="plot_rg_distribution")
        return

    if series_list is None:
        series_list = _single_as_series(rg_list)
    multi = len(series_list) > 1

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (label, data) in enumerate(series_list):
        arr = np.array(data)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if len(arr) == 0:
            logger.warning(f"plot_rg_distribution: no valid data for series '{label}'")
            continue
        centers, density = dotted_histogram(arr / 1e3, bins_per_decade=16)
        kw = {"color": _MP_COLORS[idx % len(_MP_COLORS)]} if multi else {}
        ax.loglog(centers, density, "o-", markersize=5, label=label if multi else None, **kw)

    ax.set_xlabel(r"$r_g$ [km]")
    ax.set_ylabel(r"$P(r_g)$")
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(show=show)


def plot_rg_rank(per_sim_series, filters=None, title=None, show=False,
                 band=DEFAULT_BAND):
    """Rank plot of the radius of gyration: per-user r_g sorted in descending
    order and plotted against the rank, log-log, mean over simulations.
    Missing values (NaN) are dropped first: np.sort would put them at the top
    of the descending ranking and shift every run's curve by their number."""
    logger.debug("Plotting rg rank plot")
    clean = []
    for label, per_sim in _to_km(per_sim_series):
        runs = []
        for v in per_sim:
            if v is None:
                continue
            v = np.asarray(v, dtype=float)
            runs.append(v[np.isfinite(v)])
        clean.append((label, runs))
    _plot_ranking_from_values(clean, xlabel="Rank", ylabel=r"$r_g$ [km]",
                              filters=filters, title=title, show=show, band=band,
                              name="plot_rg_rank")


def plot_rg_e_distribution(rg_list, filters=None, title=None, show=False, series_list=None,
                           per_sim_series=None, band=DEFAULT_BAND):
    """
    Plot the distribution of the *explorative* radius of gyration (rg computed
    over exploration steps only) on a log-log scale.

    Identical machinery to :func:`plot_rg_distribution`, but it writes to a
    distinct output file ("plot_rg_e_distribution") and uses an exploration
    axis label so the two plots never collide.
    """
    logger.debug("Plotting rg explorative distribution")

    if per_sim_series is not None:
        per_sim_series = _to_km(per_sim_series)
        _plot_loglog_distribution(per_sim_series, bins_per_decade=16,
                                  xlabel=r"$r_g^{\mathrm{E}}$ [km]", ylabel=r"$P(r_g^{\mathrm{E}})$",
                                  filters=filters, title=title, show=show, band=band,
                                  name="plot_rg_e_distribution")
        return

    if series_list is None:
        series_list = _single_as_series(rg_list)
    multi = len(series_list) > 1

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (label, data) in enumerate(series_list):
        arr = np.array(data)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if len(arr) == 0:
            logger.warning(f"plot_rg_e_distribution: no valid data for series '{label}'")
            continue
        centers, density = dotted_histogram(arr / 1e3, bins_per_decade=16)
        kw = {"color": _MP_COLORS[idx % len(_MP_COLORS)]} if multi else {}
        ax.loglog(centers, density, "o-", markersize=5, label=label if multi else None, **kw)

    ax.set_xlabel(r"$r_g^{\mathrm{E}}$ [km]")
    ax.set_ylabel(r"$P(r_g^{\mathrm{E}})$")
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(name="plot_rg_e_distribution", show=show)


def plot_dist_distribution(mean_dist_list, filters=None, title=None, show=False, series_list=None,
                           per_sim_series=None, band=DEFAULT_BAND, per_sim_hist=None):
    """
    Plot the distribution of individual travel distances (delta_r) on a log-log scale.

    Each data point represents a single exploration jump, not a per-user mean.

    Preferred mode (with error bands): pass ``per_sim_series`` as
    ``[(label, per_sim_values), ...]`` where ``per_sim_values[i]`` is the flat
    array of all delta_r jumps observed in simulation i.

    Legacy modes (no band): ``series_list`` of flat lists, or a single flat list.

    Parameters
    ----------
    mean_dist_list : list or array-like
        Flat list of individual delta_r values (used only when the per-sim /
        series overrides are None).
    filters : dict, optional
    title : str, optional
    show : bool, optional
    series_list : list of (str, list), optional
    per_sim_series : list of (str, list-of-list), optional
    band : {"ci", "std", "pct"}
    """
    logger.debug("Plotting travel distance distribution (all delta_r values)")

    # Preferred memory-light path: distances were histogrammed at load time onto
    # fixed shared edges, so only per-sim count vectors reach us here.
    # per_sim_hist = (edges, [(label, per_sim_counts), ...]).
    if per_sim_hist is not None:
        edges, hist_series = per_sim_hist
        # metres -> km: the counts stay, the density is then per km
        _plot_prebinned_loglog(hist_series, np.asarray(edges, dtype=float) / 1e3,
                               xlabel=r"$d$ [km]", ylabel=r"$P(d)$",
                               filters=filters, title=title, show=show, band=band,
                               name="plot_dist_distribution")
        return

    if per_sim_series is not None:
        _plot_loglog_distribution(_to_km(per_sim_series), bins_per_decade=8,
                                  xlabel=r"$d$ [km]", ylabel=r"$P(d)$",
                                  filters=filters, title=title, show=show, band=band,
                                  name="plot_dist_distribution")
        return

    if series_list is None:
        series_list = _single_as_series(mean_dist_list)
    multi = len(series_list) > 1

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (label, data) in enumerate(series_list):
        arr = np.array(data, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if len(arr) == 0:
            logger.warning(f"plot_dist_distribution: no valid data for series '{label}'")
            continue
        centers, density = dotted_histogram(arr / 1e3, bins_per_decade=8)
        kw = {"color": _MP_COLORS[idx % len(_MP_COLORS)]} if multi else {}
        ax.loglog(centers, density, "o-", markersize=5, label=label if multi else None, **kw)

    ax.set_xlabel(r"$d$ [km]")
    ax.set_ylabel(r"$P(d)$")
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(show=show)


def plot_frequency_distribution(venue_history_list, gamma, filters=None, title=None, show=False,
                                per_sim_list=None, band=DEFAULT_BAND):
    """
    Plot the rank-frequency distribution of venue visits averaged across users.

    For each user the venues are ranked in descending order of visit count.
    The visit counts at each rank are averaged across users, producing a mean
    curve.  A power-law fit line is overlaid for comparison.

    Preferred mode (with error bands): pass ``per_sim_list`` as a list over
    simulations, each element being that simulation's list of venue DataFrames.
    The per-rank average is computed within each simulation, then aggregated
    across simulations (band = spread of the per-simulation mean curves).

    Parameters
    ----------
    venue_history_list : list of pandas.DataFrame or pandas.DataFrame
        One DataFrame per user (used only when per_sim_list is None).
    gamma : float
        Exponent for the power-law fit line (f(x) = x^(-1-gamma)).
    filters : dict, optional
    title : str, optional
    show : bool, optional
    per_sim_list : list of (list of pd.DataFrame), optional
    band : {"ci", "std", "pct"}
    """
    logger.debug("Plotting frequency distribution (averaged across users)")

    def _per_sim_rank_curve(vh_list, max_len):
        """Average visits at each rank within one simulation, padded to max_len."""
        out = np.full(max_len, np.nan)
        acc = {}
        for vh in vh_list:
            sv = vh['Number of visits'].sort_values(ascending=False).values
            for rank, visits in enumerate(sv, start=1):
                acc.setdefault(rank, []).append(visits)
        for rank, vals in acc.items():
            out[rank - 1] = np.mean(vals)
        return out

    if per_sim_list is not None:
        per_sim_list = [vh for vh in per_sim_list if vh]
        if not per_sim_list:
            logger.warning("plot_frequency_distribution: no data to plot")
            return
        max_len = max(
            max((len(vh) for vh in sim_vh), default=0) for sim_vh in per_sim_list
        )
        if max_len == 0:
            logger.warning("plot_frequency_distribution: empty venue histories")
            return
        curves = np.vstack([_per_sim_rank_curve(sim_vh, max_len) for sim_vh in per_sim_list])
        ranks = np.arange(1, max_len + 1)
        center, lo, hi = aggregate_over_sims(curves, band=band)

        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        _draw_curve(ax, ranks, center, lo, hi, "blue", n_sims=len(per_sim_list),
                    label='avg data', logy=True)
        m = np.isfinite(center) & (center > 0)

        if m.any():
            x_fit = np.linspace(ranks[m].min(), ranks[m].max(), 100)
            y_fit = x_fit ** (-1 - gamma)
            y_fit = y_fit * np.nanmax(center[m]) / y_fit.max()
            ax.plot(x_fit, y_fit, color='red', linestyle='--', label=r'$f(x) = x^{-1-\gamma}$')

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel("Rank"); ax.set_ylabel("Average frequency")
        if title:
            ax.set_title(title, fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        filepath_save_plot_or_show(show=show)
        return

    if isinstance(venue_history_list, pd.DataFrame):
        venue_history_list = [venue_history_list]

    rank_visits = {}
    for vh in venue_history_list:
        sorted_visits = vh['Number of visits'].sort_values(ascending=False).values
        for rank, visits in enumerate(sorted_visits, start=1):
            rank_visits.setdefault(rank, []).append(visits)

    ranks      = np.array(sorted(rank_visits.keys()))
    avg_visits = np.array([np.mean(rank_visits[r]) for r in ranks])

    plt.figure(figsize=(8, 6))
    plt.plot(ranks, avg_visits, "o-", color='blue', label='avg data', markersize=6)

    x_fit = np.linspace(ranks.min(), ranks.max(), 100)
    y_fit = x_fit ** (-1 - gamma)
    y_fit = y_fit * avg_visits.max() / y_fit.max()
    plt.plot(x_fit, y_fit, color='red', linestyle='--', label=r'$f(x) = x^{-1-\gamma}$')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Rank")
    plt.ylabel("Average frequency")
    if title:
        plt.title(title, fontsize=7)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    filepath_save_plot_or_show(show=show)


def plot_two_radius_vs_radius(rg_list, two_rg_list, filters=None, title=None, show=False,
                              series_list=None):
    """
    Plot 2-rg vs rg.

    In single-series mode (series_list=None) behaves exactly as before.
    In multi-series mode pass series_list as [(label, rg_list, two_rg_list), ...].

    Parameters
    ----------
    rg_list : list or array-like
        Used only when series_list is None.
    two_rg_list : list or array-like
        Used only when series_list is None.
    filters : dict, optional
    title : str, optional
    show : bool, optional
    series_list : list of (str, list, list), optional
    """
    logger.debug("Plotting 2-radius vs radius of gyration")

    if series_list is None:
        series_list = [("", rg_list, two_rg_list)]
    multi = len(series_list) > 1

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (label, rg, two_rg) in enumerate(series_list):
        kw = {"color": _MP_COLORS[idx % len(_MP_COLORS)], "alpha": 0.7} if multi else {}
        ax.scatter(rg, two_rg, s=5, label=label if multi else None, **kw)

    ax.set_xlabel("Radius of Gyration")
    ax.set_ylabel("2-Radius of Gyration")
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(show=show)


def plot_S_over_time_from_df(s_over_time_list, filters=None, title=None, bins_per_decade=4,
                             show=False, series_list=None,
                             per_sim_series=None, band=DEFAULT_BAND):
    if per_sim_series is not None:
        st = _shared_sample_times(per_sim_series, bins_per_decade)
        if st is None:
            logger.warning("plot_S_over_time_from_df: no data to plot")
            return
        _plot_over_time(per_sim_series, _user_curve_S, st, ylabel="S(t)", ylog=True,
                        filters=filters, title=title, show=show, band=band,
                        name="plot_S_over_time_from_df")
        return
    return _plot_S_over_time_from_df_legacy(
        s_over_time_list, filters=filters, title=title,
        bins_per_decade=bins_per_decade, show=show, series_list=series_list)


def _plot_S_over_time_from_df_legacy(s_over_time_list, filters=None, title=None, bins_per_decade=4,
                             show=False, series_list=None):
    """
    Plot S(t) over time.

    In single-series mode (series_list=None) behaves exactly as before.
    In multi-series mode pass series_list as [(label, s_over_time_list), ...];
    s_over_time_list is then ignored.

    Parameters
    ----------
    s_over_time_list : list of pd.DataFrame
        Each DataFrame has columns ['clock', 's'].  Used only when series_list is None.
    filters : dict, optional
    title : str, optional
    bins_per_decade : int, optional
    show : bool, optional
    series_list : list of (str, list of pd.DataFrame), optional
    """
    logger.debug("Plotting S over time (from analysis data)")

    if series_list is None:
        series_list = _single_as_series(s_over_time_list)
    multi = len(series_list) > 1

    if not any(data for _, data in series_list):
        logger.warning("No S-over-time data available to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (label, data_list) in enumerate(series_list):
        if not data_list:
            logger.warning(f"plot_S_over_time_from_df: no data for series '{label}'")
            continue

        all_clocks = np.concatenate([df["clock"].values for df in data_list])
        min_c, max_c = all_clocks.min(), all_clocks.max()
        num_samples  = max(2, int(round((np.log10(max_c) - np.log10(min_c)) * bins_per_decade)))
        sample_times = np.logspace(np.log10(min_c), np.log10(max_c), num_samples)
        sample_times[-1] = max_c

        pct_by_time = {t: [] for t in sample_times}
        for df in data_list:
            clocks = df["clock"].values
            s      = df["s"].values
            idx_s1 = np.where(s >= 1.0)[0]
            if len(idx_s1) == 0:
                continue
            t_reach_1 = clocks[idx_s1[0]]
            for t in sample_times:
                if t < t_reach_1:
                    continue
                mask   = clocks <= t
                i_last = int(np.where(mask)[0][-1])
                pct_by_time[t].append(s[i_last])

        sample_t_plot = [t for t in sample_times if pct_by_time[t]]
        avg_pct_plot  = [np.mean(pct_by_time[t]) for t in sample_t_plot]

        kw = {"color": _MP_COLORS[idx % len(_MP_COLORS)]} if multi else {}
        ax.plot(sample_t_plot, avg_pct_plot, "o-", markersize=5,
                label=label if multi else None, **kw)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$S(t)$")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(show=show)


def plot_r_gyr_over_time_from_df(rg_over_time_list, filters=None, title=None, bins_per_decade=4,
                                 show=False, series_list=None,
                                 per_sim_series=None, band=DEFAULT_BAND):
    if per_sim_series is not None:
        st = _shared_sample_times(per_sim_series, bins_per_decade)
        if st is None:
            logger.warning("plot_r_gyr_over_time_from_df: no data to plot")
            return
        _plot_over_time(per_sim_series, _user_curve_rg, st,
                        ylabel=r"$r_g(t)\,/\,r_g^{last}$ [%]", ylog=True,
                        filters=filters, title=title, show=show, band=band,
                        name="plot_r_gyr_over_time_from_df")
        # absolute version (km, linear time) and difference from g-EPR
        plot_r_gyr_over_time_abs(per_sim_series, filters=filters, show=show)
        return
    return _plot_r_gyr_over_time_from_df_legacy(
        rg_over_time_list, filters=filters, title=title,
        bins_per_decade=bins_per_decade, show=show, series_list=series_list)


def _user_curve_rg_abs(df, sample_times):
    """rg(t) in km for one user on the shared grid (NaN before the first step)."""
    clocks = df["clock"].values
    rg = df["rg"].values
    idx = np.searchsorted(clocks, sample_times, side="right") - 1
    out = np.full(len(sample_times), np.nan)
    ok = idx >= 0
    out[ok] = rg[idx[ok]] / 1000.0
    return out


def plot_r_gyr_over_time_abs(per_sim_series, filters=None, show=False, n_points=64,
                             min_coverage=0.5, ref_label="ge"):
    """<r_g(t)> in km (mean over the users of each run, then mean +/- SEM over
    the runs) on a LINEAR time axis, plus a second figure with the difference
    from the reference series (g-EPR), where the small differences between the
    dynamics become visible.  Writes plot_r_gyr_over_time_abs and
    plot_r_gyr_over_time_diff."""
    clocks = [df["clock"].values for _l, per_sim in per_sim_series
              for sim in per_sim for df in sim if df is not None and len(df)]
    if not clocks:
        logger.warning("plot_r_gyr_over_time_abs: no data")
        return
    t_max = _t1_of(filters) or max(float(c.max()) for c in clocks)
    t_min = min(float(c[c > 0].min()) for c in clocks if (c > 0).any())
    st = np.linspace(max(t_min, t_max / n_points), t_max, n_points)

    curves = []                       # (label, mean, sem, n_runs)
    for label, per_sim in per_sim_series:
        per_sim = [dfs for dfs in per_sim if dfs]
        if not per_sim:
            continue
        means, n_valid, n_total = [], np.zeros(st.size), 0
        for sim_dfs in per_sim:
            U = np.vstack([_user_curve_rg_abs(df, st) for df in sim_dfs])
            n_valid += np.isfinite(U).sum(axis=0)
            n_total += U.shape[0]
            means.append(_column_mean(U))
        agg = _batch_band(means)
        if agg is None:
            continue
        _k, mean, sem, nb = agg
        weak = n_valid / max(n_total, 1) < min_coverage
        mean = np.where(weak, np.nan, mean); sem = np.where(weak, np.nan, sem)
        curves.append((label, mean, sem, nb))
    if not curves:
        return

    # ---- absolute ----
    fig, ax = plt.subplots(figsize=PANEL_RECT)
    arrays = []
    for idx, (label, mean, sem, nb) in enumerate(curves):
        _draw_points(ax, st, mean, sem, _color_of(label, idx), nb, label=label, ms=3.5)
        arrays += [mean - sem, mean + sem]
    _draw_t0_line(ax, filters)
    _fit_ylim(ax, arrays)
    _time_axis(ax)
    ax.set_ylabel(r"$\langle r_g(t) \rangle$ [km]")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    ax.set_title("Radius of gyration over time")
    filepath_save_plot_or_show(name="plot_r_gyr_over_time_abs", show=show)

    # ---- difference from the reference ----
    ref = next((c for c in curves if c[0] == ref_label), None)
    if ref is None or len(curves) < 2:
        return
    _l, m0, e0, _n = ref
    fig, ax = plt.subplots(figsize=PANEL_RECT)
    arrays = []
    for idx, (label, mean, sem, nb) in enumerate(curves):
        if label == ref_label:
            continue
        d = mean - m0
        e = np.sqrt(sem ** 2 + e0 ** 2)
        _draw_points(ax, st, d, e, _color_of(label, idx), nb, label=label, ms=3.5)
        arrays += [d - e, d + e]
    _draw_t0_line(ax, filters)
    _fit_ylim(ax, arrays)
    _zero_line(ax)
    _time_axis(ax)
    ax.set_ylabel(r"$\Delta \langle r_g \rangle$ vs g-EPR [km]")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    ax.set_title("Radius of gyration, difference from g-EPR")
    filepath_save_plot_or_show(name="plot_r_gyr_over_time_diff", show=show)


def _plot_r_gyr_over_time_from_df_legacy(rg_over_time_list, filters=None, title=None, bins_per_decade=4,
                                 show=False, series_list=None):
    """
    Plot normalised rg(t)/rg_final [%] over time.

    In single-series mode (series_list=None) behaves exactly as before.
    In multi-series mode pass series_list as [(label, rg_over_time_list), ...];
    rg_over_time_list is then ignored.

    Parameters
    ----------
    rg_over_time_list : list of pd.DataFrame
        Each DataFrame has columns ['clock', 'rg'].  Used only when series_list is None.
    filters : dict, optional
    title : str, optional
    bins_per_decade : int, optional
    show : bool, optional
    series_list : list of (str, list of pd.DataFrame), optional
    """
    logger.debug("Plotting r_gyr over time (normalized percentage, from analysis data)")

    if series_list is None:
        series_list = _single_as_series(rg_over_time_list)
    multi = len(series_list) > 1

    if not any(data for _, data in series_list):
        logger.warning("No rg-over-time data available to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (label, data_list) in enumerate(series_list):
        if not data_list:
            logger.warning(f"plot_r_gyr_over_time_from_df: no data for series '{label}'")
            continue

        all_clocks = np.concatenate([df["clock"].values for df in data_list])
        min_c, max_c = all_clocks.min(), all_clocks.max()
        num_samples  = max(2, int(round((np.log10(max_c) - np.log10(min_c)) * bins_per_decade)))
        sample_times = np.logspace(np.log10(min_c), np.log10(max_c), num_samples)
        sample_times[-1] = max_c

        pct_by_time = {t: [] for t in sample_times}
        for df in data_list:
            clocks = df["clock"].values
            rg     = df["rg"].values
            idx_start = np.where(rg > 0)[0]
            if len(idx_start) == 0:
                continue
            t_start  = clocks[idx_start[0]]
            final_rg = rg[-1]
            if final_rg == 0:
                logger.warning("Final rg is 0 despite rg > 0 earlier, skipping user")
                continue
            for t in sample_times:
                if t < t_start:
                    continue
                mask   = clocks <= t
                i_last = int(np.where(mask)[0][-1])
                pct_by_time[t].append(rg[i_last] / final_rg * 100.0)

        sample_t_plot = [t for t in sample_times if pct_by_time[t]]
        avg_pct_plot  = [np.mean(pct_by_time[t]) for t in sample_t_plot]

        kw = {"color": _MP_COLORS[idx % len(_MP_COLORS)]} if multi else {}
        ax.plot(sample_t_plot, avg_pct_plot, "o-", markersize=5,
                label=label if multi else None, **kw)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$r_g(t)\,/\,r_g^{last}$ [%]")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(show=show)

def print_batch_stats(batch_stats_list, filters=None, title=None):
    """
    Print a formatted summary table of batch-level statistics to stdout.

    Fully generic: iterates over every metric/value row present in each
    batch_stats.parquet without any hardcoded metric names.  This means
    that any new metric added to batch_stats.parquet during analysis is
    automatically displayed here without touching this function.

    One section is printed per batch, mirroring the per-batch loop used
    by the batch_grid plotter.  When more than one batch is available, a
    recap table follows with mean and std for every numeric metric.

    Parameters
    ----------
    batch_stats_list : list of (str, pd.DataFrame)
        As returned by :func:`load_all_batch_stats`:
        (batch_id, stats_df) pairs sorted by batch_id.
        Each DataFrame must have columns 'metric' and 'value'.
    filters : dict, optional
        Simulation parameters — printed as a header line for context.
    title : str, optional
        Optional title printed above the output block.
    """
    if not batch_stats_list:
        logger.warning("print_batch_stats: no batch stats to display.")
        return

    SEP_WIDE = "=" * 72

    # ------------------------------------------------------------------
    # Pass 1: collect every metric's per-batch values (generic: no metric
    # name is hardcoded, so new metrics appear automatically).
    # ------------------------------------------------------------------
    per_batch = []          # [(batch_id, {metric: value}), ...]
    metric_names = []       # preserves first-seen order
    for batch_id, stats_df in batch_stats_list:
        d = {}
        for _, row in stats_df.iterrows():
            m = str(row["metric"])
            d[m] = row["value"]
            if m not in metric_names:
                metric_names.append(m)
        per_batch.append((batch_id, d))

    # ------------------------------------------------------------------
    # Pass 2: mean / std / sem per metric, and the decimal place they imply.
    #
    # std is the SAMPLE standard deviation (ddof=1, Bessel-corrected), matching
    # the plot error bands in _compute_band() and utils.export_all_stats().
    # sem = std / sqrt(n) is the standard error OF THE MEAN (this is where the
    # sqrt(N) factor lives).  Both are undefined (nan) with a single batch:
    # printing 0.0 would falsely imply zero variance.
    #
    # Every value of a metric is then printed at the decimal place set by that
    # metric's sem rounded to 2 significant figures, so per-batch values and the
    # summary line up digit by digit and carry no fake precision.
    # ------------------------------------------------------------------
    def _clean(values):
        out = []
        for v in values:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isnan(f):
                out.append(f)
        return out

    summary = {}    # metric -> (mean, std, sem)
    decimals = {}   # metric -> int
    for m in metric_names:
        vals = _clean(d.get(m, float("nan")) for _bid, d in per_batch)
        n = len(vals)
        if n == 0:
            mean = std = sem = float("nan")
        elif n == 1:
            mean, std, sem = vals[0], float("nan"), float("nan")
        else:
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1))
            sem = std / np.sqrt(n)
        summary[m] = (mean, std, sem)
        decimals[m] = model.utils.significant_decimals(sem, std, mean)

    def _fmt(value, metric):
        return model.utils.format_to_decimals(value, decimals[metric]) or "n/a"

    # --- header ---
    print(f"\n{SEP_WIDE}")
    print(f"  {title}" if title else "  BATCH STATS SUMMARY")
    if filters:
        print("  params: " + ", ".join(f"{k}={v}" for k, v in filters.items()))
    print(f"  {len(batch_stats_list)} batch(es) found")
    print(SEP_WIDE)

    # ------------------------------------------------------------------
    # Body.  The table is TRANSPOSED relative to batch_stats.parquet: one row
    # per metric, one column per batch, then the mean/std/sem summary.  With a
    # dozen metrics a metric-per-column layout produces lines hundreds of
    # characters wide; metric-per-row stays readable and puts each metric's
    # per-batch values right next to its own summary.
    #
    # Batch ids are 22 characters, so they are replaced by short b0..bN labels
    # and a legend, keeping the columns narrow.
    # ------------------------------------------------------------------
    multi = len(per_batch) > 1

    if multi:
        print("\n  batches:")
        for i, (batch_id, _d) in enumerate(per_batch):
            print(f"    b{i} = {batch_id}")
        print()

        headers = ["metric"] + [f"b{i}" for i in range(len(per_batch))] + ["mean", "std", "sem"]
        rows = []
        for m in metric_names:
            cells = [_fmt(d.get(m, float("nan")), m) for _bid, d in per_batch]
            mean, std, sem = summary[m]
            rows.append([m] + cells + [_fmt(mean, m), _fmt(std, m), _fmt(sem, m)])
    else:
        batch_id, d = per_batch[0]
        print(f"\n  batch: {batch_id}\n")
        headers = ["metric", "value"]
        rows = [[m, _fmt(d.get(m, float("nan")), m)] for m in metric_names]

    for line in model.utils.render_aligned_table(headers, rows):
        print("  " + line)

    print(f"\n{SEP_WIDE}")
    if multi:
        print("  std = sample standard deviation across batches (ddof=1)")
        print("  sem = std / sqrt(n_batches), standard error of the mean")
        print("  values are quoted to the decimal place set by sem (2 s.f.)")
        print(SEP_WIDE)
    print()

def plot_venue_ranking(venue_history_list, mode="visits", filters=None, title=None,
                       show=False, series_list=None,
                       per_sim_series=None, band=DEFAULT_BAND, per_sim_values=None):
    """
    Plot venues sorted by descending rank on the x-axis.

    Preferred mode (with error bands): pass ``per_sim_series`` as
    ``[(label, per_sim_vh), ...]`` where ``per_sim_vh[i]`` is the list of venue
    DataFrames belonging to simulation i.  Each simulation yields its own
    descending rank curve; bands are taken across simulations at each rank
    (venues are NOT pooled across simulations — they are different entities).

    Two modes are supported via the *mode* parameter:

    ``"visits"``
        Y-axis = total number of visits to each venue, summed across all users.
        Venues are ranked by total visit count (rank 1 = most visited).

    ``"popularity"``
        Y-axis = number of *distinct* users that have visited each venue at
        least once.  Venues are ranked by user count (rank 1 = most popular).

    Works in both single-series mode (series_list=None) and multi-series mode
    (multipleplot / "mp"), following the same pattern as plot_rg_distribution().

    Parameters
    ----------
    venue_history_list : list of pandas.DataFrame
        One DataFrame per user, each containing at least the columns
        ``'Venue_name'`` (or the first column as venue identifier) and
        ``'Number of visits'``.
        Ignored when series_list is provided.
    mode : str
        ``"visits"`` or ``"popularity"``.
    filters : dict, optional
    title : str, optional
    show : bool, optional
    series_list : list of (str, list of pd.DataFrame), optional
        Multi-series override: [(label, venue_history_list), ...].
    """
    _MODE_LABELS = {
        "visits":     ("Venue rank (by total visits)",    "Total visits"),
        "popularity": ("Venue rank (by distinct visitors)", "Number of distinct visitors"),
    }
    if mode not in _MODE_LABELS:
        raise ValueError(f"plot_venue_ranking: unknown mode '{mode}'. Use 'visits' or 'popularity'.")

    xlabel, ylabel = _MODE_LABELS[mode]
    logger.debug(f"Plotting venue ranking (mode='{mode}')")

    # Preferred memory-light path: per-venue counts were already aggregated over
    # each sim's users at load time, so we get one value array per simulation.
    # per_sim_values = [(label, [array_per_sim, ...]), ...] (already mode-correct).
    if per_sim_values is not None:
        _plot_ranking_from_values(per_sim_values, xlabel=xlabel, ylabel=ylabel,
                                  filters=filters, title=title, show=show, band=band,
                                  name=f"plot_venue_ranking_{mode}")
        return

    if per_sim_series is not None:
        _plot_ranking(per_sim_series, mode=mode, xlabel=xlabel, ylabel=ylabel,
                      filters=filters, title=title, show=show, band=band,
                      name=f"plot_venue_ranking_{mode}")
        return

    if series_list is None:
        series_list = _single_as_series(venue_history_list)
    multi = len(series_list) > 1

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (label, vh_list) in enumerate(series_list):
        if not vh_list:
            logger.warning(f"plot_venue_ranking({mode}): no data for series '{label}'")
            continue

        if mode == "visits":
            # Aggregate total visits per venue across all users
            venue_totals = {}
            for vh in vh_list:
                # The venue identifier is in the first column
                id_col = vh.columns[0]
                for _, row in vh.iterrows():
                    vid = row[id_col]
                    venue_totals[vid] = venue_totals.get(vid, 0) + int(row["Number of visits"])
            y_values = np.array(sorted(venue_totals.values(), reverse=True), dtype=float)

        else:  # popularity
            # Count distinct users per venue
            venue_user_count = {}
            for user_idx, vh in enumerate(vh_list):
                id_col = vh.columns[0]
                for _, row in vh.iterrows():
                    vid = row[id_col]
                    venue_user_count.setdefault(vid, set()).add(user_idx)
            y_values = np.array(
                sorted((len(v) for v in venue_user_count.values()), reverse=True),
                dtype=float,
            )

        if len(y_values) == 0:
            logger.warning(f"plot_venue_ranking({mode}): empty y_values for series '{label}'")
            continue

        x_values = np.arange(1, len(y_values) + 1)
        color = _MP_COLORS[idx % len(_MP_COLORS)] if multi else "#E73948"
        ax.plot(x_values, y_values, marker="o", linestyle="none",
                markersize=3, color=color,
                label=label if multi else None)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(show=show)


def plot_venue_count_distribution(venue_history_list, mode="visits", filters=None,
                                   title=None, show=False, series_list=None,
                                   per_sim_series=None, band=DEFAULT_BAND,
                                   per_sim_values=None):
    """
    Plot the distribution of per-venue counts as a log-log histogram.

    Preferred mode (with error bands): pass ``per_sim_series`` as
    ``[(label, per_sim_vh), ...]`` where ``per_sim_vh[i]`` is the list of venue
    DataFrames for simulation i.  Per-venue counts are computed within each
    simulation, then the density curves are aggregated across simulations.

    Two modes are supported via the *mode* parameter:

    ``"visits"``
        X-axis = total number of visits received by a venue (summed across all
        users).  Each venue contributes one value to the distribution.

    ``"explorers"``
        X-axis = number of *distinct* users that have visited each venue at
        least once.  Each venue contributes one value.

    The Y-axis shows the probability density (normalised histogram), so the
    shape is directly comparable across datasets with different numbers of
    venues.

    Works in both single-series mode (series_list=None) and multi-series
    mode (``series_list`` provided), following the same pattern as
    plot_rg_distribution().

    Parameters
    ----------
    venue_history_list : list of pandas.DataFrame
        One DataFrame per user, each containing at least the first column as
        venue identifier and ``'Number of visits'``.
        Ignored when series_list is provided.
    mode : str
        ``"visits"`` or ``"explorers"``.
    filters : dict, optional
    title : str, optional
    show : bool, optional
    series_list : list of (str, list of pd.DataFrame), optional
        Multi-series override: [(label, venue_history_list), ...].
    """
    _MODE_LABELS = {
        "visits":    (r"$n^{\mathrm{V}}$", r"$P(n^{\mathrm{V}})$"),
        "explorers": (r"$n^{\mathrm{E}}$", r"$P(n^{\mathrm{E}})$"),
    }
    if mode not in _MODE_LABELS:
        raise ValueError(
            f"plot_venue_count_distribution: unknown mode '{mode}'. "
            "Use 'visits' or 'explorers'."
        )

    xlabel, ylabel = _MODE_LABELS[mode]

    # Preferred memory-light path: per-venue counts were already aggregated over
    # each sim's users at load time, so the value arrays feed the generic drawer
    # directly (they are exactly what _venue_counts would have produced).
    # per_sim_values = [(label, [array_per_sim, ...]), ...] (already mode-correct).
    if per_sim_values is not None:
        _plot_loglog_distribution(per_sim_values, bins_per_decade=16,
                                  xlabel=xlabel, ylabel=ylabel,
                                  filters=filters, title=title, show=show, band=band,
                                  name=f"plot_venue_count_distribution_{mode}")
        return

    if per_sim_series is not None:
        # Convert each simulation's venue DataFrames into its array of per-venue
        # counts, then reuse the generic log-log distribution drawer.
        converted = []
        for label, per_sim_vh in per_sim_series:
            per_sim_counts = [_venue_counts(vh, mode) for vh in per_sim_vh if vh]
            converted.append((label, per_sim_counts))
        _plot_loglog_distribution(converted, bins_per_decade=16,
                                  xlabel=xlabel, ylabel=ylabel,
                                  filters=filters, title=title, show=show, band=band,
                                  name=f"plot_venue_count_distribution_{mode}")
        return
    logger.debug(f"Plotting venue count distribution (mode='{mode}')")

    if series_list is None:
        series_list = _single_as_series(venue_history_list)
    multi = len(series_list) > 1

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (label, vh_list) in enumerate(series_list):
        if not vh_list:
            logger.warning(
                f"plot_venue_count_distribution({mode}): no data for series '{label}'"
            )
            continue

        if mode == "visits":
            # Sum visits per venue across all users
            venue_totals = {}
            for vh in vh_list:
                id_col = vh.columns[0]
                for _, row in vh.iterrows():
                    vid = row[id_col]
                    venue_totals[vid] = venue_totals.get(vid, 0) + int(row["Number of visits"])
            counts = np.array(list(venue_totals.values()), dtype=float)

        else:  # explorers
            # Count distinct users per venue
            venue_user_count = {}
            for user_idx, vh in enumerate(vh_list):
                id_col = vh.columns[0]
                for _, row in vh.iterrows():
                    vid = row[id_col]
                    venue_user_count.setdefault(vid, set()).add(user_idx)
            counts = np.array(
                [len(v) for v in venue_user_count.values()], dtype=float
            )

        counts = counts[counts > 0]
        if len(counts) == 0:
            logger.warning(
                f"plot_venue_count_distribution({mode}): empty counts for series '{label}'"
            )
            continue

        centers, density = dotted_histogram(counts, bins_per_decade=16)
        color = _MP_COLORS[idx % len(_MP_COLORS)] if multi else "#E73948"
        ax.loglog(centers, density, "o-", markersize=5, color=color,
                  label=label if multi else None)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(show=show)


@_each_grid_cmap
def plot_aggregate_grid(grid_matrix, filters=None, title=None, batch_id=None, mask_center=True,
                        show=False, cbar_label="Total visits", name_override=None):
    """ Plot a heatmap of an aggregate grid matrix.

        Parameters
        ----------
        grid_matrix : np.ndarray
            2D integer array with a per-cell value. By default this is the
            cumulative visit count per cell; for the exploration variant it is
            the number of distinct users that ever visited the cell.
        filters : dict, optional
            Used for the saved plot file name.
        title : str, optional
            Plot title.
        batch_id : str, optional
            When provided, included in the saved file name to distinguish individual batch plots from the param-level plot.
        mask_center : bool, optional
            Whether to mask the center cell in the plot.
        cbar_label : str, optional
            Colorbar label (default "Total visits"). Use e.g. "Number of distinct
            visitors" for the exploration variant.
        name_override : str, optional
            Base file name for the saved plot (default "plot_aggregate_grid").
            Lets the exploration variant write to a distinct file.

    """

    logger.debug(f"Plotting aggregate grid heatmap{' for batch ' + batch_id if batch_id else ''}")

    # Copy
    m = grid_matrix.copy()
    cx = m.shape[0] // 2
    cy = m.shape[1] // 2
    m_safe = np.where(m == 0, 1, m)
    if mask_center:
        m_masked = np.ma.array(m_safe, mask=False)
        m_masked.mask[cx, cy] = True
    else:
        m_masked = np.ma.array(m_safe, mask=False)
    fig, ax = plt.subplots(figsize=PANEL_SQUARE)
    cmap = _grid_cmap()

    im = ax.imshow(
        m_masked.T,
        origin="lower",
        aspect="equal",
        cmap=cmap,
    )

    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    _bare_map_axes(ax)

    if title:
        ax.set_title(title, fontsize=6)

    base = name_override or "plot_aggregate_grid"
    fname = f"{base}{'_' + batch_id if batch_id else ''}"

    filepath_save_plot_or_show(name=fname, show=show)


def plot_entropy_distribution(entropy_list, entropy_type="srand", filters=None, title=None,
                              show=False, series_list=None,
                              per_sim_series=None, band=DEFAULT_BAND):
    """
    Plot the distribution of one entropy measure across users.

    Preferred mode (with error bands): pass ``per_sim_series`` as
    ``[(label, per_sim_values), ...]`` where ``per_sim_values[i]`` is the array
    of per-user entropy values for simulation i.

    Parameters
    ----------
    entropy_list : list or array-like
        Per-user entropy values.  Ignored when per_sim_series/series_list given.
    entropy_type : str
        One of "srand", "sunc", "sreal".  Controls the axis label.
    filters : dict, optional
    title : str, optional
    show : bool, optional
    series_list : list of (str, list), optional
        Multi-series override: [(label, entropy_list), ...].
    per_sim_series : list of (str, list-of-list), optional
    band : {"ci", "std", "pct"}
    """
    _LABELS = {
        "srand": (r"$E^{\mathrm{rand}}$ [bits]", r"$P(E^{\mathrm{rand}})$"),
        "sunc":  (r"$E^{\mathrm{unc}}$ [bits]",  r"$P(E^{\mathrm{unc}})$"),
        "sreal": (r"$E$ [bits]", r"$P(E)$"),
    }
    xlabel, ylabel = _LABELS.get(entropy_type, ("Entropy", r"$P(S)$"))

    logger.debug(f"Plotting entropy distribution ({entropy_type})")

    if per_sim_series is not None:
        # Keep distinct file names per entropy type (function name is shared).
        _plot_linear_distribution(per_sim_series, xlabel=xlabel, ylabel=ylabel,
                                  n_bins=30, filters=filters, title=title,
                                  show=show, band=band,
                                  name=f"plot_entropy_distribution_{entropy_type}")
        return

    if series_list is None:
        series_list = _single_as_series(entropy_list)
    multi = len(series_list) > 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (label, data) in enumerate(series_list):
        arr = np.array(data, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            logger.warning(f"plot_entropy_distribution({entropy_type}): no valid data for '{label}'")
            continue

        color = _MP_COLORS[idx % len(_MP_COLORS)] if multi else "#E73948"

        counts, edges = np.histogram(arr, bins=30, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, counts, "o-", markersize=5, color=color,
                label=label if multi else None)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(show=show)


def plot_num_venues_distribution(num_venues_list, filters=None, title=None, show=False,
                                 series_list=None, per_sim_series=None, band=DEFAULT_BAND):
    """
    Distribution, ACROSS USERS, of S = the number of distinct venues each user
    explored (the per-user ``num_venues`` value == final exploration count).

    This is deliberately USER-centric and different from the venue-centric
    plots (venue_visits_distribution / venue_explorers_distribution, which are
    distributions over venues).  Machinery mirrors plot_entropy_distribution:
    a linear-axis density with cross-simulation error bands in the preferred
    per_sim_series mode, and a simple linear histogram otherwise.

    Parameters
    ----------
    num_venues_list : list or array-like
        Per-user S values (used only when per_sim_series/series_list is None).
    per_sim_series : list of (str, list-of-list), optional
        [(label, [per-sim arrays of per-user S]), ...] -> enables bands.
    series_list : list of (str, list), optional
        Multi-series override without bands.
    band : {"ci", "std", "pct"}
    """
    xlabel, ylabel = r"$S$", r"$P(S)$"
    logger.debug("Plotting num_venues (S) distribution")

    if per_sim_series is not None:
        _plot_linear_distribution(per_sim_series, xlabel=xlabel, ylabel=ylabel,
                                  n_bins=30, filters=filters, title=title,
                                  show=show, band=band,
                                  name="plot_num_venues_distribution")
        return

    if series_list is None:
        series_list = _single_as_series(num_venues_list)
    multi = len(series_list) > 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (label, data) in enumerate(series_list):
        arr = np.array(data, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            logger.warning(f"plot_num_venues_distribution: no valid data for '{label}'")
            continue
        color = _MP_COLORS[idx % len(_MP_COLORS)] if multi else "#E73948"
        counts, edges = np.histogram(arr, bins=30, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, counts, "o-", markersize=5, color=color,
                label=label if multi else None)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(name="plot_num_venues_distribution", show=show)


def plot_entropy_triple(s_rand_list, s_unc_list, s_real_list, filters=None, title=None, show=False,
                        per_sim_triple=None, band=DEFAULT_BAND):
    """
    Plot the distributions of S^rand, S^unc, and S^real on a single graph,
    all on linear scale.

    Preferred mode (with error bands): pass ``per_sim_triple`` as a tuple
    ``(s_rand_per_sim, s_unc_per_sim, s_real_per_sim)`` where each element is a
    list over simulations of per-user values.  Each of the three curves then
    gets its own cross-simulation band on a shared bin grid.
    """
    logger.debug("Plotting entropy triple (s_rand, s_unc, s_real)")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    if per_sim_triple is not None:
        sr_ps, su_ps, sre_ps = per_sim_triple
        specs = [
            (sr_ps,  r"$S^{rand}$", "r"),
            (su_ps,  r"$S^{unc}$",  "k"),
            (sre_ps, r"$S^{real}$", "b"),
        ]
        # Shared linear edges across all three measures and all simulations.
        pooled = []
        for per_sim, _, _ in specs:
            for v in per_sim:
                if v is not None and len(v):
                    pooled.append(np.asarray(v, dtype=float))
        pooled = np.concatenate(pooled) if pooled else np.array([])
        edges, centers = _shared_linear_edges(pooled, n_bins=30)
        if edges is not None:
            for per_sim, label, color in specs:
                per_sim = [v for v in per_sim if v is not None and len(v)]
                if not per_sim:
                    continue
                curves = _stack_density_per_sim(per_sim, edges)
                center, lo, hi = aggregate_over_sims(curves, band=band)
                _draw_curve(ax, centers, center, lo, hi, color, n_sims=len(per_sim),
                            label=label, logy=False, linestyle="-")
    else:
        specs = [
            (s_rand_list, r"$S^{rand}$", "r"),
            (s_unc_list,  r"$S^{unc}$",  "k"),
            (s_real_list, r"$S^{real}$", "b"),
        ]
        for data, label, color in specs:
            arr = np.array(data, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                continue
            counts, edges = np.histogram(arr, bins=30, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, counts, "o-", markersize=5, color=color, label=label)

    ax.set_xlabel("Entropy")
    ax.set_ylabel(r"$P(S)$")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    if title:
        fig.suptitle(title, fontsize=7, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    filepath_save_plot_or_show(show=show)


# Shared axis labels for the maximum-predictability plots.
_PI_LABELS = {
    "pi_rand": r"$\Pi^{\mathrm{rand}}$",
    "pi_unc":  r"$\Pi^{\mathrm{unc}}$",
    "pi_real": r"$\Pi^{\max}$",
}


def plot_predictability_distribution(pi_list, pi_type="pi_real", filters=None, title=None,
                                     show=False, series_list=None,
                                     per_sim_series=None, band=DEFAULT_BAND):
    """
    Plot the distribution of one maximum-predictability measure across users.

    Exact analogue of :func:`plot_entropy_distribution`: linear axes, 30 bins,
    cross-simulation bands when ``per_sim_series`` is given.  Pi is bounded in
    [1/N, 1], so the linear scale is the natural one.

    Preferred mode (with error bands): pass ``per_sim_series`` as
    ``[(label, per_sim_values), ...]`` where ``per_sim_values[i]`` is the array
    of per-user Pi values for simulation i.

    Parameters
    ----------
    pi_list : list or array-like
        Per-user Pi values.  Ignored when per_sim_series/series_list is given.
    pi_type : str
        One of "pi_rand", "pi_unc", "pi_real".  Controls the axis label and
        keeps the output file names distinct (the function name is shared).
    filters : dict, optional
    title : str, optional
    show : bool, optional
    series_list : list of (str, list), optional
        Multi-series override: [(label, pi_list), ...].
    per_sim_series : list of (str, list-of-list), optional
    band : {"ci", "std", "pct"}
    """
    xlabel = _PI_LABELS.get(pi_type, r"$\Pi$")
    _PI_YLABELS = {
        "pi_rand": r"$P(\Pi^{\mathrm{rand}})$",
        "pi_unc":  r"$P(\Pi^{\mathrm{unc}})$",
        "pi_real": r"$P(\Pi^{\max})$",
    }
    ylabel = _PI_YLABELS.get(pi_type, r"$P(\Pi)$")

    logger.debug(f"Plotting predictability distribution ({pi_type})")

    if per_sim_series is not None:
        _plot_linear_distribution(per_sim_series, xlabel=xlabel, ylabel=ylabel,
                                  n_bins=30, filters=filters, title=title,
                                  show=show, band=band,
                                  name=f"plot_predictability_distribution_{pi_type}")
        return

    if series_list is None:
        series_list = _single_as_series(pi_list)
    multi = len(series_list) > 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (label, data) in enumerate(series_list):
        arr = np.asarray(data, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            logger.warning(f"plot_predictability_distribution({pi_type}): no valid data for '{label}'")
            continue

        color = _MP_COLORS[idx % len(_MP_COLORS)] if multi else "#E73948"

        counts, edges = np.histogram(arr, bins=30, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, counts, "o-", markersize=5, color=color,
                label=label if multi else None)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(name=f"plot_predictability_distribution_{pi_type}",
                               show=show)


def plot_predictability_triple(pi_rand_list, pi_unc_list, pi_real_list, filters=None,
                               title=None, show=False, per_sim_triple=None,
                               band=DEFAULT_BAND):
    """
    Plot the distributions of Pi^rand, Pi^unc and Pi^max on a single graph,
    all on linear scale.

    The Pi counterpart of :func:`plot_entropy_triple`.  The three curves are
    Fano bounds for three source models of the same user -- uniform i.i.d.,
    empirical-marginal i.i.d., and the real correlated process -- exactly as
    reported by Song et al. (2010).  Since g decreases on [1/N, 1] and
    S^real <= S^unc <= S^rand, the curves are ordered:

        Pi^rand <= Pi^unc <= Pi^max

    and each gap is readable: rand -> unc is what spatial heterogeneity buys,
    unc -> max what temporal structure buys on top.

    Two things to keep in mind when reading the figure.  Pi^rand equals 1/|L_u|
    exactly (the correct answer for a uniform source, and a free check on the
    solver), so the red curve is the distribution of the inverse number of
    distinct venues.  And all three are UPPER BOUNDS on predictor accuracy,
    tight only when the non-modal probability mass is spread uniformly; on
    skewed distributions they overshoot what any predictor can attain.

    Preferred mode (with error bands): pass ``per_sim_triple`` as a tuple
    ``(pi_rand_per_sim, pi_unc_per_sim, pi_real_per_sim)`` where each element
    is a list over simulations of per-user values.  Each of the three curves
    then gets its own cross-simulation band on a shared bin grid.

    Colours match plot_entropy_triple (r/k/b) so the two figures read as a pair.
    """
    logger.debug("Plotting predictability triple (pi_rand, pi_unc, pi_real)")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    if per_sim_triple is not None:
        pr_ps, pu_ps, pre_ps = per_sim_triple
        specs = [
            (pr_ps,  r"$\Pi^{rand}$", "r"),
            (pu_ps,  r"$\Pi^{unc}$",  "k"),
            (pre_ps, r"$\Pi^{max}$",  "b"),
        ]
        # Shared linear edges across all three measures and all simulations,
        # so the three densities are comparable bin by bin.
        pooled = []
        for per_sim, _, _ in specs:
            for v in per_sim:
                if v is not None and len(v):
                    pooled.append(np.asarray(v, dtype=float))
        pooled = np.concatenate(pooled) if pooled else np.array([])
        edges, centers = _shared_linear_edges(pooled, n_bins=30)
        if edges is not None:
            for per_sim, label, color in specs:
                per_sim = [v for v in per_sim if v is not None and len(v)]
                if not per_sim:
                    continue
                curves = _stack_density_per_sim(per_sim, edges)
                center, lo, hi = aggregate_over_sims(curves, band=band)
                _draw_curve(ax, centers, center, lo, hi, color, n_sims=len(per_sim),
                            label=label, logy=False, linestyle="-")
    else:
        specs = [
            (pi_rand_list, r"$\Pi^{rand}$", "r"),
            (pi_unc_list,  r"$\Pi^{unc}$",  "k"),
            (pi_real_list, r"$\Pi^{max}$",  "b"),
        ]
        for data, label, color in specs:
            arr = np.asarray(data, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                continue
            counts, edges = np.histogram(arr, bins=30, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, counts, "o-", markersize=5, color=color, label=label)

    ax.set_xlabel("Maximum predictability")
    ax.set_ylabel(r"$P(\Pi)$")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    if title:
        fig.suptitle(title, fontsize=7, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    filepath_save_plot_or_show(name="plot_predictability_triple", show=show)


def plot_two_rg_ratio_distribution(ratio_list, filters=None, title=None, show=False,
                                   series_list=None, per_sim_series=None,
                                   band=DEFAULT_BAND, xlim=None, name="plot_2rad_distrib"):
    """
    Plot the distribution of the ratio 2rg / rg across users ("2rad_distrib").

    2rg is the radius of gyration restricted to a user's two most-visited
    venues, rg the full one; the ratio measures how much of the mobility
    footprint is already accounted for by those two anchor points.  It is
    dimensionless and bounded in [0, 1], so linear axes are the right choice --
    a log-log plot of a variable that piles up near 1 would hide the structure.

    Users with rg == 0 (single-point trajectories) contribute nan upstream and
    are silently dropped here.

    Preferred mode (with error bands): pass ``per_sim_series`` as
    ``[(label, per_sim_values), ...]``.
    """
    xlabel = r"$s_2$"
    ylabel = r"$P(s_2)$"

    logger.debug("Plotting 2rg/rg ratio distribution")

    if per_sim_series is not None:
        _plot_linear_distribution(per_sim_series, xlabel=xlabel, ylabel=ylabel,
                                  n_bins=30, filters=filters, title=title,
                                  show=show, band=band, xlim=xlim,
                                  name=name)
        return

    if series_list is None:
        series_list = _single_as_series(ratio_list)
    multi = len(series_list) > 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (label, data) in enumerate(series_list):
        arr = np.asarray(data, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            logger.warning(f"plot_two_rg_ratio_distribution: no valid data for '{label}'")
            continue

        color = _MP_COLORS[idx % len(_MP_COLORS)] if multi else "#E73948"

        counts, edges = np.histogram(arr, bins=30, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, counts, "o-", markersize=5, color=color,
                label=label if multi else None)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(name=name, show=show)


def plot_two_rg_ratio_distribution_zoomed(ratio_list, filters=None, title=None, show=False,
                                          series_list=None, per_sim_series=None,
                                          band=DEFAULT_BAND):
    """
    Zoomed companion of :func:`plot_two_rg_ratio_distribution`.

    Exactly the same distribution of the ratio r_g^(2) / r_g, but with the
    x-axis pinned to the full theoretical support [0, 1] instead of auto-ranging
    to the observed [min, max].  Reuses the same plotting code; only the x-limits
    and the output file name ("plot_2rad_distrib_zoomed") differ.
    """
    plot_two_rg_ratio_distribution(
        ratio_list, filters=filters, title=title, show=show,
        series_list=series_list, per_sim_series=per_sim_series, band=band,
        xlim=(0.0, 1.0), name="plot_2rad_distrib_zoomed",
    )


# ============================================================================
# Rank-persistence diagnostics: cold-start (init -> end-ge) vs recommender
# (end-ge -> final).  Uses _MP_COLORS, _series_color, build_plot_title and
# filepath_save_plot_or_show defined above.
#
# Data contract: each series provides a list of per-user `user_history`
# DataFrames carrying columns 'Departure', 'Destination', 'Clock'.
# ============================================================================

_PERSIST_K_DEFAULT = (10, 25, 50)


def _visited_sorted(series):
    s = series[series.values > 0]
    return s.sort_values(ascending=False, kind="stable")


def overlap_emersion_at_k(early, late, k):
    """Retention overlap@K and emersion@K between two snapshots.
    emersion@K = 1 - overlap@K.  K clipped to min(K, #visited_early, #visited_late)."""
    ve, vl = _visited_sorted(early), _visited_sorted(late)
    keff = int(min(k, len(ve), len(vl)))
    if keff <= 0:
        return np.nan, np.nan, 0
    se = set(ve.index[:keff])
    sl = set(vl.index[:keff])
    overlap = len(se & sl) / keff
    return overlap, 1.0 - overlap, keff


def retention_curve(early, late):
    """overlap@K for K = 1..min(#visited_early, #visited_late). Returns (ks, ov)."""
    ve, vl = _visited_sorted(early), _visited_sorted(late)
    kmax = int(min(len(ve), len(vl)))
    if kmax <= 0:
        return np.array([]), np.array([])
    ie, il = list(ve.index), list(vl.index)
    se, sl = set(), set()
    inter = 0
    ov = np.empty(kmax)
    for j in range(kmax):
        a, b = ie[j], il[j]
        if a in sl:
            inter += 1
        se.add(a)
        if b in se:                 # includes a just added -> handles a == b
            inter += 1
        sl.add(b)
        ov[j] = inter / (j + 1)
    return np.arange(1, kmax + 1), ov


def _persist_ranks(series):
    """Descending dense-min ranks (1 = most popular); zeros tie last."""
    return series.rank(method="min", ascending=False)


def _mean_sem(values):
    """mean and SEM (std/sqrt(N), ddof=1) of a 1D array; SEM=0 when N<2."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n == 0:
        return np.nan, np.nan
    m = float(np.mean(a))
    s = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return m, s


def rank_persistence_stats(series, k_values=_PERSIST_K_DEFAULT, null_label="ge"):
    """Per-batch statistics with mean and SEM across batches.

    series : list of (label, batches), where batches is a list of
             (early_snapshot, late_snapshot) pairs, ONE PER BATCH.

    For each series and K: emersion@K is computed PER BATCH (preserving the
    within-batch pairing), then reported as mean and SEM across batches.
    dE@K = mean_emersion(label) - mean_emersion(null) (unpaired difference of
    means, since batches of different recommenders are independent runs), with
    SEM combined in quadrature.  Kendall tau_b likewise per batch -> mean, SEM.
    """
    import scipy.stats as _ss

    per_label_em = {}     # label -> {k: [emersion per batch]}
    rows = {}
    for label, batches in series:
        em_by_k = {k: [] for k in k_values}
        taus = []
        for early, late in batches:
            for k in k_values:
                _, em, keff = overlap_emersion_at_k(early, late, k)
                em_by_k[k].append(em)
                if keff < k:
                    logger.debug(f"[persistence] '{label}': K={k} clipped to {keff}.")
            vis = (early.values > 0) | (late.values > 0)
            if vis.sum() >= 2:
                t, _ = _ss.kendalltau(_persist_ranks(early)[vis], _persist_ranks(late)[vis])
                taus.append(t)
        per_label_em[label] = em_by_k
        rec = {"n_batches": len(batches)}
        for k in k_values:
            m, s = _mean_sem(em_by_k[k])
            rec[f"emersion@{k}"] = m
            rec[f"emersion@{k}_sem"] = s
        tm, tsem = _mean_sem(taus)
        rec["kendall_tau_b"] = tm
        rec["kendall_tau_b_sem"] = tsem
        rows[label] = rec

    df = pd.DataFrame(rows).T
    if null_label in per_label_em:
        for k in k_values:
            nm, ns = _mean_sem(per_label_em[null_label][k])
            for label in rows:
                lm, ls = _mean_sem(per_label_em[label][k])
                df.loc[label, f"dE@{k}"] = lm - nm
                df.loc[label, f"dE@{k}_sem"] = float(np.sqrt(ls ** 2 + ns ** 2))
    return df


def _persist_title(base, transition, filters=None):
    tag = " [cold-start]" if transition == "coldstart" else ""
    try:
        return build_plot_title(base + tag, filters)
    except Exception:
        return base + tag


def plot_retention(series, transition="recommender", filters=None,
                   show=False, name=None, k_marks=_PERSIST_K_DEFAULT):
    """Retention overlap@K curve with a per-series band = mean +/- SEM across
    batches.  Each batch contributes its own overlap@K curve; curves are
    truncated to the minimum common K before aggregating, so the band uses the
    same batch count at every K.
    y (retention value) = overlap@K = |topK(start) ∩ topK(end)| / K, in [0, 1].
    series: list of (label, batches) with batches a list of (early, late) pairs."""
    multi = len(series) > 1
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, (label, batches) in enumerate(series):
        curves = []
        for early, late in batches:
            ks, ov = retention_curve(early, late)
            if ks.size:
                curves.append(ov)
        if not curves:
            continue
        kmax = min(len(ov) for ov in curves)          # min common K
        M = np.vstack([ov[:kmax] for ov in curves])    # n_batches x kmax
        mean = np.nanmean(M, axis=0)
        n = M.shape[0]
        sem = (np.nanstd(M, axis=0, ddof=1) / np.sqrt(n)) if n > 1 else np.zeros(kmax)
        ks = np.arange(1, kmax + 1)
        color = _series_color(idx, multi)
        ax.plot(ks, mean, color=color, lw=1.6,
                label=label if multi else None, zorder=3)
        if n > 1:
            ax.fill_between(ks, mean - sem, mean + sem, color=color,
                            alpha=0.20, lw=0, zorder=2)
    for k in k_marks:
        ax.axvline(k, ls=":", lw=0.8, color="gray", alpha=0.6, zorder=1)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("retention window")
    ax.set_ylabel("retention value")
    ax.set_title(_persist_title("Top-K retention", transition, filters))
    if multi:
        ax.legend(fontsize=8)
    filepath_save_plot_or_show(name=name or f"plot_retention_{transition}", show=show)

# =============================================================================
# TIME-RESOLVED VENUE DYNAMICS
# =============================================================================
# The four plots below all read the per-batch popularity_timeseries.parquet
# artifact (per-cell CUMULATIVE visit counts sampled every POP_SNAPSHOT_STEP
# clock units) and, for the grid heatmaps, reshape it back into physical grid
# matrices.  They obey the same aggregation contract as the rank_persistence
# plots: every metric is computed WITHIN a single run (batch) and the per-run
# curves are then averaged across runs with a mean +/- SEM band.  Visit counts
# of different runs are never mixed.
#
# Series contract (multiplot-aware):
#   * time-series plots  -> series = [(label, [ts_df_batch1, ts_df_batch2, ...]), ...]
#   * overlap plot       -> series = [(label, [(early, late), ...]), ...]  (as retention)

def _ts_matrix(df):
    """Extract (times, cells, V) from a popularity_timeseries DataFrame.

    times : 1D float array of snapshot clocks, ascending.
    cells : list[str] of cell names (row order of V).
    V     : (n_cells, n_times) float array of CUMULATIVE visit counts.
    """
    time_cols = [c for c in df.columns
                 if isinstance(c, str) and c.startswith("t_")]
    time_cols = sorted(time_cols, key=lambda c: int(c[2:]))
    if not time_cols:
        return np.array([]), [], np.zeros((0, 0))
    times = np.array([int(c[2:]) for c in time_cols], dtype=float)
    cells = df["cell"].astype(str).tolist()
    V = df[time_cols].to_numpy(dtype=float)
    return times, cells, V


def _ts_matrix_upto(df, t_max=None):
    """_ts_matrix restricted to snapshots <= t_max (None -> no clipping).

    Users stop at their first jump whose clock exceeds the target t1, so the
    max clock overshoots t1 by up to one waiting time (temporal_max).  The
    snapshots beyond t1 only hold those few residual jumps: the new plots clip
    there so that "end of run" really means t1."""
    times, cells, V = _ts_matrix(df)
    if t_max is None or times.size == 0:
        return times, cells, V
    keep = times <= float(t_max)
    return times[keep], cells, V[:, keep]


def _t1_of(filters):
    try:
        return float((filters or {}).get("t1"))
    except (TypeError, ValueError):
        return None


def _cell_name_to_coords(name):
    """Parse a 'Cell_(cx,cy)' name into (cx, cy) ints; None if it doesn't match."""
    m = re.match(r"^Cell_\((\d+),\s*(\d+)\)$", str(name))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _batch_band(curves):
    """Aggregate a list of per-batch 1D curves into (kmax, mean, sem, n).

    Curves are truncated to the shortest common length before aggregating, so
    the band rests on the same batch count at every x (mirrors plot_retention).
    NaNs are ignored per column.  Returns None when there is no data.
    """
    curves = [np.asarray(c, dtype=float) for c in curves
              if c is not None and len(c) > 0]
    if not curves:
        return None
    kmax = min(len(c) for c in curves)
    if kmax <= 0:
        return None
    M = np.vstack([c[:kmax] for c in curves])          # (n_batches, kmax)
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(M, axis=0)
        n = np.sum(np.isfinite(M), axis=0)
        std = np.nanstd(M, axis=0, ddof=1)
        sem = np.where(n > 1, std / np.sqrt(np.maximum(n, 1)), 0.0)
    return kmax, mean, sem, M.shape[0]


def _draw_t0_line(ax, filters):
    """Vertical dashed marker at the baseline/recommender boundary t0, if known."""
    if not filters:
        return
    t0 = filters.get("t0")
    try:
        t0 = float(t0)
    except (TypeError, ValueError):
        return
    if np.isfinite(t0) and t0 > 0:
        ax.axvline(t0, ls="--", lw=1.0, color="gray", alpha=0.7, zorder=1)
        ax.text(t0, 0.98, r" $t_0$", color="gray", fontsize=FONT_T0,
                ha="left", va="top", transform=ax.get_xaxis_transform(),
                gid="t0_label")


def _color_of(label, idx):
    """Fixed colour of a recommender label (SERIES_COLORS), else palette by index."""
    return SERIES_COLORS.get(label, _MP_COLORS[idx % len(_MP_COLORS)])


def _fit_ylim(ax, arrays, pad=0.06, floor=None, ceil=None):
    """Set the y-limits to the range covered by `arrays` (curves and band
    edges) plus a small margin, clipped only at the theoretical bounds, so
    that each figure is as spread out as its data allow."""
    vals = [np.asarray(a, dtype=float).ravel() for a in arrays if a is not None]
    vals = np.concatenate(vals) if vals else np.array([])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo
    if span <= 0:
        span = abs(hi) if hi != 0 else 1.0
    lo -= pad * span
    hi += pad * span
    if floor is not None:
        lo = max(lo, floor)
    if ceil is not None:
        hi = min(hi, ceil)
    if hi <= lo:
        hi = lo + 1e-9
    ax.set_ylim(lo, hi)


def _zero_line(ax):
    """Dotted horizontal line at y = 0, only if 0 is inside the y-range."""
    lo, hi = ax.get_ylim()
    if lo < 0 < hi:
        ax.axhline(0, ls=":", lw=0.8, color="gray", alpha=0.6, zorder=1)


def align_cycles(per_series, fingerprint, what="runs"):
    """Keep only the runs whose cycle (baseline) is present in EVERY series.

    per_series  : list of (label, [(batch_id, obj), ...])
    fingerprint : obj -> hashable key identifying the baseline of the run
                  (e.g. t0_fingerprint: the popularity at t0 is identical for
                  all the continuations of the same baseline)
    Returns the same structure restricted to the common cycles, in the same
    order for every series (sorted by the batch id of the first series).
    """
    if len(per_series) < 2:
        return per_series
    maps = []
    for label, runs in per_series:
        d = {}
        for bid, obj in runs:
            try:
                fp = fingerprint(obj)
            except Exception as exc:                 # never break a figure
                logger.warning(f"cycle alignment: fingerprint failed for {label}/{bid}: {exc}")
                fp = None
            if fp is None:
                continue
            if fp in d:
                logger.warning(f"cycle alignment: '{label}' has two runs from the same "
                               f"baseline ({d[fp][0]}, {bid}); keeping the first.")
                continue
            d[fp] = (bid, obj)
        maps.append((label, d))
    common = set.intersection(*(set(d) for _l, d in maps))
    ref = maps[0][1]
    order = sorted(common, key=lambda fp: ref[fp][0])
    for label, d in maps:
        extra = len(d) - len(common)
        if extra:
            logger.warning(f"cycle alignment ({what}): '{label}': {extra} run(s) excluded "
                           f"(their baseline is missing in another series).")
    logger.debug(f"cycle alignment ({what}): {len(common)} common cycle(s) across "
                + ", ".join(label for label, _ in maps))
    return [(label, [d[fp] for fp in order]) for label, d in maps]


def _bands(series, curve_fn):
    """Per-series mean +/- SEM across runs.

    curve_fn(df) -> (x, y) for one run, or (None, None).
    Returns [(label, colour, x, mean, sem, n_runs), ...].
    """
    out = []
    for idx, (label, batches) in enumerate(series):
        curves, x_ref = [], None
        for df in batches:
            x, y = curve_fn(df)
            if x is None or np.size(x) == 0:
                continue
            if x_ref is None or np.size(x) < np.size(x_ref):
                x_ref = np.asarray(x)
            curves.append(y)
        agg = _batch_band(curves)
        if agg is None:
            continue
        kmax, mean, sem, n = agg
        out.append((label, _color_of(label, idx), x_ref[:kmax], mean, sem, n))
    return out


def _draw_points(ax, x, mean, sem, color, n, label=None, ms=4.5, lw=1.4,
                 sem_lo=None, capsize=2.5, err_alpha=1.0):
    """One time curve: a line through the snapshots, a marker on every
    snapshot and the error bar (SEM across runs) on the markers.
    sem_lo: optional lower error (e.g. clipped on a log axis)."""
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    ok = np.isfinite(x) & np.isfinite(mean)
    if not ok.any():
        return
    ax.plot(x[ok], mean[ok], marker="o", ms=ms, lw=lw, color=color,
            label=label, zorder=3)
    if n > 1 and sem is not None:
        sem = np.asarray(sem, dtype=float)
        hi_e = np.where(np.isfinite(sem[ok]), sem[ok], 0.0)
        lo_e = hi_e if sem_lo is None else np.clip(
            np.nan_to_num(np.asarray(sem_lo, dtype=float)[ok]), 0.0, None)
        ax.errorbar(x[ok], mean[ok], yerr=[lo_e, hi_e],
                    fmt="none", ecolor=color, elinewidth=1.1, capsize=capsize,
                    capthick=1.1, alpha=err_alpha, zorder=2)


def _thin_bands(bands, t0, keep_t0=False):
    """Keep one snapshot every TIME_THIN_EVERY (see the config at the top):
    those at an odd multiple of the snapshot step from t0 (for 2), the last
    one and, with keep_t0, t0 itself.  The values are not recomputed."""
    every = int(TIME_THIN_EVERY or 1)
    if every <= 1 or t0 is None:
        return bands
    out = []
    for label, color, x, mean, sem, n in bands:
        x = np.asarray(x, dtype=float)
        if x.size < 3:
            out.append((label, color, x, mean, sem, n))
            continue
        step = float(np.median(np.diff(x)))
        k = np.rint((x - float(t0)) / step).astype(int)
        keep = np.mod(k, every) == (1 % every)
        keep[-1] = True
        if keep_t0:
            keep |= (k == 0)
        out.append((label, color, x[keep], np.asarray(mean)[keep],
                    np.asarray(sem)[keep], n))
    return out


def _draw_bands(ax, bands, legend=True):
    """Draw the curves of _bands() (line + points with error bars); returns
    the arrays used for _fit_ylim."""
    arrays = []
    for label, color, x, mean, sem, n in bands:
        _draw_points(ax, x, mean, sem, color, n, label=label if legend else None)
        arrays += [mean - sem, mean + sem]
    return arrays


# ---------------------------------------------------------------------------
# Feature 1: venue accumulation velocity of sets frozen at the end of the
# baseline (top-N / whole minimum-count block)
# ---------------------------------------------------------------------------

def _ref_index(times, t_ref):
    """Index of the last snapshot <= t_ref (t_ref None -> last snapshot).
    Returns -1 when every snapshot is after t_ref."""
    if times.size == 0:
        return -1
    if t_ref is None:
        return times.size - 1
    return int(np.searchsorted(times, float(t_ref), side="right")) - 1


def _fixed_sets_at(col, top_n):
    """(top_idx, worst_idx) among the VISITED cells (count > 0) of one snapshot
    column.  top_idx is ordered best-first (rank 1, 2, ...); worst_idx
    least-visited first.  Ties are broken by the stable (cell-name) order."""
    order = np.argsort(-col, kind="stable")          # descending
    pos = order[col[order] > 0]
    k = int(min(top_n, pos.size))
    if k == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return pos[:k], pos[::-1][:k]


def _min_block(col):
    """All visited cells (count > 0) having the minimum positive count."""
    pos = np.flatnonzero(col > 0)
    if pos.size == 0:
        return pos
    return pos[col[pos] == col[pos].min()]


def _rank_block(col, ranks):
    """Visited cells (count > 0) with rank ranks[0]..ranks[1] (both included,
    rank 1 = most visited) in one snapshot column; ties broken by the stable
    (cell-name) order, as in _fixed_sets_at."""
    order = np.argsort(-col, kind="stable")
    pos = order[col[order] > 0]
    a, b = int(ranks[0]), int(ranks[1])
    return pos[a - 1:b]


_FINAL_SETS = ("final", "worst_final", "mid_final")


def _velocity_fixed(df, which, top_n, t0, t1):
    """One run -> (interval_times, velocity of the set, set size, count of the
    set at t0) or None.

    The set is frozen at the snapshot <= t0 (end of the baseline):
      which="top"   -> the top-N venues (_fixed_sets_at);
      which="worst" -> the WHOLE block of venues with the minimum positive
                       count (no arbitrary tie-breaking);
    or at the last snapshot <= t1 (end of the run):
      which="final" -> the top-N venues at the end of the run (the winners).
    Per-cell velocity on [t_j, t_{j+1}] = (count_{j+1} - count_j) / (t_{j+1} - t_j),
    averaged over the cells of the set.
    """
    times, _cells, V = _ts_matrix_upto(df, t1)
    if times.size < 2 or V.shape[0] == 0:
        return None
    j0 = _ref_index(times, t0)
    if j0 < 0:
        return None
    col = V[:, times.size - 1] if which in _FINAL_SETS else V[:, j0]
    if which in ("top", "final"):
        idx = _fixed_sets_at(col, top_n)[0]
    elif which in ("mid", "mid_final"):
        idx = _rank_block(col, MID_RANKS)
    else:                                   # "worst", "worst_final"
        idx = _min_block(col)
    vel = np.diff(V, axis=1) / np.diff(times)[None, :]
    it = times[1:]
    cur = np.nanmean(vel[idx, :], axis=0) if idx.size else np.full(it.size, np.nan)
    count = int(col[idx].min()) if idx.size else None
    return it, cur, int(idx.size), count


def plot_venue_velocity_fixed(series, which="top", filters=None, show=False, top_n=10,
                              metric="visits", name=None):
    """Accumulation velocity (per venue) of a set frozen at the end of the
    baseline: which="top" -> top-N, which="worst" -> whole minimum-count block;
    or at the end of the run: which="final" -> top-N at t1 (the final winners;
    each series has its own winners).
    One curve per series (mean +/- SEM across runs).

    series : list of (label, [ts_df per batch]).
    """
    t0 = (filters or {}).get("t0")
    t1 = _t1_of(filters)
    if t0 is None:
        logger.warning("plot_venue_velocity_fixed: no t0 in filters — skipped.")
        return

    def curve(df):
        r = _velocity_fixed(df, which, top_n, t0, t1)
        return (None, None) if r is None else (r[0], r[1])

    bands = _bands(series, curve)
    if not bands:
        logger.warning(f"plot_venue_velocity_fixed({which}): no data.")
        return

    # Size of the set, from the first series.
    ref_batches = dict(series)[bands[0][0]]
    x_all, sizes, counts = None, [], []
    for df in ref_batches:
        r = _velocity_fixed(df, which, top_n, t0, t1)
        if r is None:
            continue
        it, _cur, size, cnt = r
        if x_all is None or it.size < x_all.size:
            x_all = it
        sizes.append(size)
        counts.append(cnt)

    # Velocity per day for long runs (new parameters), per second otherwise.
    days = _use_days(t1 if t1 is not None else (x_all[-1] if x_all is not None else 0))
    fac = _DAY if days else 1.0
    bands = [(l, c, x, m * fac, e * fac, n) for (l, c, x, m, e, n) in bands]

    fig, ax = plt.subplots(figsize=PANEL_RECT)
    arrays = _draw_bands(ax, bands, legend=True)
    _draw_t0_line(ax, filters)
    _fit_ylim(ax, arrays)
    unit = "explorers" if metric == "explorers" else "visits"
    _time_axis(ax)
    ax.set_ylabel(rf"$\nu_A$ [{unit} / {'day' if days else 's'}]")
    ax.grid(True, ls="--", alpha=0.4)
    if which in VELOCITY_LEGEND_ABOVE:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + VELOCITY_LEGEND_HEADROOM * (hi - lo))
        ax.legend(ncol=2, loc="upper center").set_zorder(20)
    else:
        ax.legend()

    pop = r"$n^{\mathrm{E}}$" if metric == "explorers" else r"$n^{\mathrm{V}}$"
    t_txt = r"t_{\mathrm{end}}" if which in _FINAL_SETS else "t_0"
    if which in ("top", "final"):
        set_txt = rf"head set $H_{{{top_n}}}({t_txt})$"
    elif which in ("mid", "mid_final"):
        set_txt = rf"ranks {MID_RANKS[0]}–{MID_RANKS[1]} at ${t_txt}$"
    else:
        s = np.asarray(sizes, dtype=float)
        size_txt = f"{s.mean():.0f}" + (f" ± {s.std(ddof=1):.0f}" if s.size > 1 else "")
        cnt_txt = "/".join(str(c) for c in sorted({c for c in counts if c is not None}))
        logger.debug(f"velocity, bottom set at {t_txt} ({metric}): count = {cnt_txt}, "
                    f"{size_txt} venues per run")
        set_txt = rf"bottom set $B({t_txt})$"
    ax.set_title(f"Accumulation velocity, {set_txt} ({pop})")
    filepath_save_plot_or_show(name=name or (f"plot_venue_velocity_fixed_{which}"
                               + (f"{MID_RANKS[0]}-{MID_RANKS[1]}" if which.startswith("mid") else "")
                               + f"_{metric}"),
                               show=show)


# ---------------------------------------------------------------------------
# Feature 2: top-N stability / turnover over time
# ---------------------------------------------------------------------------

def _turnover_curve(df, top_n, t_max=None):
    """Number of venues entering the top-N between consecutive snapshots.

    turnover_j = |topN(t_{j+1}) \\ topN(t_j)| (number of new entrants, 0..N),
    where topN is by cumulative counts (visited cells only).  Evaluated at the
    interval-end times.  Returns (interval_times, turnover_curve).
    """
    times, cells, V = _ts_matrix_upto(df, t_max)
    if times.size < 2 or V.shape[0] == 0:
        return np.array([]), np.array([])
    n_int = times.size - 1
    it = times[1:]
    turnover = np.full(n_int, np.nan)

    def _topset(col):
        order = np.argsort(col, kind="stable")
        pos = order[col[order] > 0]
        k = int(min(top_n, pos.size))
        return set(pos[-k:].tolist()) if k > 0 else set()

    prev = _topset(V[:, 0])
    for j in range(n_int):
        cur = _topset(V[:, j + 1])
        if prev or cur:
            turnover[j] = float(len(cur - prev))
        prev = cur
    return it, turnover


def plot_top10_turnover(series, filters=None, show=False, top_n=10, name=None,
                        metric="visits"):
    """Top-N turnover over time: venues entering the top-N between consecutive
    snapshots (higher = less stable top-N).  One curve per series with a
    mean +/- SEM band across runs; y-range fitted to the data.

    series : list of (label, [ts_df per batch]).
    """
    t1 = _t1_of(filters)
    bands = _bands(series, lambda df: _turnover_curve(df, top_n, t_max=t1))
    if not bands:
        logger.warning("plot_top10_turnover: no data to plot.")
        return
    bands = _thin_bands(bands, (filters or {}).get("t0"))
    multi = len(series) > 1
    fig, ax = plt.subplots(figsize=PANEL_RECT)
    arrays = _draw_bands(ax, bands, legend=multi)
    _draw_t0_line(ax, filters)
    _fit_ylim(ax, arrays)
    _time_axis(ax)
    ax.set_ylabel(r"$c$")
    ax.grid(True, ls="--", alpha=0.4)
    if multi:
        ax.legend()
    pop = r"$n^{\mathrm{E}}$" if metric == "explorers" else r"$n^{\mathrm{V}}$"
    ax.set_title(rf"Turnover of the head set $H_{{{top_n}}}$ ({pop})")
    filepath_save_plot_or_show(name=name or f"plot_top{top_n}_turnover_{metric}",
                               show=show)


# ---------------------------------------------------------------------------
# Feature 4b: Kendall tau of the top-N between consecutive snapshots over time
# ---------------------------------------------------------------------------

def _kendall_over_time_curve(df, top_n, t_max=None):
    """Kendall tau_b between consecutive snapshots, restricted to the union of
    their top-N venues.  Evaluated at the interval-end times.
    Returns (interval_times, tau_curve)."""
    times, cells, V = _ts_matrix_upto(df, t_max)
    if times.size < 2 or V.shape[0] == 0:
        return np.array([]), np.array([])
    n_int = times.size - 1
    it = times[1:]
    tau = np.full(n_int, np.nan)

    def _topidx(col):
        order = np.argsort(col, kind="stable")
        pos = order[col[order] > 0]
        k = int(min(top_n, pos.size))
        return pos[-k:] if k > 0 else np.array([], dtype=int)

    for j in range(n_int):
        a = V[:, j]; b = V[:, j + 1]
        union = np.union1d(_topidx(a), _topidx(b))
        if union.size < 2:
            continue
        # Ranks within the union (descending: 1 = most visited); ties averaged.
        ra = pd.Series(a[union]).rank(method="average", ascending=False).to_numpy()
        rb = pd.Series(b[union]).rank(method="average", ascending=False).to_numpy()
        t, _ = scipy.stats.kendalltau(ra, rb)
        tau[j] = t
    return it, tau


def plot_kendall_over_time(series, filters=None, show=False, top_n=10, name=None,
                           metric="visits"):
    """Kendall tau_b of the top-N venues between consecutive snapshots, over
    time.  tau near 1 = stable ordering across the interval; low/negative =
    reshuffling.  One curve per series with a mean +/- SEM band across runs;
    y-range fitted to the data (within [-1, 1]).

    series : list of (label, [ts_df per batch]).
    """
    t1 = _t1_of(filters)
    bands = _bands(series, lambda df: _kendall_over_time_curve(df, top_n, t_max=t1))
    if not bands:
        logger.warning("plot_kendall_over_time: no data to plot.")
        return
    bands = _thin_bands(bands, (filters or {}).get("t0"))
    multi = len(series) > 1
    fig, ax = plt.subplots(figsize=PANEL_RECT)
    arrays = _draw_bands(ax, bands, legend=multi)
    _draw_t0_line(ax, filters)
    _fit_ylim(ax, arrays, floor=-1.02, ceil=1.02)
    _zero_line(ax)
    _time_axis(ax)
    ax.set_ylabel(r"Kendall's $\tau_b$")
    ax.grid(True, ls="--", alpha=0.4)
    if multi:
        ax.legend()
    pop = r"$n^{\mathrm{E}}$" if metric == "explorers" else r"$n^{\mathrm{V}}$"
    ax.set_title(rf"Kendall $\tau_b$ of the head set $H_{{{top_n}}}$ ({pop})")
    filepath_save_plot_or_show(name=name or f"plot_kendall_over_time_top{top_n}_{metric}",
                               show=show)


def _plot_time_window(series, curve_fn, filters, show, window, ylabel, title,
                      name, ylim=None):
    """Common body of the zoomed time plots.  curve_fn(df, t_max) -> (x, y)
    for one run; the window is (start, end) in days; a point at time t
    refers to the interval (t - delta t, t], so the points with
    start < t <= end are drawn, all of them.  The axis is always in days."""
    lo, hi = float(window[0]) * _DAY, float(window[1]) * _DAY

    def curve(df):
        x, y = curve_fn(df, hi)
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return None, None
        keep = (x > lo + 1e-6) & (x <= hi + 1e-6)
        if not keep.any():
            return None, None
        return x[keep], np.asarray(y)[keep]

    bands = _bands(series, curve)
    if not bands:
        logger.warning(f"{name}: no data in the window {window}.")
        return
    multi = len(series) > 1
    fig, ax = plt.subplots(figsize=PANEL_RECT)
    arrays = _draw_bands(ax, bands, legend=multi)
    t0 = (filters or {}).get("t0")
    try:
        if t0 is not None and lo < float(t0) < hi:
            _draw_t0_line(ax, filters)
    except (TypeError, ValueError):
        pass
    if ylim is not None:
        _fit_ylim(ax, arrays, floor=ylim[0], ceil=ylim[1])
        _zero_line(ax)
    else:
        _fit_ylim(ax, arrays)
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    span_days = (hi - lo) / _DAY
    step = next((st for st in (1, 2, 4, 8, 16) if span_days / st <= 8), 32)
    ax.set_xlim(lo, hi + 0.3 * _DAY)
    ax.xaxis.set_major_locator(MultipleLocator(step * _DAY))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / _DAY:g}"))
    ax.set_xlabel(r"$t$ [days]")
    ax.set_ylabel(ylabel)
    ax.grid(True, ls="--", alpha=0.4)
    if multi:
        ax.legend()
    a, b = f"{window[0]:g}", f"{window[1]:g}"
    # Same title as the full plot: a longer one would stick out of the axes and,
    # with bbox_inches="tight", make the PDF wider than the full plot, so that
    # side by side at the same width in LaTeX it would come out smaller.  The
    # window is given by the axis (and by the caption).
    ax.set_title(title)
    filepath_save_plot_or_show(name=f"{name}_days{a}-{b}", show=show)


def plot_kendall_over_time_window(series, filters=None, show=False, top_n=10,
                                  metric="explorers", window=None, name=None):
    """plot_kendall_over_time restricted to a time window (default ZOOM_WINDOW),
    every snapshot drawn."""
    window = window or ZOOM_WINDOW
    if not window:
        return
    pop = r"$n^{\mathrm{E}}$" if metric == "explorers" else r"$n^{\mathrm{V}}$"
    _plot_time_window(
        series, lambda df, t_max: _kendall_over_time_curve(df, top_n, t_max=t_max),
        filters, show, window, ylabel=r"Kendall's $\tau_b$",
        title=rf"Kendall $\tau_b$ of the head set $H_{{{top_n}}}$ ({pop})",
        name=name or f"plot_kendall_over_time_top{top_n}_{metric}", ylim=(-1.02, 1.02))


def plot_top10_turnover_window(series, filters=None, show=False, top_n=10,
                               metric="explorers", window=None, name=None):
    """plot_top10_turnover restricted to a time window (default ZOOM_WINDOW),
    every snapshot drawn."""
    window = window or ZOOM_WINDOW
    if not window:
        return
    pop = r"$n^{\mathrm{E}}$" if metric == "explorers" else r"$n^{\mathrm{V}}$"
    _plot_time_window(
        series, lambda df, t_max: _turnover_curve(df, top_n, t_max=t_max),
        filters, show, window, ylabel=r"$c$",
        title=rf"Turnover of the head set $H_{{{top_n}}}$ ({pop})",
        name=name or f"plot_top{top_n}_turnover_{metric}")


# ---------------------------------------------------------------------------
# Feature 3: aggregate physical-grid visit heatmaps at time snapshots
# ---------------------------------------------------------------------------

def timeseries_to_grid_snapshots(ts_batches, cell_number, t_max=None):
    """Per-run grids at every snapshot time, then MEAN across runs.

    Each run is turned into its own (cell_number x cell_number) grid at each
    snapshot (cells mapped back via their 'Cell_(cx,cy)' name), and only then
    are the runs averaged: counts of different runs are never pooled before
    the per-run grid exists, coherently with every other time-resolved plot.
    Only snapshot times present in EVERY run and <= t_max (t1) are kept.

    Returns a list of (t, mean_matrix[cx, cy]) sorted by t; [] if nothing usable.
    """
    per_run = []                      # [{t: matrix}]
    common = None
    for df in ts_batches:
        times, cells, V = _ts_matrix_upto(df, t_max)
        if times.size == 0:
            continue
        per_run.append({float(t): _run_grid(cells, V[:, j], cell_number)
                        for j, t in enumerate(times)})
        ts = set(float(t) for t in times)
        common = ts if common is None else common & ts
    if not per_run or not common:
        return []
    return [(t, np.mean([r[t] for r in per_run], axis=0)) for t in sorted(common)]


@_each_grid_cmap
def plot_aggregate_grid_snapshots(ts_batches, cell_number, filters=None,
                                  show=False, mask_center=False, metric="visits",
                                  subfolder=None, cbar_label=None):
    """Physical-grid show: one heatmap per time snapshot (every
    POP_SNAPSHOT_STEP clock, up to t1) of the cumulative visits (or distinct
    explorers) of ALL users, MEAN over runs.  Shared colour scale across the
    snapshots, so the temporal growth is visible.  Written to a dedicated
    subfolder of the plots folder.  NOT a multiplot (per dataset).
    """
    snaps = timeseries_to_grid_snapshots(ts_batches, cell_number,
                                         t_max=_t1_of(filters))
    if not snaps:
        logger.warning("plot_aggregate_grid_snapshots: no snapshots to plot.")
        return
    subfolder = subfolder or f"aggregate_grid_snapshots_{metric}"
    cbar_label = cbar_label or (
        "Explorers" if metric == "explorers" else "Visits")

    vmax = max(1e-12, max(float(m.max()) for _, m in snaps))
    cx = int(cell_number) // 2
    cy = int(cell_number) // 2
    for t, m in snaps:
        m_masked = np.ma.masked_where(m <= 0, m)
        if mask_center:
            m_masked = np.ma.array(m_masked, mask=np.ma.getmaskarray(m_masked).copy())
            m_masked.mask[cx, cy] = True
        fig, ax = plt.subplots(figsize=PANEL_SQUARE)
        im = ax.imshow(m_masked.T, origin="lower", aspect="equal", cmap=_grid_cmap(),
                       vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
        _bare_map_axes(ax)
        t_txt = (f"{t / _DAY:g} days" if _use_days(_t1_of(filters) or t)
                 else f"{int(t)} s")
        ax.set_title(f"Grid at $t$ = {t_txt}")
        filepath_save_plot_or_show(name=f"grid_t_{int(t)}",
                                   show=show,
                                   subfolder=subfolder)


# =============================================================================
# FIXED-SET RANK TRACKING (top-N frozen at end of baseline / at end of run)
# =============================================================================
# Conceptually different from plot_retention: retention compares two snapshots
# as a function of the window K.  Here the SET is frozen once (the top-N venues
# at a reference instant) and each of its N members is followed through time,
# reading its position in the full ranking at every snapshot.
#
#   ref="t0"    -> the top-N at the end of the grid-EPR baseline: where do the
#                  "old champions" end up once the recommender is switched on?
#   ref="final" -> the top-N at the end of the run (chosen a posteriori): where
#                  were the "final champions" before, and when did they emerge?
#
# Aggregation: only positions matter, never venue identities.  In every run
# the curves are indexed by the member's ORIGINAL POSITION (#1 ... #N at the
# reference instant), and curve #p is the arithmetic mean across runs of
# "the venue that was p-th", with a +/- SEM band.
# -----------------------------------------------------------------------------

def _rank_matrix(V):
    """Descending 'min' ranks (1 = most visited) per snapshot column.
    V: (n_cells, n_times) -> R of the same shape (float)."""
    R = np.empty_like(V, dtype=float)
    for j in range(V.shape[1]):
        R[:, j] = scipy.stats.rankdata(-V[:, j], method="min")
    return R


def _fixed_set_rank_curves(df, top_n, t_ref, t_max=None):
    """For one batch: (times, R_set) where R_set is (top_n, n_times) holding the
    rank over time of the top-N venues at the snapshot <= t_ref, rows ordered
    by their position at the reference (#1 first).  Missing rows are NaN."""
    times, cells, V = _ts_matrix_upto(df, t_max)
    if times.size == 0 or V.shape[0] == 0:
        return np.array([]), None
    j_ref = _ref_index(times, t_ref)
    if j_ref < 0:
        return np.array([]), None
    top_idx, _ = _fixed_sets_at(V[:, j_ref], top_n)
    out = np.full((top_n, times.size), np.nan)
    if top_idx.size:
        R = _rank_matrix(V)
        out[:top_idx.size, :] = R[top_idx, :]
    return times, out


def plot_fixed_set_rank_tracking(series, ref="t0", filters=None, show=False,
                                 top_n=10, metric="visits", name=None):
    """Rank over time of a FROZEN top-N set, one panel per series.

    series : list of (label, [ts_df per batch]) -- popularity_timeseries
             (metric="visits") or exploration_timeseries (metric="explorers").
    ref    : "t0" (end of baseline, read from filters["t0"]) or "final".

    Two files are written:
      <base>_ranks    : per series, the N trajectories (colour = position at
                        the reference), log-y inverted (rank 1 on top), with the
                        top-N boundary drawn as a dashed line;
      <base>_intopN   : fraction of the frozen set still/already inside the
                        current top-N, mean +/- SEM across runs, all series
                        overlaid.
    Both y-ranges are fitted to the data.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    t_ref = (filters or {}).get("t0") if ref == "t0" else None
    if ref == "t0" and t_ref is None:
        logger.warning("plot_fixed_set_rank_tracking: no t0 in filters — skipped.")
        return
    multi = len(series) > 1

    prepared = []     # (label, color, times, [R_set per batch])
    for idx, (label, batches) in enumerate(series):
        per_batch, t_common = [], None
        for df in batches:
            times, R = _fixed_set_rank_curves(df, top_n, t_ref, t_max=_t1_of(filters))
            if R is None:
                continue
            if t_common is None or times.size < t_common.size:
                t_common = times
            per_batch.append(R)
        if per_batch:
            m = t_common.size
            prepared.append((label, _color_of(label, idx), t_common,
                             [R[:, :m] for R in per_batch]))
    if not prepared:
        logger.warning(f"plot_fixed_set_rank_tracking({ref}): no data.")
        return

    ref_txt = r"$t_0$" if ref == "t0" else r"$t_{\mathrm{end}}$"
    pop = r"$n^{\mathrm{E}}$" if metric == "explorers" else r"$n^{\mathrm{V}}$"
    base = name or f"plot_top{top_n}_rank_tracking_{ref}_{metric}"

    # ---------------- figure 1: per-series panels ----------------
    n = len(prepared)
    ncols = n if n <= 3 else 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows),
                             sharey=True, squeeze=False)
    cmap = plt.cm.viridis
    pos_colors = [cmap(i / max(1, top_n - 1)) for i in range(top_n)]
    rmin, rmax = np.inf, 1.0
    for k, (label, _c, times, Rs) in enumerate(prepared):
        ax = axes[k // ncols][k % ncols]
        stack = np.stack(Rs)                         # (n_batches, top_n, n_t)
        for p in range(top_n):
            agg = _batch_band(list(stack[:, p, :]))
            if agg is None or not np.isfinite(agg[1]).any():
                continue
            kk, center, sem, nb = agg
            x = times[:kk]
            lo = np.maximum(center - sem, 1.0)      # ranks are >= 1 (log axis)
            hi = center + sem
            rmax = max(rmax, float(np.nanmax(hi)))
            rmin = min(rmin, float(np.nanmin(lo)))
            # error bar clipped at rank 1 (log axis)
            _draw_points(ax, x, center, sem, pos_colors[p], nb, ms=3.0, lw=1.2,
                         sem_lo=np.minimum(sem, center - 1.0), capsize=0,
                         err_alpha=0.5)
        ax.axhline(top_n + 0.5, ls="--", lw=1.2, color="crimson", alpha=0.7)
        _draw_t0_line(ax, filters)
        ax.set_yscale("log")
        ax.set_title(label or "series")
        _time_axis(ax)
        ax.grid(True, which="both", ls="--", alpha=0.3)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    for r in range(nrows):
        axes[r][0].set_ylabel(r"Rank $r_i(t)$")
    if not np.isfinite(rmin):
        rmin = 1.0
    axes[0][0].set_ylim(rmax * 1.15, max(rmin / 1.15, 0.5))   # inverted: rank 1 on top
    for row in axes:
        for ax in row:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    handles = [Line2D([0], [0], color=pos_colors[p], lw=2, label=f"#{p + 1}")
               for p in range(top_n)]
    handles.append(Line2D([0], [0], color="crimson", ls="--", lw=1,
                          label=rf"$H_{{{top_n}}}$ boundary"))
    # Legend BELOW the panels: tight_layout ignores figure legends, and the
    # saver uses bbox_inches="tight", so it is kept without squeezing panels.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=min(len(handles), 6),
               title=f"position at {ref_txt}")
    fig.suptitle(rf"Rank of the head set $H_{{{top_n}}}$ at {ref_txt} ({pop})")
    filepath_save_plot_or_show(name=f"{base}_ranks", show=show)

    # ---------------- figure 2: fraction inside current top-N ----------------
    bands = []
    for label, color, times, Rs in prepared:
        agg = _batch_band([np.nanmean(R <= top_n, axis=0) for R in Rs])
        if agg is None:
            continue
        kmax, mean, sem, nb = agg
        bands.append((label, color, times[:kmax], mean, sem, nb))
    if not bands:
        return
    bands = _thin_bands(bands, (filters or {}).get("t0"), keep_t0=True)
    fig, ax = plt.subplots(figsize=PANEL_RECT)
    arrays = _draw_bands(ax, bands, legend=multi)
    _draw_t0_line(ax, filters)
    _fit_ylim(ax, arrays, floor=-0.02, ceil=1.02)
    _time_axis(ax)
    ax.set_ylabel(r"Membership at $t_0$" if ref == "t0" else r"Membership at $t_{\mathrm{end}}$")
    ax.grid(True, ls="--", alpha=0.4)
    if multi:
        ax.legend()
    ax.set_title(rf"Membership of the head set $H_{{{top_n}}}$ at {ref_txt} ({pop})")
    filepath_save_plot_or_show(name=f"{base}_in_top{top_n}", show=show)


def _membership_after_t0(df, ref, top_n, t0, t1):
    """One run, t >= t0: fraction of the frozen top-N (at t0 or at the end of
    the run) that belongs to the current top-N.

    The current top-N is selected with the SAME rule as the frozen one
    (_fixed_sets_at: exactly N venues), so the overlap is |A ∩ B| / N in both
    cases and the end value of panel (a) equals the start value of panel (b)
    exactly, run by run.  Returns (times, fraction) or (None, None).
    """
    times, _cells, V = _ts_matrix_upto(df, t1)
    j0 = _ref_index(times, t0)
    jf = times.size - 1
    if j0 < 0 or jf <= j0:
        return None, None
    frozen = _fixed_sets_at(V[:, j0 if ref == "t0" else jf], top_n)[0]
    if frozen.size == 0:
        return None, None
    frozen = set(frozen.tolist())
    frac = [len(frozen & set(_fixed_sets_at(V[:, j], top_n)[0].tolist())) / len(frozen)
            for j in range(j0, times.size)]
    return times[j0:], np.asarray(frac, dtype=float)


def plot_top10_membership_panels(series, filters=None, show=False, top_n=10,
                                 metric="visits", name=None):
    """Two panels on the same y axis, from t0 to the end of the run:
      (a) exit  — fraction of the top-N at t0 still inside the current top-N;
      (b) entry — fraction of the final top-N already inside the current top-N.
    The end value of (a) equals the start value of (b): both are the overlap
    between the top-N at t0 and the top-N at the end of the run.

    series : list of (label, [ts_df per batch]).
    """
    t0 = (filters or {}).get("t0")
    t1 = _t1_of(filters)
    if t0 is None:
        logger.warning("plot_top10_membership_panels: no t0 in filters — skipped.")
        return
    panels = (("t0", rf"(a) $H_{{{top_n}}}(t_0)$"),
              ("final", rf"(b) $H_{{{top_n}}}(t_{{\mathrm{{end}}}})$"))
    fig, axes = plt.subplots(1, 2, figsize=(2 * PANEL_RECT[0], PANEL_RECT[1]), sharey=True)
    arrays, drew = [], False
    for ax, (ref, title) in zip(axes, panels):
        bands = _bands(series, lambda df, r=ref: _membership_after_t0(df, r, top_n, t0, t1))
        bands = _thin_bands(bands, t0, keep_t0=True)
        arrays += _draw_bands(ax, bands, legend=True)
        drew = drew or bool(bands)
        ax.set_title(title)
        ax.grid(True, ls="--", alpha=0.4)
    if not drew:
        plt.close(fig)
        logger.warning("plot_top10_membership_panels: no data.")
        return
    _fit_ylim(axes[0], arrays, floor=-0.02, ceil=1.02)
    for ax in axes:
        _time_axis(ax)
    axes[0].set_ylabel(r"Membership $M_{t_0}(t)$")
    axes[1].set_ylabel(r"Membership $M_{t_{\mathrm{end}}}(t)$")
    axes[1].tick_params(labelleft=True)
    if len(series) > 1:
        axes[0].legend()
    pop = r"$n^{\mathrm{E}}$" if metric == "explorers" else r"$n^{\mathrm{V}}$"
    fig.suptitle(rf"Membership of the head set $H_{{{top_n}}}$ ({pop})")
    filepath_save_plot_or_show(name=name or f"plot_top{top_n}_membership_panels_{metric}",
                               show=show)


# =============================================================================
# EFFECTIVE PHYSICAL GRIDS: end-of-run comparison of 4 series (linear scale)
# =============================================================================
# "Effective grid" = popularity of every cell given by ALL users, built from
# the per-batch time series (visits or distinct explorers).  Each batch is
# turned into a grid at the requested snapshot and NORMALISED to a share
# (cell / total), then the shares are averaged across batches.  Normalising is
# what makes t0 (short baseline) and t1 (whole run) visually comparable, and
# keeps series with a different number of runs on the same footing.
# -----------------------------------------------------------------------------

def _run_grid(cells, col, cell_number):
    """One run's snapshot column -> (K, K) matrix indexed [cx, cy]."""
    K = int(cell_number)
    m = np.zeros((K, K), dtype=float)
    for c_idx, name in enumerate(cells):
        cc = _cell_name_to_coords(name)
        if cc is None:
            continue
        cx, cy = cc
        if 0 <= cx < K and 0 <= cy < K:
            m[cx, cy] += col[c_idx]
    return m


def grid_at(ts_dfs, cell_number, ref="final", t_ref=None, t_max=None):
    """Mean (across runs) popularity grid (counts) at one snapshot.

    ref="final" -> last snapshot <= t_max (t1) of each run; ref="t0" ->
    last snapshot <= t_ref.  Returns (matrix, n_runs, t_used) or (None, 0, None).
    """
    acc, nb, t_used = None, 0, None
    for df in ts_dfs:
        times, cells, V = _ts_matrix_upto(df, t_max)
        j = _ref_index(times, None if ref == "final" else t_ref)
        if j < 0:
            continue
        m = _run_grid(cells, V[:, j], cell_number)
        tot = m.sum()
        if tot <= 0:
            continue
        acc = m if acc is None else acc + m
        nb += 1
        t_used = times[j] if t_used is None else min(t_used, times[j])
    if nb == 0:
        return None, 0, None
    return acc / nb, nb, t_used


@_each_grid_cmap
def plot_grid_final_comparison(series, cell_number, filters=None, show=False,
                               metric="visits", mask_center=False,
                               name_suffix="", run_notes=None):
    """End-of-run effective grids of 2 to 6 series (1x2, 1x3, 2x2 or
    GRID_LAYOUT_6 = 3x2 figure),
    linear colour scale shared by all the panels.
    series    : list of (label, [ts_df per batch]).
    run_notes : optional {label: text} replacing "(n runs)" in the subtitles
                (used by the single-run version to show the run id)."""
    from matplotlib.colors import Normalize

    panels = []
    single = True
    for label, ts_dfs in series:
        g, nb, _ = grid_at(ts_dfs, cell_number, ref="final", t_max=_t1_of(filters))
        if g is not None:
            single = single and nb == 1
            note = (run_notes or {}).get(label)
            if note:
                logger.debug(f"grid_final_comparison: {label} -> {note}")
            panels.append((label, g))
    if not 2 <= len(panels) <= 6:
        logger.warning(f"plot_grid_final_comparison: {len(panels)} grid(s) with data, "
                       f"2 to 6 needed — skipped.")
        return
    vmax = max(float(m.max()) for _s, m in panels)
    if vmax <= 0:
        logger.warning("plot_grid_final_comparison: all grids empty — skipped.")
        return
    cmap = _grid_cmap()
    norm = Normalize(vmin=0.0, vmax=vmax)
    if len(panels) == 4:
        nrows, ncols = 2, 2
    elif len(panels) >= 5:
        nrows, ncols = GRID_LAYOUT_6
    else:
        nrows, ncols = 1, len(panels)
    # Maps in an nrows x ncols grid plus a narrow last column holding ONE
    # colourbar: the scale (same norm) is common to all the panels.
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(PANEL_SQUARE[0] * ncols + COLORBAR_EXTRA,
                              PANEL_SQUARE[1] * nrows))
    gs = GridSpec(nrows, ncols + 1, figure=fig, width_ratios=[1] * ncols + [0.06])
    axes = [[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)]
    im = None
    for k, (sub, m) in enumerate(panels):
        ax = axes[k // ncols][k % ncols]
        mm = np.ma.masked_where(m <= 0, m)
        if mask_center:
            mm = np.ma.array(mm, mask=np.ma.getmaskarray(mm).copy())
            mm.mask[m.shape[0] // 2, m.shape[1] // 2] = True
        im = ax.imshow(mm.T, origin="lower", aspect="equal", cmap=cmap, norm=norm)
        ax.set_title(sub)
        _bare_map_axes(ax)
    for k in range(len(panels), nrows * ncols):          # empty cells (5 panels)
        axes[k // ncols][k % ncols].axis("off")
    cax = fig.add_subplot(gs[:, ncols])
    sym = r"$n^{\mathrm{E}}$" if metric == "explorers" else r"$n^{\mathrm{V}}$"
    name = "Exploration popularity" if metric == "explorers" else "Visit popularity"
    fig.colorbar(im, cax=cax, label=sym)
    fig.suptitle(rf"{name} at $t_{{\mathrm{{end}}}}$" + (", single run" if single else ""))
    fig._thesis_shared_cbar = (nrows, ncols)
    filepath_save_plot_or_show(name=f"plot_grid_final_comparison_{metric}{name_suffix}",
                               show=show)


# =============================================================================
# BATCH STATISTIC AS A FUNCTION OF THE NUMBER OF LATENT FACTORS OF MF
# =============================================================================

def _batch_metric_values(batch_stats_list, metric):
    """One value per run of a batch_stats metric (runs without it skipped)."""
    vals = []
    for _bid, df in batch_stats_list or []:
        if df is None or "metric" not in df.columns:
            continue
        v = df.loc[df["metric"].astype(str) == metric, "value"]
        if len(v):
            try:
                vals.append(float(v.iloc[0]))
            except (TypeError, ValueError):
                pass
    return np.asarray([x for x in vals if np.isfinite(x)], dtype=float)


# Titles of the "statistic vs parameter" plots: the name of the statistic (the
# y axis has its symbol).
METRIC_TITLES = {
    "I": "Moran's $I$",
    "gini_collective": "Collective Gini coefficient",
    "mean_rg": "Radius of gyration",
    "mean_rg_explorative": "Exploration radius of gyration",
    "mean_travel_distance": "Mean travel distance",
}


def plot_metric_summary(series_stats, metric, ylabel, filters=None, show=False,
                        scale=1.0):
    """A batch statistic (e.g. the collective Gini coefficient), one point per
    series: mean over the runs (one value per run, from batch_stats) with an
    error bar of +/- one standard deviation ACROSS the runs.  The x axis lists
    the series in the given order (g-EPR, User-kNN, Item-kNN, MF).

    series_stats : list of (label, batch_stats_list)."""
    pts = []
    for label, bsl in series_stats:
        v = _batch_metric_values(bsl, metric) * scale
        if v.size == 0:
            logger.warning(f"plot_metric_summary({metric}): no values for '{label}'.")
            continue
        sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
        pts.append((label, float(v.mean()), sd, v.size))
    if not pts:
        logger.warning(f"plot_metric_summary({metric}): nothing to plot.")
        return
    logger.debug(f"{metric} (mean, std, runs): " +
                "; ".join(f"{lb} {m:.4f} {sd:.4f} {n}" for lb, m, sd, n in pts))

    fig, ax = plt.subplots(figsize=PANEL_RECT)
    for i, (label, m, sd, _n) in enumerate(pts):
        ax.errorbar(i, m, yerr=sd, marker="o", ms=8, lw=1.8, capsize=6,
                    capthick=1.8, ls="none", color=_color_of(label, i), zorder=3)
    ax.set_xticks(range(len(pts)))
    ax.set_xticklabels([_pretty_label(lb) for lb, *_r in pts])
    ax.set_xlim(-0.6, len(pts) - 0.4)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.set_title(METRIC_TITLES.get(metric, ylabel))
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", metric)
    filepath_save_plot_or_show(name=f"plot_{safe}_summary", show=show)


def plot_metric_vs_latent_factors(series_stats, metric, ylabel, filters=None,
                                  show=False, ref_label="ge", scale=1.0):
    """A batch statistic (e.g. Moran's I) of MF as a function of the number
    of latent factors f: one point per f, mean +/- SEM over the runs, joined
    by a line.  The non-MF series (g-EPR, User-kNN, ...) are drawn as
    horizontal lines with a band of +/- one SEM.

    series_stats : list of (label, algorithm_1, num_latent_factors,
                   batch_stats_list)."""
    mf_pts, refs = [], []
    for label, alg, f, bsl in series_stats:
        v = _batch_metric_values(bsl, metric) * scale
        if v.size == 0:
            continue
        m, e = _mean_sem(v)
        if str(alg).startswith("mf") and f is not None:
            mf_pts.append((float(f), m, e, v.size))
        else:
            refs.append((label, m, e))
    if len({p[0] for p in mf_pts}) < 2:
        logger.debug(f"plot_metric_vs_latent_factors({metric}): fewer than 2 MF "
                    f"series with different f — skipped.")
        return
    mf_pts.sort()
    f = np.array([p[0] for p in mf_pts]); m = np.array([p[1] for p in mf_pts])
    e = np.array([p[2] for p in mf_pts]); n = min(p[3] for p in mf_pts)

    fig, ax = plt.subplots(figsize=PANEL_RECT)
    _draw_points(ax, f, m, e, SERIES_COLORS.get("mf", "#FF9800"), n,
                 label="MF", ms=6, lw=1.6)
    arrays = [m - e, m + e]
    for idx, (label, rm, re_) in enumerate(refs):
        col = _color_of(label, idx)
        ax.axhline(rm, color=col, ls="--", lw=1.6, label=label, zorder=1)
        if np.isfinite(re_) and re_ > 0:
            ax.axhspan(rm - re_, rm + re_, color=col, alpha=0.15, lw=0, zorder=0)
        arrays += [np.array([rm - re_, rm + re_])]
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax.xaxis.set_major_locator(FixedLocator(f))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    _fit_ylim(ax, arrays)
    ax.set_xlabel(r"$f$")
    ax.set_ylabel(ylabel)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    ax.set_title(METRIC_TITLES.get(metric, ylabel))
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", metric)
    filepath_save_plot_or_show(name=f"plot_{safe}_vs_latent_factors", show=show)


# Which sensitivity the plot "metric_vs_k" draws against (set by
# model.campaigns):
#   None    -> x = top_k (length of the recommendation list), as "k_top"
#   "k"     -> x = score_k (number of terms in the score; SWEEP_REF when the
#              series has no score_k, i.e. the main runs)
#   "k_top" -> x = top_k (length of the recommendation list)
SWEEP = None
SWEEP_REF = 25
_SWEEP_MEANING = {
    "k":     {"uk": "peer group", "ik": "venues in the score"},
    "k_top": {"uk": "list length", "ik": "list length", "mf": "list length"},
}
_SWEEP_XLABEL = {None: r"$k$", "k": r"$k$", "k_top": r"$k_{\mathrm{top}}$"}
_SWEEP_FILE = {None: "top_k", "k": "k", "k_top": "k_top"}


def sweep_value(plot_params):
    """x of a series in plot "metric_vs_k" according to SWEEP."""
    if SWEEP == "k":
        v = plot_params.get("score_k")
        return SWEEP_REF if v is None else v
    return plot_params.get("top_k")


def plot_metric_vs_top_k(series_stats, metric, ylabel, filters=None,
                         show=False, scale=1.0):
    """A batch statistic as a function of k = top_k, one curve per recommender
    (mean +/- SEM over the runs at each k, joined by a line).  Only the
    recommenders with at least two different k are drawn as curves; the other
    series (g-EPR) are horizontal references with a band of +/- one SEM.
    Meaning of k: number of terms in the score (SWEEP = "k": peer group of
    User-kNN, visited venues in the score of Item-kNN) or length of the
    recommendation list (SWEEP = "k_top" or None).

    series_stats : list of (label, algorithm_1, top_k, batch_stats_list)."""
    meaning = _SWEEP_MEANING.get(SWEEP, _SWEEP_MEANING["k_top"])
    by_alg, refs = {}, []
    for label, alg, k, bsl in series_stats:
        v = _batch_metric_values(bsl, metric) * scale
        if v.size == 0:
            continue
        m, e = _mean_sem(v)
        alg = str(alg)
        if alg in meaning and k is not None:
            by_alg.setdefault(alg, []).append((float(k), m, e, v.size))
        else:
            refs.append((label, m, e))
    curves = {a: sorted(p) for a, p in by_alg.items() if len({q[0] for q in p}) >= 2}
    if not curves:
        logger.debug(f"plot_metric_vs_top_k({metric}): no recommender with at least "
                    f"2 different k — skipped.")
        return

    fig, ax = plt.subplots(figsize=PANEL_RECT)
    arrays, all_k = [], set()
    for alg in ("uk", "ik", "mf"):
        if alg not in curves:
            continue
        pts = curves[alg]
        k = np.array([p[0] for p in pts]); m = np.array([p[1] for p in pts])
        e = np.array([p[2] for p in pts]); n = min(p[3] for p in pts)
        all_k |= set(k.tolist())
        _draw_points(ax, k, m, e, SERIES_COLORS.get(alg, "#666666"), n,
                     label=LABEL_NAMES.get(alg, alg),
                     ms=6, lw=1.6)
        arrays += [m - e, m + e]
    for idx, (label, rm, re_) in enumerate(refs):
        col = _color_of(label, idx)
        ax.axhline(rm, color=col, ls="--", lw=1.6, label=_pretty_label(label), zorder=1)
        if np.isfinite(re_) and re_ > 0:
            ax.axhspan(rm - re_, rm + re_, color=col, alpha=0.15, lw=0, zorder=0)
        arrays += [np.array([rm - re_, rm + re_])]
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax.xaxis.set_major_locator(FixedLocator(sorted(all_k)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    _fit_ylim(ax, arrays)
    xl = _SWEEP_XLABEL.get(SWEEP, r"$k$")
    ax.set_xlabel(xl)
    ax.set_ylabel(ylabel)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    ax.set_title(METRIC_TITLES.get(metric, ylabel))
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", metric)
    filepath_save_plot_or_show(name=f"plot_{safe}_vs_{_SWEEP_FILE.get(SWEEP, 'top_k')}",
                               show=show)


# =============================================================================
# SINGLE-RUN SELECTION (same cycle across series)
# =============================================================================

def t0_fingerprint(df, t0):
    """Hashable fingerprint of a run's popularity at the last snapshot <= t0.
    Every continuation resumed from the same baseline has
    IDENTICAL history up to t0, so runs of the same cycle share this value."""
    times, cells, V = _ts_matrix(df)
    j = _ref_index(times, t0)
    if j < 0:
        return None
    col = V[:, j]
    keep = col > 0
    return tuple(sorted(zip((str(c) for c in np.asarray(cells)[keep]),
                            col[keep].astype(np.int64).tolist())))


def pick_same_cycle_runs(series_with_ids, t0, ref_label="ge"):
    """Choose ONE run per series, all from the same cycle (same baseline).

    series_with_ids : list of (label, [(batch_id, ts_df), ...])
    Cycles are identified by the t0 fingerprint (identical pre-t0 history).
    Anchor runs of `ref_label` (or of the first series) are tried from the
    most recent (last batch id = latest timestamp) backwards; the first cycle
    present in EVERY series is used.  If no cycle is complete, the most
    recent anchor cycle is used and series lacking it fall back to their own
    most recent run, flagged as not paired.

    Returns {label: (batch_id, ts_df, paired_bool)}.
    """
    if not series_with_ids:
        return {}
    by_label = {l: sorted(r, key=lambda x: x[0]) for l, r in series_with_ids if r}
    if not by_label:
        return {}
    anchor_label = ref_label if ref_label in by_label else next(iter(by_label))
    fps = {l: [(bid, df, t0_fingerprint(df, t0)) for bid, df in runs]
           for l, runs in by_label.items()}

    def _match(label, fp):
        hit = [(bid, df) for bid, df, f in fps[label] if fp is not None and f == fp]
        return hit[-1] if hit else None

    anchors = fps[anchor_label][::-1]                 # newest first
    for a_bid, a_df, fp in anchors:
        picks = {l: _match(l, fp) for l in by_label if l != anchor_label}
        if all(v is not None for v in picks.values()):
            out = {anchor_label: (a_bid, a_df, True)}
            out.update({l: (v[0], v[1], True) for l, v in picks.items()})
            return {l: out[l] for l, _ in series_with_ids if l in out}

    a_bid, a_df, fp = anchors[0]
    out = {anchor_label: (a_bid, a_df, True)}
    for l in by_label:
        if l == anchor_label:
            continue
        m = _match(l, fp)
        if m is not None:
            out[l] = (m[0], m[1], True)
        else:
            logger.warning(f"single-run grids: no cycle is shared by all series; "
                           f"'{l}' uses its most recent run (NOT the same cycle "
                           f"as {anchor_label}/{a_bid}).")
            out[l] = (by_label[l][-1][0], by_label[l][-1][1], False)
    return {l: out[l] for l, _ in series_with_ids if l in out}


# =============================================================================
# RANK ORIGIN: where did the venue now at final rank r come from?
# =============================================================================
# x = rank at the end of the run (t1).  y = MEAN over runs of the rank the
# same venue had at the end of the baseline (t0).  Only positions matter: in
# each run, "the venue that finished r-th" is looked up at t0, then runs are
# averaged position by position (mean +/- SEM).
# To keep the curve readable over thousands of ranks, x is binned in log
# space (single ranks in the top, wider bins further down): within each run
# the t0 ranks are averaged inside the bin first, then across runs.
# Ranks: 1 = most visited; ties share the average rank.  Venues never visited
# by t0 share the last t0 rank.  y = x means no reshuffling; y above the
# diagonal = the final position was reached by venues CLIMBING from below.
# -----------------------------------------------------------------------------

def _int_log_edges(n_max, per_decade=16):
    """Integer rank-bin edges: [1, 2, 3, ..., ..., n_max + 1], log-spaced."""
    n_max = max(1, int(n_max))
    raw = np.logspace(0, np.log10(n_max + 1), per_decade * max(1, int(np.ceil(np.log10(n_max + 1)))) + 1)
    e = np.unique(np.floor(raw).astype(int))
    e = e[(e >= 1) & (e <= n_max + 1)]
    if e[0] != 1:
        e = np.insert(e, 0, 1)
    if e[-1] != n_max + 1:
        e = np.append(e, n_max + 1)
    return e


def _rank_origin_run(df, t0, t1, edges):
    """One run -> mean t0 rank per final-rank bin (NaN where empty)."""
    times, cells, V = _ts_matrix_upto(df, t1)
    j0 = _ref_index(times, t0)
    jf = times.size - 1
    if j0 < 0 or jf <= j0:
        return None
    a, f = V[:, j0], V[:, jf]
    vis = f > 0
    if not vis.any():
        return None
    r0 = scipy.stats.rankdata(-a, method="average")[vis]
    rf = scipy.stats.rankdata(-f[vis], method="average")
    b = np.searchsorted(edges, rf, side="right") - 1
    out = np.full(edges.size - 1, np.nan)
    for k in np.unique(b):
        if 0 <= k < out.size:
            out[k] = float(np.mean(r0[b == k]))
    return out


def plot_rank_origin(series, filters=None, show=False, metric="visits", name=None):
    """series : list of (label, [ts_df per batch]).  Two files: log-log and lin."""
    t0 = (filters or {}).get("t0")
    t1 = _t1_of(filters)
    if t0 is None:
        logger.warning("plot_rank_origin: no t0 in filters — skipped.")
        return
    n_max = 1
    for _l, dfs in series:
        for df in dfs:
            times, _c, V = _ts_matrix_upto(df, t1)
            if times.size:
                n_max = max(n_max, int((V[:, -1] > 0).sum()))
    edges = _int_log_edges(n_max)
    lo_e, hi_e = edges[:-1].astype(float), (edges[1:] - 1).astype(float)
    x = np.sqrt(lo_e * hi_e)                 # geometric bin centre (exact for single ranks)

    multi = len(series) > 1
    curves = []
    for idx, (label, dfs) in enumerate(series):
        runs = [c for c in (_rank_origin_run(df, t0, t1, edges) for df in dfs)
                if c is not None]
        agg = _batch_band(runs)
        if agg is None:
            continue
        _k, mean, sem, nb = agg
        curves.append((label, _series_color(idx, multi), mean, sem, nb))
    if not curves:
        logger.warning("plot_rank_origin: no data.")
        return

    base = name or f"plot_rank_origin_{metric}"
    for tag, is_log in (("log", True), ("lin", False)):
        fig, ax = plt.subplots(figsize=PANEL_SQUARE)
        ymax = 1.0
        for label, color, mean, sem, nb in curves:
            ok = np.isfinite(mean)
            ax.plot(x[ok], mean[ok], marker="o", ms=3.5, ls="none", color=color,
                    label=label if multi else None, zorder=3)
            if nb > 1:
                ax.fill_between(x[ok], np.maximum(mean[ok] - sem[ok], 1.0),
                                mean[ok] + sem[ok], color=color, alpha=0.2,
                                lw=0, zorder=2)
            ymax = max(ymax, np.nanmax(mean + sem))
        top = max(ymax, x.max())
        # diagonal y = x (no reshuffling): drawn, not in the legend (it is
        # described in the caption)
        ax.plot([1, top], [1, top], ls="--", lw=1.2, color="gray", zorder=1)
        if is_log:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"Rank at $t_{\mathrm{end}}$")
        ax.set_ylabel(r"Mean rank at $t_0$")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        if multi:
            ax.legend()
        pop = r"$n^{\mathrm{E}}$" if metric == "explorers" else r"$n^{\mathrm{V}}$"
        ax.set_title(f"Rank origin ({pop})")
        filepath_save_plot_or_show(name=f"{base}_{tag}", show=show)

# =============================================================================
# RADIUS OF GYRATION AND CENTRE OF MASS BY RING ("rg_by_ring")
# =============================================================================
# How individual mobility changes with the distance from the centre of the
# grid at which the individual lives (thesis, Fig. "rg by ring").  For every
# individual, from its visits (initial position + destinations, each visit
# with the same weight, as r_cm and r_g in the thesis):
#   r_cm^<=  centre of mass of the visits up to t0
#   r_cm^>   centre of mass of the visits after t0
#   r_g(t0)  radius of gyration of the visits up to t0
#   r_g(t1)  radius of gyration of all the visits up to t1
# Positions are measured from the centre of the grid; individuals are split
# in rings of |r_cm^<=| (RING_EDGES_KM).  Statistics are computed run by run;
# points = mean over the realisations, bars = standard error between them.
#   plot_rg_by_ring_ratio         <r_g(t1)> / <r_g(t0)> per ring        (a)
#   plot_rg_by_ring_radial_shift  <(r_cm^> - r_cm^<=) . r_cm^<= / |r_cm^<=|>
#                                 [km] per ring (negative = towards the centre)  (b)
#   rg_by_ring.tsv                every measure, per series and ring
# The per-individual values of each run are cached in
#   <project>/.cache/rg_by_ring/<parameter folder>/<run>.parquet
# (hidden folder, safe to delete: it is rebuilt from the histories).
RING_EDGES_KM = [0, 5, 10, 15, 20, 25, np.inf]
_RING_COLS = ["Departure_x_coord", "Departure_y_coord",
              "Destination_x_coord", "Destination_y_coord", "Clock"]
_RING_VALUES = ["uid", "n0", "c0x", "c0y", "rg0", "np", "cpx", "cpy", "rg1"]
_RING_LEGEND_HEADROOM = 0.35
# True = one realisation per series (set by model.runner.plot, first_run_only).
FIRST_RUN_ONLY = False


def _ring_user_values(path, t0):
    """n0, r_cm^<=, r_g(t0), n>, r_cm^>, r_g(t1) of one individual (metres)."""
    h = pd.read_parquet(path, columns=_RING_COLS)
    if h.empty:
        return None
    x = np.concatenate([[h["Departure_x_coord"].iloc[0]], h["Destination_x_coord"].to_numpy()])
    y = np.concatenate([[h["Departure_y_coord"].iloc[0]], h["Destination_y_coord"].to_numpy()])
    clock = np.concatenate([[0.0], h["Clock"].to_numpy(dtype=float)])
    pre = clock <= t0
    post = ~pre

    def cm_rg(m):
        cx, cy = x[m].mean(), y[m].mean()
        return cx, cy, float(np.sqrt(((x[m] - cx) ** 2 + (y[m] - cy) ** 2).mean()))

    c0x, c0y, rg0 = cm_rg(pre)
    _, _, rg1 = cm_rg(np.ones_like(pre))
    if post.any():
        cpx, cpy = float(x[post].mean()), float(y[post].mean())
    else:
        cpx = cpy = np.nan
    return int(pre.sum()), c0x, c0y, rg0, int(post.sum()), cpx, cpy, rg1


def _ring_read_run(args):
    """Per-individual values of one run (from the cache when present)."""
    run_dir, t0, cache = args
    if os.path.isfile(cache):
        return pd.read_parquet(cache)
    rows = []
    for f in glob.glob(os.path.join(run_dir, "user_*_history.parquet")):
        v = _ring_user_values(f, t0)
        if v is not None:
            rows.append((int(os.path.basename(f).split("_")[1]),) + v)
    df = pd.DataFrame(rows, columns=_RING_VALUES).sort_values("uid").reset_index(drop=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    df.to_parquet(cache)
    return df


def _ring_stats(df):
    """Per ring of |r_cm^<=|: radius-of-gyration and centre-of-mass measures."""
    c0 = np.hypot(df.c0x, df.c0y)
    ok = (df.rg0 > 0) & (c0 > 0)
    df, c0 = df[ok], c0[ok]
    ratio = df.rg1 / df.rg0
    dx, dy = df.cpx - df.c0x, df.cpy - df.c0y              # NaN if no visit after t0
    radial = (dx * df.c0x / c0 + dy * df.c0y / c0) / 1000.0  # [km]
    modulus = np.hypot(dx, dy) / 1000.0                      # [km]
    ring = np.digitize(c0 / 1000.0, RING_EDGES_KM[1:-1])
    out = []
    for k in range(len(RING_EDGES_KM) - 1):
        m = ring == k
        if not m.any():
            out.append(dict(ring=k, n=0))
            continue
        out.append(dict(ring=k, n=int(m.sum()),
                        ratio_mean=float(ratio[m].mean()),
                        ratio_median=float(np.median(ratio[m])),
                        ratio_of_means=float(df.rg1[m].mean() / df.rg0[m].mean()),
                        radial=float(np.nanmean(radial[m])),
                        modulus=float(np.nanmean(modulus[m]))))
    return out


def plot_rg_by_ring(series, filters=None, show=False, max_workers=None):
    """series: [(label, plot_params), ...] of continuations (plot_params with
    t0/t1/algorithms/grid as given to the loaders).  Draws the two figures and
    the table described above.  Uses only the first run of each series when
    FIRST_RUN_ONLY is True (quick look)."""
    from concurrent.futures import ProcessPoolExecutor
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Bbox
    import model.utils as mu

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = os.path.join(root, "combined_analysis_data")
    cache_root = os.path.join(root, ".cache", "rg_by_ring")
    tasks, owner = [], []
    for label, pp in series:
        if not pp.get("t0"):
            logger.warning("plot_rg_by_ring: needs continuations (t0); skipped")
            return
        folder = mu._build_combined_param_folder_name(pp)
        runs = sorted(d for d in glob.glob(os.path.join(data, folder, "*")) if os.path.isdir(d))
        if FIRST_RUN_ONLY:
            runs = runs[:1]
        if not runs:
            logger.warning(f"plot_rg_by_ring: no runs for '{label}'")
        for r in runs:
            tasks.append((r, float(pp["t0"]),
                          os.path.join(cache_root, folder, os.path.basename(r) + ".parquet")))
            owner.append((label, os.path.basename(r)))
    if not tasks:
        return
    n_new = sum(not os.path.isfile(t[2]) for t in tasks)
    logger.debug(f"plot_rg_by_ring: {len(tasks)} runs ({n_new} to read from the histories)")
    if n_new:
        logger.info(f"  r_g by ring: reading the histories of {n_new} run(s) "
                    f"(kept in .cache/ for the next time)")
    workers = max(1, min(max_workers or os.cpu_count() or 1, len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_ring_read_run, t): i for i, t in enumerate(tasks)}
        prog = mu.Progress("r_g by ring", len(tasks), "runs")
        dfs = [None] * len(tasks)
        for fut in mu.completed_with_progress(futs, prog):
            dfs[futs[fut]] = fut.result()
            prog.update(prog.done + 1)

    rows = []
    for (label, run), df in zip(owner, dfs):
        for st in _ring_stats(df):
            rows.append(dict(label=label, run=run, **st))
    st = pd.DataFrame(rows)
    measures = ["ratio_mean", "ratio_median", "ratio_of_means", "radial", "modulus"]
    spec = {"n_per_run": ("n", "mean"), "runs": ("run", "nunique")}
    for c in measures:
        spec[c] = (c, "mean")
        spec[c + "_sem"] = (c, "sem")
    agg = st.groupby(["label", "ring"]).agg(**spec).reset_index()

    plots_dir = _resolve_plot_dir()
    agg.to_csv(os.path.join(plots_dir, "rg_by_ring.tsv"), sep="\t", index=False)

    names = [f"{int(a)}–{int(b)}" if np.isfinite(b) else f">{int(a)}"
             for a, b in zip(RING_EDGES_KM[:-1], RING_EDGES_KM[1:])]
    xs = np.arange(len(names))
    panels = [
        ("ratio_of_means",
         r"$\langle r_g(t_{\mathrm{end}}) \rangle \,/\, \langle r_g(t_0) \rangle$",
         "Growth of the radius of gyration", 1.0, "plot_rg_by_ring_ratio"),
        ("radial", r"$\langle \Delta r_{cm}^{\mathrm{rad}} \rangle$ [km]",
         "Radial shift of the centre of mass", 0.0, "plot_rg_by_ring_radial_shift"),
    ]
    labels = [lab for lab, _pp in series]
    figs = []
    for col, ylab, title, ref_y, name in panels:
        fig, ax = plt.subplots(figsize=PANEL_RECT)
        handles = []
        for i, label in enumerate(labels):
            a = agg[agg.label == label].sort_values("ring")
            if a.empty:
                continue
            color = _color_of(label, i)
            marker = "o" if label == "ge" else "s"
            ax.errorbar(xs[a.ring.to_numpy()], a[col], yerr=a[col + "_sem"], marker=marker,
                        ms=5, lw=1.6, capsize=3, color=color,
                        zorder=10 if label == "ge" else 3)
            handles.append(Line2D([], [], marker=marker, ms=5, lw=1.6, color=color,
                                  label=LABEL_NAMES.get(label, label)))
        ax.axhline(ref_y, ls="--", lw=1, color="gray")
        ax.set_xticks(xs)
        ax.set_xticklabels(names)
        ax.set_xlabel(r"$|r_{cm}^{\leq}|$ [km]")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, ls="--", alpha=0.4)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + _RING_LEGEND_HEADROOM * (hi - lo))
        ax.legend(handles=handles, ncol=2, loc="upper center").set_zorder(20)
        _style_figure(fig, name)
        if PANEL_TAG:
            _apply_panel_tag(fig, PANEL_TAG)
        figs.append((fig, name))
    if show:
        plt.show()
        return
    # same crop for both figures, so that the two PDFs have identical sizes
    boxes = []
    for fig, _ in figs:
        fig.canvas.draw()
        boxes.append(fig.get_tightbbox(fig.canvas.get_renderer()))
    common = Bbox.union(boxes).padded(0.1)
    for fig, name in figs:
        fig.savefig(os.path.join(plots_dir, f"{name}{_NAME_TAG}.pdf"), bbox_inches=common)
        plt.close(fig)
    logger.debug(f"Saved plot: rg_by_ring figures in {plots_dir}")