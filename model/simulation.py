import numpy as np
import model.user
import model.utils
import model.utils_plot
import model.utils_cf
import model.grid
import pandas as pd
import os
import gc
import concurrent.futures
from datetime import datetime

import logging
logger = logging.getLogger(__name__)


# Folder names are built in model.utils (build_param_folder_name for the
# baselines, _build_combined_param_folder_name for the continuations), the
# single source of truth for the write side (here) and the read side (loaders).
from model.utils import sp_abbr
from model.utils import POWERLAW_DEFAULTS as _PL

# ---------------------------------------------------------------------------
# Disk-lightening switch (leva A).
#
# When False, per-user grid parquets are NOT persisted in the ANALYSIS folders
# (analysis_data / combined_analysis_data).  The batch aggregate
# grids (aggregate_grid + aggregate_grid_exploration) are instead built in
# memory from the live per-user grids and saved once per batch, so every
# grid-based metric (collective entropy, Moran's I) and every heatmap that
# reads the batch aggregate keep working unchanged.
#
# RAW grids are always kept: _reconstruct_users_from_raw() reloads
# user_{i}_grid.parquet to restore each user's grid state, so the resume phase
# depends on them.  save_combined_raw_data() grids follow this flag (the
# combined analysis reads grids from the in-memory live users, not from the
# combined raw folder, so dropping them is safe for a single-cycle sweep).
#
# Default True preserves the historical behaviour; the grid/init sweep driver
# sets it to False.
# ---------------------------------------------------------------------------
SAVE_PER_USER_GRIDS = True

# ---------------------------------------------------------------------------
# Temporal sampling step (clock units) for the per-cell popularity time series
# saved by run_analysis (continuations only).  Every multiple of this
# step, up to the final clock, becomes one snapshot column in
# popularity_timeseries.parquet.  That single per-batch artifact feeds the
# time-resolved venue plots (accumulation velocity, top-k turnover/stability and
# Kendall-tau-over-time).  Change here to sample more/less finely.
# ---------------------------------------------------------------------------
POP_SNAPSHOT_STEP = 86400          # one day


class _VenueSnapshot:
    """
    Lightweight proxy that exposes a frozen copy of a single user's
    venue_history to build_cf_data().

    build_cf_data() (and _build_uv_matrix inside it) only reads
    user.venue_history — it never touches user_history, grid, clock, or
    any other User attribute.  Passing a _VenueSnapshot instead of the
    live User object guarantees that every user in a CF interval is
    evaluated against the same temporal baseline: the state of
    venue_history at the *start* of that interval, before any user has
    been advanced.  This removes the ordering asymmetry that would
    otherwise arise from simulating users sequentially.

    Parameters
    ----------
    venue_history : list
        The live user.venue_history at the moment the snapshot is taken.
        A shallow copy of the outer list is stored; individual rows are
        not copied because build_cf_data only reads them.
    """
    __slots__ = ("venue_history",)

    def __init__(self, venue_history):
        # Shallow-copy the list so that rows appended during simulation
        # of this interval do not appear in the frozen view.
        self.venue_history = list(venue_history)


# PARALLELIZATION HELPERS


# Memory-safety cap on the number of concurrent worker processes.
#
# The simulation phase fans one task per user across a ProcessPoolExecutor, and
# each fork holds a heavy chunk of trajectory data; with the historical default
# of min(cpu_count, n_tasks) a 24-core box spawned 24 workers, whose combined
# RSS overran RAM+swap and, when the driver died mid-run, left the workers
# orphaned and stuck. Capping the pool keeps peak concurrent memory bounded.
#
# Override at launch without editing code, e.g.  AIEPR_MAX_WORKERS=8 python ...
_MEM_SAFE_WORKER_CAP = max(1, int(os.environ.get("AIEPR_MAX_WORKERS", "16")))


def _resolve_seed(params):
    """The "seed" of params as an int, or a fresh random one when absent/None."""
    seed = (params or {}).get("seed")
    if seed is None:
        return int(np.random.SeedSequence().entropy % (2**31))
    return int(seed) % (2**31)


def _default_n_workers(n_tasks=None):
    """Default worker count: logical cores, capped for memory safety and by the
    task count. The cap is _MEM_SAFE_WORKER_CAP (env AIEPR_MAX_WORKERS)."""
    n_cpu = os.cpu_count() or 1
    n = min(n_cpu, _MEM_SAFE_WORKER_CAP)
    if n_tasks is not None:
        return max(1, min(n, n_tasks))
    return max(1, n)


# --------------------------------------------------------------------------
# PARALLEL PLOTTING
#
# Each requested plot is an independent unit of work that ends by writing its
# own PDF to disk, so we can fan them out across processes.  Two design points:
#
#   * matplotlib's pyplot state is not process-safe, but separate PROCESSES
#     each have their own interpreter, so process-level parallelism is safe as
#     long as every worker uses the non-interactive Agg backend and writes to a
#     distinct file (which it does — file names differ per plot).
#
#   * The per-simulation payloads (lists of DataFrames) can be large.  To avoid
#     pickling them to every worker, we stash the tasks in a module-level global
#     and rely on the Unix "fork" start method: forked workers inherit the
#     parent's memory copy-on-write, so the big data is shared, not copied.
#     Each task is referenced by a small integer key.
# --------------------------------------------------------------------------

_PENDING_PLOTS = {}     # key -> (func, kwargs); populated in the parent before fork


def _init_plot_worker():
    """Force the headless Agg backend inside each plotting worker process."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")
    except Exception:
        pass


def _plot_dispatch(key):
    """Worker entry point: look up the inherited task and run it."""
    func, kwargs = _PENDING_PLOTS[key]
    func(**kwargs)
    return key


def _log_series_loaded(series):
    logger.info(f"  series {series['label']} loaded: {series['n_sims']} run(s), "
                f"{len(series['rg_list'])} individuals")


def _run_plot_tasks(tasks, parallel=True, max_workers=None, show=False):
    """Run a list of (name, func, kwargs) plot tasks, serially or in parallel.

    Parallelism is disabled automatically when show=True (interactive display
    must stay on the main process) or when there is a single task.
    """
    global _PENDING_PLOTS

    if show or not parallel or len(tasks) <= 1:
        for name, func, kwargs in tasks:
            logger.debug(f"Plotting (serial): {name}")
            func(**kwargs)
        return

    max_workers = max_workers or _default_n_workers(len(tasks))

    # Publish tasks for the forked workers, then dispatch by key.
    _PENDING_PLOTS = {i: (func, kwargs) for i, (_, func, kwargs) in enumerate(tasks)}
    names = {i: name for i, (name, _, _) in enumerate(tasks)}
    try:
        ctx = __import__("multiprocessing").get_context("fork")
    except ValueError:  # pragma: no cover - non-fork platforms
        ctx = None

    if ctx is None:
        # Fork unavailable (e.g. Windows): fall back to serial to stay safe.
        for name, func, kwargs in tasks:
            logger.debug(f"Plotting (serial fallback): {name}")
            func(**kwargs)
        _PENDING_PLOTS = {}
        return

    logger.debug(f"Plotting {len(tasks)} figures across {max_workers} processes")
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, mp_context=ctx, initializer=_init_plot_worker
        ) as ex:
            futures = {ex.submit(_plot_dispatch, k): k for k in _PENDING_PLOTS}
            prog = model.utils.Progress("figures", len(futures))
            for fut in model.utils.completed_with_progress(futures, prog):
                k = futures[fut]
                prog.update(prog.done + 1)
                try:
                    fut.result()
                    logger.debug(f"  done: {names[k]}")
                except Exception as exc:
                    logger.error(f"  FAILED: {names[k]}: {exc}")
    finally:
        _PENDING_PLOTS = {}


def _series_dict_from_sim(label, tmp):
    """Extract the per-series payload (flat lists + per-sim groupings) from a
    freshly loaded Simulation.  Shared by the serial and parallel load paths."""
    return {
        "label":               label,
        "rg_list":             list(tmp.rg_list),
        "two_rg_list":         list(tmp.two_rg_list),
        "rg_e_list":           list(tmp.rg_e_list),
        "all_rg_over_time":    list(tmp.all_rg_over_time),
        "all_s_over_time":     list(tmp.all_s_over_time),
        "batch_stats_list":    list(tmp.batch_stats_list),
        "_plot_params":        dict(tmp._plot_params),
        "s_rand_list":         list(tmp.s_rand_list),
        "s_unc_list":          list(tmp.s_unc_list),
        "s_real_list":         list(tmp.s_real_list),
        "pi_rand_list":        list(tmp.pi_rand_list),
        "pi_unc_list":         list(tmp.pi_unc_list),
        "pi_real_list":        list(tmp.pi_real_list),
        "two_rg_ratio_list":   list(tmp.two_rg_ratio_list),
        "num_venues_list":     list(getattr(tmp, "num_venues_list", [])),
        # Compact, memory-light payloads (Solution A): prebinned distance
        # histograms and pre-reduced per-venue arrays replace the raw per-jump
        # lists and the per-user venue DataFrames.
        "dist_edges":          getattr(tmp, "dist_edges", None),
        "dist_hist_per_sim":   list(getattr(tmp, "dist_hist_per_sim", [])),
        "venue_visits_per_sim": list(getattr(tmp, "venue_visits_per_sim", [])),
        "venue_users_per_sim":  list(getattr(tmp, "venue_users_per_sim", [])),
        "venue_reduced":        getattr(tmp, "_venue_reduced", False),
        "rg_list_per_sim":          list(tmp.rg_list_per_sim),
        "two_rg_list_per_sim":      list(tmp.two_rg_list_per_sim),
        "rg_e_list_per_sim":        list(tmp.rg_e_list_per_sim),
        "all_rg_over_time_per_sim": list(tmp.all_rg_over_time_per_sim),
        "all_s_over_time_per_sim":  list(tmp.all_s_over_time_per_sim),
        # Kept only when venue was NOT reduced (e.g. freq path); empty otherwise.
        "venue_history_per_sim":    ([] if getattr(tmp, "_venue_reduced", False)
                                     else list(tmp.venue_history_per_sim)),
        "s_rand_per_sim":           list(tmp.s_rand_per_sim),
        "s_unc_per_sim":            list(tmp.s_unc_per_sim),
        "s_real_per_sim":           list(tmp.s_real_per_sim),
        "pi_rand_per_sim":          list(tmp.pi_rand_per_sim),
        "pi_unc_per_sim":           list(tmp.pi_unc_per_sim),
        "pi_real_per_sim":          list(tmp.pi_real_per_sim),
        "two_rg_ratio_per_sim":     list(tmp.two_rg_ratio_per_sim),
        "num_venues_per_sim":       list(getattr(tmp, "num_venues_per_sim", [])),
        "n_sims":                   getattr(tmp, "_n_sims", 1),
    }


def _load_one_series_worker(args):
    """Worker entry point: load a single mp series in its own process.

    Each process has its own GIL, so this parallelises the parquet decode /
    DataFrame construction that thread-level parallelism cannot (it is held by
    the GIL).  `inner_threads` caps the per-process read threads to avoid
    oversubscribing the cores when several series load at once.
    """
    label, p_dict, grid_params, plot_list, inner_threads = args
    model.utils.LOAD_THREADS = max(1, inner_threads)
    tmp = Simulation()
    # mp never draws freq_distribution, so venue_history can always be reduced.
    tmp.load_analysis_data(p_dict, grid_params=grid_params, plot_list=plot_list,
                           reduce_venue=True)
    return _series_dict_from_sim(label, tmp)


def _initial_position(starting_position, sc, rng=np.random):
    """Starting point of one individual: uniform in the square [-sc, sc]^2
    ("random"), or normal with sigma = sc/3 truncated to the square
    ("core_periphery"); None otherwise ("centered": the central cell).
    rng: np.random or a np.random.Generator."""
    if starting_position == "random":
        return (float(rng.uniform(-sc, sc)), float(rng.uniform(-sc, sc)))
    if starting_position == "core_periphery":
        def coord():
            while True:
                v = rng.normal(0, sc / 3)
                if -sc <= v <= sc:
                    return float(v)
        return (coord(), coord())
    return None


def _compute_d_weights(starting_position, cell_lenght, cell_number, spatial_center, user_number, seed):
    """
    Build the global, shared d-EPR attractiveness matrix d_i.

    The weight-assignment logic mirrors the initial spatial distribution of
    users, with one exception:
      - "centered": every user starts at the very same point, which is not
        a real spatial distribution, so all cells get a constant/uniform
        weight instead of a degenerate single-cell histogram.
      - "random" / "core_periphery": d_i is proportional to the number of users
        whose starting position falls in cell i -- i.e. the histogram of
        the population's initial placement.

    Parameters
    ----------
    starting_position : str
        "centered", "random", or "core_periphery".
    cell_lenght, cell_number : int
        Grid geometry.
    spatial_center : float or None
        Half-extent of the grid, used to sample "random"/"core_periphery" positions.
        Not used when starting_position == "centered".
    user_number : int
        Number of users (needed to build the histogram for "random"/"core_periphery").
    seed : int
        RNG seed for reproducibility, independent of the per-user seeds used
        later to actually run each user.

    Returns
    -------
    d_weights : np.ndarray of shape (cell_number, cell_number)
        The global attractiveness matrix.
    precomputed_positions : list of tuple or None
        Initial positions to be reused verbatim by each user task so that
        the histogram and the actual starting cells stay consistent.
        None when starting_position == "centered" (every user already
        deterministically starts at the central cell, nothing to precompute).
    """
    if starting_position == "centered":
        return np.ones((cell_number, cell_number), dtype=float), None

    rng = np.random.default_rng(seed)
    sc = spatial_center

    if starting_position not in ("random", "core_periphery"):
        raise ValueError(f"_compute_d_weights: unknown starting_position '{starting_position}'")
    positions = [_initial_position(starting_position, sc, rng) for _ in range(user_number)]

    d_weights = np.zeros((cell_number, cell_number), dtype=float)
    for pos in positions:
        cx, cy = model.grid.Grid.position_to_cell_coords_static(pos, cell_lenght, cell_number)
        d_weights[cx, cy] += 1.0

    return d_weights, positions


class _SimParamsStub:
    """
    Lightweight, picklable stand-in for a Simulation instance.

    User.__init__/make_decision and utils_cf.make_decision_cf only ever
    read `.ro`, `.gamma`, `.grid_cell_lenght`, `.grid_cell_number` and the
    power-law sampling bounds off the `simulation` object they're handed. This
    stub carries exactly that and nothing else, so worker processes never need
    to (de)serialize the real Simulation object (see module note above).

    The four *_min/*_max fields (A3/A4) are the spatial (jump-length) and
    temporal (waiting-time) power-law support bounds (default
    model.utils.POWERLAW_DEFAULTS).
    """
    __slots__ = ("ro", "gamma", "grid_cell_lenght", "grid_cell_number",
                 "spatial_min", "spatial_max", "temporal_min", "temporal_max")

    def __init__(self, ro, gamma, grid_cell_lenght=None, grid_cell_number=None,
                 spatial_min=_PL["spatial_min"], spatial_max=_PL["spatial_max"],
                 temporal_min=_PL["temporal_min"], temporal_max=_PL["temporal_max"]):
        self.ro = ro
        self.gamma = gamma
        self.grid_cell_lenght = grid_cell_lenght
        self.grid_cell_number = grid_cell_number
        self.spatial_min = spatial_min
        self.spatial_max = spatial_max
        self.temporal_min = temporal_min
        self.temporal_max = temporal_max


def _simulate_one_user(task):
    """
    Worker for run(): build one User from scratch and run it to time_target.
    Receives only plain data + a picklable sim_stub, never a live Simulation.
    """
    (uid_idx, a_param, b_param, grid_exists, sim_stub,
     starting_position, spatial_center, time_target, seed,
     d_weights, precomputed_init_pos) = task

    np.random.seed(seed)

    if precomputed_init_pos is not None:
        # Position fixed ahead of time by run() so that it matches the
        # histogram used to build d_weights (d-EPR, starting_position
        # "random"/"core_periphery"). Do not resample.
        init_pos = precomputed_init_pos
    else:
        init_pos = _initial_position(starting_position, spatial_center)

    user = model.user.User(
        uid=f"User_{uid_idx}",
        a_param=a_param,
        b_param=b_param,
        grid=grid_exists,
        simulation=sim_stub,
        initial_position=init_pos,
        d_weights=d_weights,
    )
    while user.clock < time_target:
        user.make_decision()
    return user


def _resume_epr_one_user(task):
    """Worker for resume(), pure-EPR branch: advance one already-built user."""
    user, time_target, seed = task
    np.random.seed(seed)
    while user.clock < time_target:
        user.make_decision()
    return user


def _resume_cf_one_user(task):
    """Worker for resume(), CF branch: advance one user within one interval."""
    user, neighbors, cf_type, t_end, seed = task
    np.random.seed(seed)
    # Reset per-interval CF counters; aggregated by resume() after the interval.
    user._cf_counts = {}
    while user.clock < t_end:
        user.make_decision_cf(neighbors, cf_type)
    return user


def _log_cf_interval_summary(users_list, interval_label, n_intervals, algorithm):
    """
    Aggregate the per-user CF counters collected during one resume interval
    into a single detailed log line (run.log), instead of thousands of
    per-event warnings.  Returns the short version for the terminal.

    For every outcome we report both the total number of occurrences and the
    number of distinct users that experienced it at least once, e.g.:

        Resume CF interval 4/5 [mf]: CF explorations 88,123 (4,870 users) |
        fallback no_recs 1,203 (412 users) | fallback exhausted 567 (230 users)
    """
    import model.utils_cf as _cf

    totals = {}
    users_hit = {}
    for user in users_list:
        counts = getattr(user, "_cf_counts", None) or {}
        for key, val in counts.items():
            if val <= 0:
                continue
            totals[key] = totals.get(key, 0) + val
            users_hit[key] = users_hit.get(key, 0) + 1

    # Friendly labels, reported in a fixed, meaningful order.
    ordered = [
        (_cf._CF_SUCCESS,   "CF explorations"),
        (_cf._CF_NO_RECS,   "fallback no_recs"),
        (_cf._CF_EXHAUSTED, "fallback exhausted"),
    ]
    # Append any unexpected keys at the end so nothing is silently dropped.
    for key in totals:
        if key not in {k for k, _ in ordered}:
            ordered.append((key, key))

    parts = []
    for key, label in ordered:
        if key in totals:
            parts.append(f"{label} {totals[key]:,} ({users_hit.get(key, 0):,} users)")

    summary = " | ".join(parts) if parts else "no CF activity recorded"

    total_fb = totals.get(_cf._CF_NO_RECS, 0) + totals.get(_cf._CF_EXHAUSTED, 0)
    total_expl = total_fb + totals.get(_cf._CF_SUCCESS, 0)
    fb_pct = (100.0 * total_fb / total_expl) if total_expl else 0.0

    logger.debug(
        f"Resume CF interval {interval_label}/{n_intervals} [{algorithm}]: "
        f"{summary} | EPR fallback rate {fb_pct:.1f}% of explorations"
    )
    return (f"CF explorations {totals.get(_cf._CF_SUCCESS, 0):,}, "
            f"fallback {fb_pct:.1f}%")


def _analyze_one_user(task):
    """
    Worker for run_analysis(): compute and write every per-user analysis
    artifact (history/venue/stats/grid parquet).

    Receives only already-extracted DataFrames -- never a live User or the
    Simulation -- so there is nothing problematic to pickle. This is where
    most of the analysis-phase wall-clock time actually goes: compute_stats
    calls compute_real_entropy, which is an O(n^2) pure-Python loop over the
    trajectory.
    """
    i, user_history_df, venue_history_df, grid_df, folder, t0, save_grids = task

    # Centre-of-mass displacement between the baseline (Clock <= t0) and the
    # continuation (Clock > t0); NaN for baseline runs (t0 None).
    _cm_dx, _cm_dy, _cm_mag = model.utils.compute_cm_displacement(
        user_history_df, t0=t0)

    uh = user_history_df
    vh = venue_history_df

    stats_df        = model.utils.compute_stats(uh, vh)
    rg_over_time_df = model.utils.compute_rg_over_time(uh)
    s_over_time_df  = model.utils.compute_s_over_time(uh)

    dist_col  = uh["Distance"].dropna()
    dist_col  = dist_col[dist_col > 0]
    mean_dist = float(dist_col.mean()) if len(dist_col) > 0 else 0.0
    num_venues = int(len(vh))

    extra_rows = pd.DataFrame([
        {"metric": "mean_travel_distance", "value": mean_dist},
        {"metric": "num_venues",            "value": float(num_venues)},
        {"metric": "cm_displacement_dx",    "value": _cm_dx},
        {"metric": "cm_displacement_dy",    "value": _cm_dy},
        {"metric": "cm_displacement_mag",   "value": _cm_mag},
    ])
    stats_df = pd.concat([stats_df, extra_rows], ignore_index=True)

    uh = uh.copy()
    uh["rg"] = rg_over_time_df["rg"].values
    s_map = dict(zip(s_over_time_df["clock"].values, s_over_time_df["s"].values))
    uh["s"] = uh["Clock"].map(s_map).ffill()

    uh.to_parquet(os.path.join(folder, f"user_{i}_history.parquet"), index=False)
    vh.to_parquet(os.path.join(folder, f"user_{i}_venue.parquet"),  index=False)
    stats_df.to_parquet(os.path.join(folder, f"user_{i}_stats.parquet"), index=False)

    # Per-user grid persistence (only when save_grids is on).  The batch
    # aggregate is built in-memory upstream, so no metric depends on these
    # files being present.
    if grid_df is not None and save_grids:
        grid_df.to_parquet(os.path.join(folder, f"user_{i}_grid.parquet"), index=False)

    return i


# ---------------------------------------------------------------------------
# TIME-RESOLVED VENUE DYNAMICS -- single dispatcher shared by plot() (one series)
# and multipleplot() ("mp"), so both modes draw exactly the same figures.
#
# Guiding principle: every quantity (velocity, rank, turnover, tau, grid, share)
# is computed WITHIN one run; only then are runs averaged (mean +/- SEM, or
# mean grid).  Visits of different runs are never pooled.
#
# Sources (per batch):
#   popularity_timeseries.parquet   -> metric "visits"
#   exploration_timeseries.parquet  -> metric "explorers" (distinct users)
# All time series are clipped at t1 (users' last jump overshoots the target).
# ---------------------------------------------------------------------------
TIME_RESOLVED_PLOTS = (
    "venue_velocity_fixed_top",    # velocity of the top-10 frozen at t0
    "venue_velocity_fixed_worst",  # velocity of the whole minimum-count block frozen at t0
    "venue_velocity_fixed_final",  # velocity of the top-10 at the end of the run (t1)
    "top10_turnover",              # #venues entering the top-10 per interval
    "kendall_over_time",           # tau_b of the top-10, consecutive snapshots
    "top10_rank_tracking",         # rank over time of top-10 frozen at t0 / at end
    "top10_membership_panels",     # t >= t0: (a) exit of the old top-10,
                                   #          (b) entry of the final top-10
    "rank_origin",                 # x = final rank, y = mean t0 rank of that venue
    "grid_final_comparison",       # final grids of 2-6 series (1x2/1x3/2x2/3x2, linear);
                                   #   utils_plot.GRID_FINAL_SERIES picks the series
                                   #   also in a single-run version
                                   #   (same cycle across series, "_single_run")
    "aggregate_grid_snapshots",    # physical-grid show every step (single dataset)
)

# Batch statistics drawn against the number of latent factors f of MF
# (plot "metric_vs_k", multiplot).  Keys are batch_stats metric names; add e.g.
#   "gini_collective": r"$G^{\mathrm{coll}}$",
#   "mean_rg_explorative": r"$\langle r_g^{\mathrm{E}} \rangle$ [m]",
#   "mean_rg": r"$\langle r_g \rangle$ [m]",
METRICS_VS_K = {
    "I": r"$I$",
    "gini_collective": r"$G^{\mathrm{coll}}$",
    "mean_rg": r"$\langle r_g \rangle$ [km]",
    "mean_rg_explorative": r"$\langle r_g^{\mathrm{E}} \rangle$ [km]",
    "mean_travel_distance": r"$\bar d$ [km]",
}
# Factor applied to a METRICS_VS_K metric before plotting (lengths m -> km).
METRICS_VS_K_SCALE = {"mean_rg": 1e-3, "mean_rg_explorative": 1e-3,
                      "mean_travel_distance": 1e-3}

# Batch statistics drawn as one point per series, mean +/- one standard
# deviation across the runs (plot "metric_summary", multiplot).  Keys are
# batch_stats metric names; add e.g.  "I": r"$I$",
METRICS_SUMMARY = {
    "gini_collective": r"$G^{\mathrm{coll}}$",
}

# Which popularity notions the time-series plots are drawn for.  Remove
# "explorers" to halve the number of figures.
TIME_SERIES_METRICS = ("visits", "explorers")


def _snapshot_pairs(snap_list, early_col, late_col):
    """[(batch_id, popularity_snapshots df)] -> [(early Series, late Series)]
    per run, indexed by cell (input of the rank_persistence plots)."""
    out = []
    for _bid, snap in snap_list:
        if snap is None or len(snap) == 0:
            continue
        idx = snap["cell"].values
        out.append((pd.Series(snap[early_col].values, index=idx, dtype=float),
                    pd.Series(snap[late_col].values, index=idx, dtype=float)))
    return out


def _draw_time_resolved(plot_list, series_params, filters, show=False,
                        mask_center=False, top_n=10, single_dataset=False):
    """
    series_params  : list of (label, plot_params), one entry per series.
    filters        : parameters of the figures (t0, t1, ...).
    single_dataset : True for one series -> enables the per-dataset grid show
                     (aggregate_grid_snapshots); in "mp" it is skipped.

    In multi-series figures only the cycles (baselines) present in EVERY
    series are used, in the same order for all series (cycle alignment, see
    utils_plot.align_cycles): a cycle is recognised by the popularity at t0,
    identical for all the continuations of the same baseline.
    """
    up = model.utils_plot
    wanted = [pl for pl in TIME_RESOLVED_PLOTS if pl in plot_list]
    if not wanted:
        return
    t0 = (filters or {}).get("t0")

    _loaders = {"visits":    model.utils.load_combined_popularity_timeseries,
                "explorers": model.utils.load_combined_exploration_timeseries}
    cn_of = {label: pparams.get("cell_number") for label, pparams in series_params}

    for metric in TIME_SERIES_METRICS:
        ts_ids = []                                   # [(label, [(bid, df)])]
        for label, pparams in series_params:
            runs = list(_loaders[metric](pparams))
            if runs:
                ts_ids.append((label, runs))
        if not ts_ids:
            logger.warning(f"time-resolved plots ({metric}): no time series found.")
            continue
        if len(ts_ids) > 1 and t0 is not None:
            ts_ids = up.align_cycles(ts_ids, lambda df: up.t0_fingerprint(df, t0),
                                     what=f"{metric} time series")
        series = [(label, [df for _bid, df in runs]) for label, runs in ts_ids if runs]
        if not series:
            continue
        # Fixed colour per recommender in every figure of this block.
        up._ACTIVE_LABELS = [label for label, _ in series]

        if "venue_velocity_fixed_top" in wanted:
            up.plot_venue_velocity_fixed(series, which="top", filters=filters,
                                         show=show, top_n=top_n, metric=metric)
            if up.VELOCITY_EXTRA:
                up.plot_venue_velocity_fixed(series, which="mid", filters=filters,
                                             show=show, top_n=top_n, metric=metric)
        if "venue_velocity_fixed_worst" in wanted:
            up.plot_venue_velocity_fixed(series, which="worst", filters=filters,
                                         show=show, top_n=top_n, metric=metric)
        if "venue_velocity_fixed_final" in wanted:
            up.plot_venue_velocity_fixed(series, which="final", filters=filters,
                                         show=show, top_n=top_n, metric=metric)
            if up.VELOCITY_EXTRA:
                for _w in ("worst_final", "mid_final"):
                    up.plot_venue_velocity_fixed(series, which=_w, filters=filters,
                                                 show=show, top_n=top_n, metric=metric)
        if "top10_turnover" in wanted:
            up.plot_top10_turnover(series, filters=filters, show=show,
                                   top_n=top_n, metric=metric)
            if metric == "explorers" and up.ZOOM_WINDOW:
                up.plot_top10_turnover_window(series, filters=filters, show=show,
                                              top_n=top_n, metric=metric)
        if "kendall_over_time" in wanted:
            up.plot_kendall_over_time(series, filters=filters, show=show,
                                      top_n=top_n, metric=metric)
            if metric == "explorers" and up.ZOOM_WINDOW:
                up.plot_kendall_over_time_window(series, filters=filters, show=show,
                                                 top_n=top_n, metric=metric)
        if "top10_rank_tracking" in wanted:
            for ref in ("t0", "final"):
                up.plot_fixed_set_rank_tracking(series, ref=ref, filters=filters,
                                                show=show, top_n=top_n,
                                                metric=metric)
        if "top10_membership_panels" in wanted:
            up.plot_top10_membership_panels(series, filters=filters, show=show,
                                            top_n=top_n, metric=metric)
        if "rank_origin" in wanted:
            up.plot_rank_origin(series, filters=filters, show=show, metric=metric)

        if ("grid_final_comparison" in wanted and not single_dataset
                and metric in up.GRID_METRICS):
            # Optional subset/order of the series (e.g. ge + five values of f).
            if up.GRID_FINAL_SERIES:
                by_label = dict(series)
                gseries = [(lb, by_label[lb]) for lb in up.GRID_FINAL_SERIES
                           if lb in by_label]
            else:
                gseries = series
            cns = {cn_of.get(label) for label, _ in gseries}
            if not 2 <= len(gseries) <= 6:
                logger.debug(f"grid_final_comparison: drawn with 2 to 6 series "
                            f"(here {len(gseries)}; set utils_plot.GRID_FINAL_SERIES "
                            f"to choose them) — skipped.")
            elif len(cns) != 1 or None in cns:
                logger.warning(f"grid_final_comparison: different/unknown "
                               f"cell_number {cns} — skipped.")
            else:
                cn = cns.pop()
                up.plot_grid_final_comparison(gseries, cn, filters=filters,
                                              show=show, metric=metric,
                                              mask_center=mask_center)
                # Single run per series, all from the SAME cycle.
                if t0 is not None:
                    g_labels = {label for label, _ in gseries}
                    single = up.pick_same_cycle_runs(
                        [(lb, r) for lb, r in ts_ids if lb in g_labels], t0)
                    order = [label for label, _ in gseries if label in single]
                    if len(order) == len(gseries):
                        up.plot_grid_final_comparison(
                            [(label, [single[label][1]]) for label in order], cn,
                            filters=filters, show=show, metric=metric,
                            mask_center=mask_center, name_suffix="_single_run",
                            run_notes={label: f"run {single[label][0]}"
                                       for label in order})

        if ("aggregate_grid_snapshots" in wanted and single_dataset
                and metric in up.GRID_METRICS):
            for label, dfs in series:
                if cn_of.get(label):
                    up.plot_aggregate_grid_snapshots(
                        dfs, cn_of[label], filters=filters, show=show,
                        mask_center=mask_center, metric=metric)


class Simulation:
    """
    One dataset of the g-EPR model.  It is driven by model.runner (main.py);
    the methods are used in this order:

    Phase I, baseline (0 -> t0)
        run(params, grid_params)      simulate N individuals from scratch
        save_raw_data()               trajectories -> raw_data/
        run_analysis(...)             statistics   -> analysis_data/

    Phase II, continuation (t0 -> t1, with a recommender)
        _reconstruct_users_from_raw(params + batch_id, grid_params)
                                      rebuild the individuals of one baseline
        resume(params, grid_params)   continue them up to params["time"] (t1,
                                      an absolute clock, not a duration)
        save_combined_raw_data()      -> combined_raw_data/
        run_analysis(..., combined=True)  -> combined_analysis_data/

    Figures
        load_analysis_data(params) + plot(plot_list)                one series
        load_multiple_plot_data(...) + multipleplot(plot_list)      several series

    params holds the model parameters (ro, gamma, a_param, b_param, ...),
    algorithm ("ce", "ge", "de" for the baseline; "ge", "uk", "ik", "mf", "ra"
    for the continuation) and, for the
    recommenders, top_k, score_k, cf_updates, num_latent_factors.
    """

    def __init__(self):
        # Populated during run() -- holds User objects
        self.users_list = []

        # Populated during load_analysis_data() -- used by plot()
        self.rg_list = []
        self.two_rg_list = []
        self.venue_history_list = []
        self.all_rg_over_time = []
        self.all_s_over_time = []
        self.mean_dist_list = []

        # Entropy lists -- populated during load_analysis_data()
        self.s_rand_list = []
        self.s_unc_list  = []
        self.s_real_list = []

        # Maximum-predictability lists -- populated during load_analysis_data()
        self.pi_rand_list = []
        self.pi_unc_list  = []
        self.pi_real_list = []

        # Per-user 2rg / rg ratio -- populated during load_analysis_data()
        self.two_rg_ratio_list = []

        # Multi-series data for multipleplot ("mp") mode
        # Each element is a dict: {"label": str, "rg_list": [...], "two_rg_list": [...],
        #   "mean_dist_list": [...], "rg_distribution": [...], "all_rg_over_time": [...],
        #   "all_s_over_time": [...]}
        self._mp_series = []

        # Single-user references for mobility / freq_distribution plots
        self.last_user_history = None
        self.last_venue_history = None
        self.last_stats = None

        # Aggregate grids -- populated during load_analysis_data()
        self.aggregate_grid            = None
        self.aggregate_grid_user_count = 0
        self.batch_grids               = []

        # Exploration aggregate grids (distinct users per cell) -- populated
        # during load_analysis_data() when an exploration grid plot is requested.
        self.aggregate_grid_exploration            = None
        self.aggregate_grid_exploration_user_count = 0
        self.batch_grids_exploration               = []

        # Batch-level stats -- populated during load_analysis_data()
        self.batch_stats_list = []

        # Set during run(), needed by User objects at decision time
        self.ro    = None
        self.gamma = None

        # d-EPR: global, shared attractiveness matrix built during run()
        # (algorithm == "de"); None for every other algorithm.
        self.d_weights = None

        # Stored after run() for save_raw_data() folder naming
        self._simulate_params = None

        # Stored after load_analysis_data() for plot() titles/filters
        self._plot_params = None

    # ==========================================================================
    # PHASE 1: SIMULATE
    # ==========================================================================

    def run(self, simulate_params, grid_params=None):
        """
        Run the EPR mobility simulation from scratch and store results in self.users_list.

        Only EPR algorithms are accepted here. CF algorithms ("uk", "ik", "mf",
        "ra") require pre-existing trajectory data
        and must be launched via resume() after _reconstruct_users_from_raw().

        Parameters
        ----------
        simulate_params : dict
            Must contain: "ro", "gamma", "a_param", "b_param", "user_number", "time".
            Must contain: "algorithm" -- one of:
              "ce"  : classic EPR (no grid)
              "ge"  : grid EPR
              "de"  : d-EPR -- same as "ge" except exploration picks the next
                      venue via a gravity-model probability law over a global,
                      shared cell-attractiveness matrix (see _compute_d_weights
                      and User._gravity_exploration_venue).
            CF algorithms raise ValueError here.
        grid_params : dict, optional
            Required when algorithm is "ge" or "de". Must contain "cell_lenght" and
            "cell_number".
        """
        ro           = simulate_params["ro"]
        gamma        = simulate_params["gamma"]
        a_param      = simulate_params["a_param"]
        b_param      = simulate_params["b_param"]
        user_number  = simulate_params["user_number"]
        time         = simulate_params["time"]
        algorithm         = simulate_params["algorithm"]
        starting_position = simulate_params["starting_position"]
        # Power-law support bounds (default model.utils.POWERLAW_DEFAULTS).
        spatial_min  = simulate_params.get("spatial_min", _PL["spatial_min"])
        spatial_max  = simulate_params.get("spatial_max", _PL["spatial_max"])
        temporal_min = simulate_params.get("temporal_min", _PL["temporal_min"])
        temporal_max = simulate_params.get("temporal_max", _PL["temporal_max"])

        if algorithm not in ("ce", "ge", "de"):
            raise ValueError(
                f"run(): algorithm='{algorithm}' is a CF algorithm and cannot be used "
                f"to start a simulation from scratch. "
                f"Use 'ce', 'ge' or 'de' for run(), then load the saved batch and call "
                f"resume() with algorithm='{algorithm}'."
            )

        # Derive grid flag from algorithm
        grid_exists = algorithm != "ce"

        # When grid is disabled every user starts at (0,0) regardless of what
        # starting_position says.  Normalise to "centered" before storing params
        # so that the saved folder name reflects the actual simulation behaviour.
        if not grid_exists and starting_position != "centered":
            logger.warning(
                f"run(): algorithm='ce' (no grid) but starting_position='{starting_position}' — "
                f"all users will start at (0,0). Overriding to 'centered' for folder naming."
            )
            starting_position = "centered"
            simulate_params = {**simulate_params, "starting_position": "centered"}

        sp_abbr(starting_position)          # validates the value
        logger.debug(f"Starting position: '{starting_position}'")

        # Random seed of the run.  Given -> the run is reproducible (same seed,
        # parameters and code -> same trajectories); absent -> drawn at random.
        # Either way it is stored in simulate_params, hence in run_info.txt.
        seed = _resolve_seed(simulate_params)
        simulate_params = {**simulate_params, "seed": seed}
        np.random.seed(seed)              # main-process draws (replacement users)
        logger.debug(f"Random seed: {seed}")

        # Store on self so User objects can access ro/gamma via self.simulation,
        # and save_raw_data() can build the correct folder name
        self.ro               = ro
        self.gamma            = gamma
        self.a_param          = a_param
        self.b_param          = b_param
        self._simulate_params = simulate_params
        self.grid_exists      = grid_exists
        self._starting_position = starting_position
        self.spatial_min      = spatial_min
        self.spatial_max      = spatial_max
        self.temporal_min     = temporal_min
        self.temporal_max     = temporal_max

        if grid_exists:
            self._grid_params = grid_params
            self.grid_cell_lenght = grid_params["cell_lenght"]
            self.grid_cell_number = grid_params["cell_number"]
            if starting_position in ("random", "core_periphery"):
                # Initial positions are spread over the whole grid.
                self._spatial_center = (self.grid_cell_lenght * self.grid_cell_number) / 2
                #This definition is done following the grid logic, where the grid is defined from (0,0) to
                #(cell_lenght*cell_number, cell_lenght*cell_number), so the center is in the middle of the grid, not at (0,0)

        n_workers = _default_n_workers(user_number)
        logger.debug(
            f"Starting simulation for {user_number} users (algorithm='{algorithm}'), "
            f"using {n_workers} worker process(es)..."
        )

        # ---------------------------------------------------------------
        # EPR loop: create users and run them to full time, in parallel.
        # Each task builds and runs one user end-to-end; users never share
        # any mutable state (each gets its own private Grid instance), so
        # this is safe. See the module-level note above _simulate_one_user
        # for why a lightweight sim_stub is used instead of `self`, and
        # why every task carries its own RNG seed.
        # ---------------------------------------------------------------
        sim_stub = _SimParamsStub(
            ro, gamma,
            getattr(self, "grid_cell_lenght", None),
            getattr(self, "grid_cell_number", None),
            spatial_min=spatial_min, spatial_max=spatial_max,
            temporal_min=temporal_min, temporal_max=temporal_max,
        )
        spatial_center = getattr(self, "_spatial_center", None)
        base_seed = seed

        # ---------------------------------------------------------------
        # d-EPR: precompute the global, shared attractiveness matrix d_i.
        # Must happen before dispatching tasks so every worker receives the
        # same matrix, and (for "random"/"core_periphery") so each user's initial
        # position is fixed centrally and matches the histogram used to
        # build the weights -- workers no longer resample it themselves.
        # ---------------------------------------------------------------
        precomputed_positions = None
        if algorithm == "de":
            weight_seed = (seed + 1_000_003) % (2**31)
            self.d_weights, precomputed_positions = _compute_d_weights(
                starting_position=self._starting_position,
                cell_lenght=self.grid_cell_lenght,
                cell_number=self.grid_cell_number,
                spatial_center=spatial_center,
                user_number=user_number,
                seed=weight_seed,
            )
            logger.debug(
                f"d-EPR: built global weight matrix "
                f"(starting_position='{self._starting_position}', "
                f"sum={self.d_weights.sum():.0f}, "
                f"nonzero_cells={np.count_nonzero(self.d_weights)}/{self.d_weights.size})."
            )
        else:
            self.d_weights = None

        tasks = [
            (i, a_param, b_param, grid_exists, sim_stub,
             self._starting_position, spatial_center, time,
             (base_seed + i) % (2**32 - 1),
             self.d_weights,
             precomputed_positions[i] if precomputed_positions is not None else None)
            for i in range(user_number)
        ]

        results = {}
        completed = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_simulate_one_user, t): t[0] for t in tasks}
            prog = model.utils.Progress("simulation", user_number, "individuals")
            for future in model.utils.completed_with_progress(futures, prog):
                idx = futures[future]
                results[idx] = future.result()
                completed += 1
                prog.update(completed)

        self.users_list = [results[i] for i in range(user_number)]

        # ---------------------------------------------------------------
        # Post-run: replace users with empty user_history
        # ---------------------------------------------------------------
        replacement_uid = user_number
        users_replaced = 0
        valid_users = []

        for user in self.users_list:
            if len(user.user_history) > 1:
                valid_users.append(user)
            else:
                logger.warning(
                    f"{user.uid}: user_history is empty — discarding and replacing."
                )
                while True:
                    init_pos = _initial_position(self._starting_position,
                                                 self._spatial_center)

                    replacement = model.user.User(
                        uid=f"User_{replacement_uid}",
                        a_param=a_param,
                        b_param=b_param,
                        grid=grid_exists,
                        simulation=sim_stub,
                        initial_position=init_pos,
                        d_weights=self.d_weights,
                    )
                    while replacement.clock < time:
                        replacement.make_decision()

                    replacement_uid += 1

                    if len(replacement.user_history) > 1:
                        valid_users.append(replacement)
                        users_replaced += 1
                        break
                    else:
                        logger.warning(
                            f"User_{replacement_uid - 1}: replacement also empty, retrying..."
                        )

        self.users_list = valid_users

        if users_replaced > 0:
            logger.debug(
                f"Replaced {users_replaced} empty user(s) with valid replacements."
            )
        # Known limitation: each user's last step ends past `time` (overshoot).
        # It is not trimmed here (user_history, venue_history and the grid
        # would all have to be rolled back); the time series are clipped at
        # t1 when they are analysed.

    def resume(self, simulate_params, grid_params=None):
        """
        Continue the simulation on users already in self.users_list.

        Users are advanced until their clock reaches simulate_params["time"],
        which is interpreted as an *absolute* target clock, not an additional
        duration.  This means the value should be larger than the clock values
        the users already carry (i.e. the end-time of the previous run phase).

        The algorithm is taken from simulate_params["algorithm"] and must be
        one of: "ce", "ge", "de", "uk", "ik", "mf", "ra".

        Parameters
        ----------
        simulate_params : dict
            Must contain: "ro", "gamma", "a_param", "b_param", "time", "algorithm".
            Optional: "top_k" (int, default 10) for CF algorithms.
            "starting_position" and "user_number" must NOT be included:
            they are inherited from the loaded simulation.
        grid_params : dict, optional
            When the simulation uses a grid, must contain "cell_lenght" and
            "cell_number".  When None (default) self._grid_params is reused
            if it exists.
        """
        if not self.users_list:
            raise RuntimeError(
                "resume() called but self.users_list is empty. "
                "Call run() or _reconstruct_users_from_raw() first."
            )

        ro           = simulate_params["ro"]
        gamma        = simulate_params["gamma"]
        time_target  = simulate_params["time"]
        algorithm    = simulate_params["algorithm"]
        top_k        = simulate_params.get("top_k", 10)
        # A1: number of times the CF neighbour/similarity structures are rebuilt
        # over the resume window (n_intervals = cf_updates + 1). Configurable;
        # defaults to the historical hardcoded value of 6.
        cf_updates   = simulate_params.get("cf_updates", 6)
        # A2: truncated-SVD rank used by Matrix Factorization ("mf").
        # Configurable; defaults to the historical hardcoded value of 20.
        num_latent_factors = simulate_params.get("num_latent_factors", 20)
        # k: number of terms in the score of uk / ik (thesis, Section 3.2).
        score_k = int(simulate_params.get("score_k", model.utils_cf.SCORE_K_DEFAULT))
        # Power-law support bounds for the resume phase. Not saved in the
        # baseline on disk, so they cannot be recovered from the loaded batch:
        # model.runner passes those of the baseline.
        spatial_min  = simulate_params.get("spatial_min", _PL["spatial_min"])
        spatial_max  = simulate_params.get("spatial_max", _PL["spatial_max"])
        temporal_min = simulate_params.get("temporal_min", _PL["temporal_min"])
        temporal_max = simulate_params.get("temporal_max", _PL["temporal_max"])

        is_cf = algorithm not in ("ce", "ge", "de")

        # grid and starting_position are inherited from the loaded simulation,
        # not taken from simulate_params.
        starting_position = self._starting_position

        # Inject into the params dict so that folder naming helpers (_make_run_folder,
        # _make_combined_run_folder) have access to them via self._simulate_params.
        # The random seed of the continuation (given or drawn) is stored too, so
        # it ends up in run_info.txt.
        seed = _resolve_seed(simulate_params)
        simulate_params = {**simulate_params, "starting_position": starting_position,
                           "seed": seed}
        if algorithm in ("uk", "ik"):
            simulate_params["score_k"] = score_k
        logger.debug(f"Random seed: {seed}")

        self.ro               = ro
        self.gamma            = gamma
        self._simulate_params = simulate_params

        # Resolve grid_params
        if grid_params is not None:
            self._grid_params     = grid_params
            self.grid_cell_lenght = grid_params["cell_lenght"]
            self.grid_cell_number = grid_params["cell_number"]
        elif getattr(self, "_grid_params", None) is not None:
            grid_params = self._grid_params  # reuse from previous run cause it must match the loaded state

        n_users   = len(self.users_list)

        # Compute the effective start of this resume window from the users' clocks
        current_clocks = [u.clock for u in self.users_list]
        resume_start   = sum(current_clocks) / n_users   # approximate; each user may differ
        extra_time     = time_target - resume_start

        if extra_time <= 0:
            logger.warning(
                f"resume() called but time_target={time_target} is not larger than "
                f"the current mean user clock ({resume_start:.1f}). Nothing to do."
            )
            return

        logger.debug(
            f"Resuming simulation: {n_users} users, "
            f"clock range [{min(current_clocks):.1f}, {max(current_clocks):.1f}] "
            f"→ target={time_target}, algorithm='{algorithm}'"
        )

        # Detach every user from the live Simulation *before* any parallel
        # dispatch below: user.simulation only needs to expose .ro/.gamma
        # (see _SimParamsStub), and keeping the real Simulation reference
        # would drag the whole users_list -- and therefore every other
        # user -- through pickle for each individual task.
        sim_stub = _SimParamsStub(
            ro, gamma,
            getattr(self, "grid_cell_lenght", None),
            getattr(self, "grid_cell_number", None),
            spatial_min=spatial_min, spatial_max=spatial_max,
            temporal_min=temporal_min, temporal_max=temporal_max,
        )
        for user in self.users_list:
            user.simulation = sim_stub

        n_workers = _default_n_workers(n_users)
        base_seed = seed

        if not is_cf:
            # ---------------------------------------------------------------
            # Pure EPR resume, parallel across users (each user only ever
            # touches its own history/grid, so this is safe).
            # ---------------------------------------------------------------
            tasks = [
                (self.users_list[i], time_target, (base_seed + i) % (2**32 - 1))
                for i in range(n_users)
            ]
            results = {}
            completed = 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_resume_epr_one_user, t): i for i, t in enumerate(tasks)}
                prog = model.utils.Progress("simulation", n_users, "individuals")
                for future in model.utils.completed_with_progress(futures, prog):
                    idx = futures[future]
                    results[idx] = future.result()
                    completed += 1
                    prog.update(completed)
            self.users_list = [results[i] for i in range(n_users)]

        else:
            # ---------------------------------------------------------------
            # CF resume: run collaborative filtering for the full resume window.
            # Similarities are rebuilt once per interval (a natural, unavoidable
            # sync point since they depend on the whole population); within an
            # interval every user advances independently and in parallel
            # against a frozen, read-only neighbor snapshot. The pool is
            # created once and reused across all intervals to avoid repeatedly
            # paying process-startup cost.
            # ---------------------------------------------------------------
            n_intervals  = cf_updates + 1
            cf_duration  = time_target - resume_start
            interval_len = cf_duration / n_intervals

            check_run_dir = None  # created on first interval, reused afterwards
            # Generate the batch timestamp once so that the check folder shares
            # the same ID as the raw/analysis batch saved for this run.
            batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._batch_ts = batch_ts   # store so save_raw_data / run_analysis can reuse it

            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                for interval in range(n_intervals):
                    t_end = resume_start + (interval + 1) * interval_len
                    logger.debug(
                        f"Resume CF interval {interval + 1}/{n_intervals}: "
                        f"rebuilding similarities, running to {t_end:.1f}..."
                    )

                    # Freeze venue_history for all users at the start of this
                    # interval so that build_cf_data sees a symmetric snapshot:
                    # every user is represented by the same temporal baseline,
                    # regardless of the order in which they will be simulated.
                    frozen_users = [
                        _VenueSnapshot(user.venue_history)
                        for user in self.users_list
                    ]

                    cf_neighbors, check_run_dir = model.utils_cf.build_cf_data(
                        users_list     = frozen_users,
                        top_k          = top_k,
                        cf_type        = algorithm,
                        simulate_params= self._simulate_params,
                        grid_exists    = self.grid_exists,
                        grid_params    = getattr(self, "_grid_params", None),
                        check_run_dir  = check_run_dir,
                        interval_idx   = interval,
                        batch_ts       = batch_ts,
                        num_latent_factors = num_latent_factors,
                        score_k        = score_k,
                    )

                    tasks = [
                        (self.users_list[i], cf_neighbors[i], algorithm, t_end,
                         (base_seed + interval * n_users + i) % (2**32 - 1))
                        for i in range(n_users)
                    ]
                    results = {}
                    futures = {executor.submit(_resume_cf_one_user, t): i for i, t in enumerate(tasks)}
                    prog = model.utils.Progress(
                        f"update {interval + 1}/{n_intervals}", n_users, "individuals")
                    for future in model.utils.completed_with_progress(futures, prog):
                        idx = futures[future]
                        results[idx] = future.result()
                        prog.update(len(results))
                    self.users_list = [results[i] for i in range(n_users)]
                    short = _log_cf_interval_summary(
                        self.users_list, interval + 1, n_intervals, algorithm
                    )
                    health = model.utils_cf.compute_recommender_health(cf_neighbors)
                    if health is not None:
                        short += f", peakiness {health['peakiness_median']:.3f}"
                    logger.info(f"  {algorithm} update {interval + 1}/{n_intervals}: {short}")
                    logger.debug(
                        f"Resume CF interval {interval + 1}/{n_intervals}: done "
                        f"({n_users}/{n_users} users)"
                    )

        for user in self.users_list:
            header    = user.user_history[0]
            clock_idx = header.index("Clock")
            state_idx = header.index("State(E/R)")
            if len(user.user_history) > 1 and user.user_history[-1][clock_idx] > time_target:
                last_row = user.user_history.pop()
                if last_row[state_idx] == "E":
                    user.venue_history.pop()

        logger.debug("Resume complete.")


    def _reconstruct_users_from_raw(self, load_params, grid_params=None):
        """
        Load a raw_data batch from disk and reconstruct live User objects so that
        resume() can continue the simulation from the saved state.

        The method locates the correct parameter folder in raw_data/ using
        load_params, then selects a specific batch sub-folder (by batch_id or
        automatically the most recent one), reads the Parquet files for every
        user, and rebuilds each User object in memory.

        Reconstructed state per user
        ----------------------------
        - user_history     : full list-of-lists, header row prepended
        - venue_history    : full list-of-lists, header row prepended
        - _venue_index     : rebuilt from venue_history
        - num_distinct_venues : len(venue_history) - 1
        - clock            : last Clock value in user_history
        - position         : (Destination_x_coord, Destination_y_coord) of last row
        - previous_position: (Departure_x_coord, Departure_y_coord) of last row
        - departure        : Departure field of last row
        - state            : State(E/R) field of last row
        - grid.matrix      : loaded from user_{i}_grid.parquet when grid is active
        - d_weights        : global, shared d-EPR matrix reloaded from
                              d_weights.parquet (algorithm="de" only; None otherwise)
        - simulation       : reference to self (re-attached here)

        Parameters
        ----------
        load_params : dict
            Standard simulation parameter keys: "ro", "gamma", "a_param",
            "b_param", "time", "algorithm", "starting_position" (optional).
            Additional key:
                "batch_id" (str, optional)
                    Exact timestamp folder name to load (e.g. "20240601_120000_123456").
                    When absent, the most-recent batch folder is selected automatically.
        grid_params : dict, optional
            Required when load_params["algorithm"] is not "ce".
            Must contain "cell_lenght" and "cell_number".
            When None, self._grid_params is used if available.

        Returns
        -------
        str
            Path to the raw_data batch folder that was loaded, so it can be
            passed to run_analysis() or save_raw_data() if needed.

        Raises
        ------
        FileNotFoundError
            When the parameter folder or the requested batch folder does not exist.
        RuntimeError
            When no batch folders are found inside the parameter folder.
        """

        # Resolve grid_params
        if grid_params is not None:
            self._grid_params     = grid_params
            self.grid_cell_lenght = grid_params["cell_lenght"]
            self.grid_cell_number = grid_params["cell_number"]

        algorithm         = load_params["algorithm"]
        grid_exists       = algorithm != "ce"
        starting_position = load_params["starting_position"]
        batch_id          = load_params.get("batch_id", None)

        # When grid is disabled starting_position has no effect; normalise to
        # "centered" so the folder lookup matches what run() would have saved.
        if not grid_exists and starting_position != "centered":
            logger.warning(
                f"_reconstruct_users_from_raw(): algorithm='ce' (no grid) but "
                f"starting_position='{starting_position}' — overriding to 'centered' "
                f"for folder lookup."
            )
            starting_position = "centered"
            load_params = {**load_params, "starting_position": "centered"}

        # ------------------------------------------------------------------
        # Locate the parameter folder in raw_data/
        # ------------------------------------------------------------------
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_folder  = os.path.join(project_root, "raw_data")

        alg = load_params["algorithm"]
        param_folder_name = model.utils.build_param_folder_name(
            load_params, grid_params if grid_exists else None)

        param_folder = os.path.join(base_folder, param_folder_name)
        if not os.path.exists(param_folder):
            raise FileNotFoundError(
                f"_reconstruct_users_from_raw: parameter folder not found:\n  {param_folder}"
            )

        # ------------------------------------------------------------------
        # Select the batch sub-folder
        # ------------------------------------------------------------------
        if batch_id is not None:
            run_folder = os.path.join(param_folder, batch_id)
            if not os.path.exists(run_folder):
                raise FileNotFoundError(
                    f"_reconstruct_users_from_raw: requested batch_id not found:\n  {run_folder}"
                )
        else:
            # Pick the most-recent batch
            candidates = sorted(
                d for d in os.listdir(param_folder)
                if os.path.isdir(os.path.join(param_folder, d))
            )
            if not candidates:
                raise RuntimeError(
                    f"_reconstruct_users_from_raw: no batch folders found in:\n  {param_folder}"
                )
            run_folder = os.path.join(param_folder, candidates[-1])
            logger.debug(
                f"batch_id not specified — using most recent batch: {candidates[-1]}"
            )

        logger.debug(f"Reconstructing users from: {run_folder}")

        # ------------------------------------------------------------------
        # Discover how many users are in this batch
        # ------------------------------------------------------------------
        user_indices = sorted(
            int(fname[len("user_"):-len("_history.parquet")])
            for fname in os.listdir(run_folder)
            if fname.startswith("user_") and fname.endswith("_history.parquet")
        )
        if not user_indices:
            raise RuntimeError(
                f"_reconstruct_users_from_raw: no user_*_history.parquet files in:\n  {run_folder}"
            )

        # ------------------------------------------------------------------
        # Restore simulation-level attributes needed by User objects
        # ------------------------------------------------------------------
        self.ro               = load_params["ro"]
        self.gamma            = load_params["gamma"]
        self.a_param          = load_params["a_param"]
        self.grid_exists      = grid_exists
        self._starting_position = starting_position

        if grid_exists and grid_params is not None:
            self.grid_cell_lenght = grid_params["cell_lenght"]
            self.grid_cell_number = grid_params["cell_number"]

        # ------------------------------------------------------------------
        # d-EPR: reload the global, shared attractiveness matrix if present
        # ------------------------------------------------------------------
        if alg == "de":
            d_weights_path = os.path.join(run_folder, "d_weights.parquet")
            if os.path.exists(d_weights_path):
                self.d_weights = pd.read_parquet(d_weights_path).values.astype(float)
            else:
                logger.warning(
                    f"_reconstruct_users_from_raw(): algorithm='de' but "
                    f"d_weights.parquet not found in {run_folder} — "
                    f"reconstructed users will fall back to grid power-law exploration."
                )
                self.d_weights = None
        else:
            self.d_weights = None

        self.users_list = []
        n_users   = len(user_indices)
        prog      = model.utils.Progress("reconstruction", n_users, "individuals")

        history_header = [
            "State(E/R)", "Departure",
            "Departure_x_coord", "Departure_y_coord",
            "Destination", "Destination_x_coord", "Destination_y_coord",
            "Distance", "Angle", "Delta_t", "Clock",
        ]

        for pos, i in enumerate(user_indices):
            h_path = os.path.join(run_folder, f"user_{i}_history.parquet")
            v_path = os.path.join(run_folder, f"user_{i}_venue.parquet")
            g_path = os.path.join(run_folder, f"user_{i}_grid.parquet")

            history_df = pd.read_parquet(h_path)
            venue_df   = pd.read_parquet(v_path) if os.path.exists(v_path) else None

            # Rebuild history as list-of-lists (matching the live format)
            user_history  = [history_header] + history_df.values.tolist()

            venue_header  = list(venue_df.columns)
            venue_history = [venue_header] + venue_df.values.tolist()
            
            # Last row of history -> current state
            last = user_history[-1]
            col  = {name: idx for idx, name in enumerate(history_header)}

            last_clock    = last[col["Clock"]]
            last_position = (last[col["Destination_x_coord"]], last[col["Destination_y_coord"]])
            last_prev_pos = (last[col["Departure_x_coord"]],   last[col["Departure_y_coord"]])
            last_departure= last[col["Departure"]]
            last_state    = last[col["State(E/R)"]]

            # Rebuild _venue_index: venue_name -> row index in venue_history
            venue_index = {}
            for row_idx, row in enumerate(venue_history[1:], start=1):
                venue_index[row[0]] = row_idx

            # ------------------------------------------------------------------
            # Build a minimal User object without calling first_exploration()
            # We achieve this by calling __new__ and populating attributes manually
            # ------------------------------------------------------------------
            user = model.user.User.__new__(model.user.User)
            user.uid          = f"User_{i}"
            user.a_param      = load_params["a_param"]
            user.b_param      = load_params["b_param"]
            user.simulation   = self

            # d-EPR: same global, shared matrix reloaded once above for the
            # whole batch (None for every other algorithm).
            user.d_weights    = self.d_weights

            user.has_grid = grid_exists
            if grid_exists and os.path.exists(g_path):
                grid_df = pd.read_parquet(g_path)
                g = model.grid.Grid.__new__(model.grid.Grid)
                g.cell_lenght    = grid_params["cell_lenght"]
                g.cell_number    = grid_params["cell_number"]
                g.spatial_center = (g.cell_lenght * g.cell_number) / 2
                g.matrix         = grid_df.values.astype(int)
                user.grid = g
            elif grid_exists:
                logger.warning(f"User_{i}: grid.parquet not found, creating fresh grid.")
                user.grid = model.grid.Grid(
                    cell_lenght=grid_params["cell_lenght"],
                    cell_number=grid_params["cell_number"],
                )

            user.user_history       = user_history
            user.venue_history      = venue_history
            user._venue_index       = venue_index
            user.num_distinct_venues= len(venue_history) - 1

            user.clock              = last_clock
            user.position           = last_position
            user.previous_position  = last_prev_pos
            user.departure          = last_departure
            user.state              = last_state

            # Transient movement attributes (not needed to resume, but must exist)
            user.next_position      = None
            user.destination        = None
            user.next_venue_return  = None
            user.delta_r            = None
            user.angle              = None
            user.delta_t            = None
            user.r_gyr              = None
            user.stats              = [["metric", "value"]]
            user._row_index         = None

            self.users_list.append(user)

            prog.update(pos + 1)

        logger.debug(
            f"Reconstructed {n_users} users from batch: {os.path.basename(run_folder)}"
        )

        # Store so that save_combined_raw_data() and run_analysis() can build
        # the two-level folder structure without the caller having to pass it again.
        # The baseline's batch id and seed (from its run_info.txt) are kept too,
        # so the continuation's run_info.txt says which baseline it continues.
        load_params = {**load_params, "batch_id": os.path.basename(run_folder)}
        _seed0 = model.utils.read_run_info_value(run_folder, "seed")
        if _seed0 is not None:
            load_params["seed"] = _seed0
        self._load_params = load_params

        return run_folder

    def save_raw_data(self):
        """
        Serialise user_history and venue_history for every simulated user to
        Parquet files under raw_data/. When a grid is active, the grid matrix
        is also saved per user.

        Must be called after run().

        Folder layout
        -------------
        raw_data/
            ro_{ro}_gamma_{gamma}_a_{a_param}_b_{b_param}_time_{time}/
                {YYYYMMDD_HHMMSS_ffffff}/
                    [d_weights.parquet]    (only for algorithm="de"; one matrix per batch)
                    user_0_history.parquet
                    user_0_venue.parquet
                    [user_0_grid.parquet]   (only when grid is active)
                    ...

        Returns
        -------
        str
            Path to the timestamped run folder that was just created.
        """
        logger.debug("Saving raw simulation data...")

        if self.grid_exists:
            run_folder = self._make_run_folder("raw_data", self._simulate_params, self._grid_params)
        else:
            run_folder = self._make_run_folder("raw_data", self._simulate_params)

        # d-EPR: persist the global, shared attractiveness matrix once per
        # batch (not per-user, since it is identical for every user).
        if getattr(self, "d_weights", None) is not None:
            dw_df = pd.DataFrame(self.d_weights)
            dw_df.columns = dw_df.columns.astype(str)
            dw_df.to_parquet(os.path.join(run_folder, "d_weights.parquet"), index=False)

        total     = len(self.users_list)
        prog      = model.utils.Progress("saving", total, "individuals")

        for i, user in enumerate(self.users_list):
            user_df  = pd.DataFrame(user.user_history[1:],  columns=user.user_history[0])
            venue_df = pd.DataFrame(user.venue_history[1:], columns=user.venue_history[0])

            if user_df.empty:
                logger.warning(f"User_{i} (uid={user.uid}): user_history is empty — saving empty Parquet. "
                       f"This will cause errors on reconstruction. Consider increasing simulation time.")

            user_df.to_parquet( os.path.join(run_folder, f"user_{i}_history.parquet"), index=False)
            venue_df.to_parquet(os.path.join(run_folder, f"user_{i}_venue.parquet"),   index=False)

            if self.grid_exists:
                grid_df = pd.DataFrame(user.grid.matrix)
                grid_df.columns = grid_df.columns.astype(str)
                grid_df.to_parquet(os.path.join(run_folder, f"user_{i}_grid.parquet"), index=False)

            prog.update(i + 1)

        logger.debug(f"Raw data saved to: {run_folder}")
        model.utils.write_run_report(
            run_folder, "raw_data run",
            params=getattr(self, "_simulate_params", None),
            grid_params=getattr(self, "_grid_params", None))
        return run_folder

    def save_combined_raw_data(self):
        """
        Save raw trajectories for a chained simulation (load → resume) under
        combined_raw_data/ using a two-level parameter folder structure that
        makes the provenance explicit.

        Folder layout
        -------------
        combined_raw_data/
            <base_params>/          ← parameters of the loaded batch (load_params)
                <resume_params>/    ← parameters of the resumed simulation (_simulate_params)
                    {YYYYMMDD_HHMMSS_ffffff}/
                        [d_weights.parquet]    (only for algorithm="de")
                        user_0_history.parquet
                        user_0_venue.parquet
                        [user_0_grid.parquet]

        Must be called after _reconstruct_users_from_raw() and resume().

        Returns
        -------
        str
            Path to the timestamped run folder that was just created.
        """
        logger.debug("Saving combined raw simulation data...")

        if not hasattr(self, "_load_params") or self._load_params is None:
            raise RuntimeError(
                "save_combined_raw_data() requires _load_params to be set. "
                "Call _reconstruct_users_from_raw() before this method."
            )

        gp = getattr(self, "_grid_params", None)
        run_folder = self._make_combined_run_folder(
            "combined_raw_data",
            base_params   = self._load_params,
            resume_params = self._simulate_params,
            grid_params   = gp,
            batch_ts      = getattr(self, "_batch_ts", None),
        )

        # d-EPR: persist the global, shared attractiveness matrix once per
        # batch (carried over unchanged from the loaded simulation).
        if getattr(self, "d_weights", None) is not None:
            dw_df = pd.DataFrame(self.d_weights)
            dw_df.columns = dw_df.columns.astype(str)
            dw_df.to_parquet(os.path.join(run_folder, "d_weights.parquet"), index=False)

        total     = len(self.users_list)
        prog      = model.utils.Progress("saving", total, "individuals")

        for i, user in enumerate(self.users_list):
            user_df  = pd.DataFrame(user.user_history[1:],  columns=user.user_history[0])
            venue_df = pd.DataFrame(user.venue_history[1:], columns=user.venue_history[0])

            user_df.to_parquet( os.path.join(run_folder, f"user_{i}_history.parquet"), index=False)
            venue_df.to_parquet(os.path.join(run_folder, f"user_{i}_venue.parquet"),   index=False)

            if self.grid_exists:
                grid_df = pd.DataFrame(user.grid.matrix)
                grid_df.columns = grid_df.columns.astype(str)
                grid_df.to_parquet(os.path.join(run_folder, f"user_{i}_grid.parquet"), index=False)

            prog.update(i + 1)

        logger.debug(f"Combined raw data saved to: {run_folder}")
        _lp = getattr(self, "_load_params", None) or {}
        model.utils.write_run_report(
            run_folder, "combined_raw_data run (load -> resume)",
            params=getattr(self, "_simulate_params", None),
            grid_params=getattr(self, "_grid_params", None),
            extra_lines=[f"load {k} = {v}" for k, v in sorted(_lp.items(), key=str)])
        return run_folder

    # ==========================================================================
    # PHASE 2: ANALYSIS
    # ==========================================================================

    def run_analysis(self, analyse_params, raw_run_folder=None, grid_params=None, combined=False,
                     explicit_out=None, explicit_t0=None, users_override=None,
                     metrics=None):
        """
        Compute all analysis quantities from raw trajectory data and save them
        to analysis_data/.

        Calling patterns
        ----------------
        (a) Standalone (ANALYSE mode)
            raw_run_folder=None -- data loaded from raw_data/ via analyse_params.
            grid_params must be supplied when the simulation was run with a grid,
            so that the correct raw_data folder is located and the analysis_data
            folder is named consistently.

        (b) Chained (SIMULATE_AND_ANALYSE mode)
            Pass raw_run_folder returned by save_raw_data(). Users already in
            memory are used directly without any disk read.
            grid_params is taken from self._grid_params when not explicitly
            provided (automatically set during run()).

        Files saved per user
        --------------------
        user_{i}_history.parquet      : full user trajectory with added columns:
                                        'rg'  — radius of gyration at each step
                                        's'   — cumulative distinct venues (exploration rows only)
        user_{i}_venue.parquet        : venue visit counts   (copy from raw)
        user_{i}_stats.parquet        : rg, rg_two, x_cm, y_cm

        Parameters
        ----------
        analyse_params : dict
            Must contain: "ro", "gamma", "a_param", "b_param", "time".
            Note: "user_number" is intentionally absent -- analysis always
            processes complete run folders, never a subset of users.
        raw_run_folder : str, optional
            Path to a specific raw_data run folder (simulate_and_analyse mode).
        grid_params : dict, optional
            Grid parameters dict with keys "cell_lenght" and "cell_number".
            Required when the simulation used a grid. In simulate_and_analyse
            mode this is inferred automatically from self._grid_params.

        Returns
        -------
        str
            Path to the timestamped analysis run folder that was just created.
        """
        logger.debug("Starting analysis phase...")

        # In simulate_and_analyse mode, inherit grid_params from the run phase
        if grid_params is None and getattr(self, "_grid_params", None) is not None:
            grid_params = self._grid_params

        # Re-analysis-from-disk path (analyse_all / analyse_all_combined): the
        # caller hands us the users already read from a specific run folder, so
        # we bypass all filter/lock-based disk loading below.
        if users_override is not None:
            self.users_list = users_override

        # Load from disk only when there are no in-memory users
        if not self.users_list:
            #Directly callable. Main doesnt include this functionality
            if raw_run_folder is not None:
                logger.debug(f"Loading raw data from: {raw_run_folder}")
                i = 0
                while True:
                    h_path = os.path.join(raw_run_folder, f"user_{i}_history.parquet")
                    if not os.path.exists(h_path):
                        break
                    v_path = os.path.join(raw_run_folder, f"user_{i}_venue.parquet")
                    uh = pd.read_parquet(h_path)
                    vh = pd.read_parquet(v_path) if os.path.exists(v_path) else None
                    self.users_list.append((uh, vh))
                    i += 1
                logger.debug(f"Loaded {i} users from raw run folder")
            else:
                #Analyse mode: load from raw_data/ using analyse_params filters
                logger.debug("Loading raw data via analyse_params...")
                alg = analyse_params.get("algorithm", "ce")
                grid_active = alg != "ce"
                filters = {k: v for k, v in analyse_params.items()
                           if k not in ("algorithm", "starting_position")}
                if grid_active:
                    filters.update(grid_params) #Adding grid values
                loaded = model.utils.load_raw_parquet(
                    filters=filters,
                    grid=grid_active,
                    return_run_folders=True,
                )
                self.users_list = [entry["data"] for entry in loaded]
                logger.debug(f"Loaded {len(self.users_list)} users from raw_data")

        # Continuation (combined) analysis
        if combined:
            if explicit_out is not None:
                # Re-analysis from disk: caller supplies the exact output folder
                # and the baseline/continuation clock boundary t0, so neither
                # _load_params nor _make_combined_run_folder are needed.
                batch_run_folder = explicit_out
                _t0              = explicit_t0
            else:
                if not hasattr(self, "_load_params") or self._load_params is None:
                    raise RuntimeError(
                        "run_analysis(combined=True) requires _load_params to be set. "
                        "Call _reconstruct_users_from_raw() before this method."
                    )
                batch_run_folder = self._make_combined_run_folder(
                    "combined_analysis_data",
                    base_params   = self._load_params,
                    resume_params = analyse_params,
                    grid_params   = grid_params,
                    batch_ts      = getattr(self, "_batch_ts", None),
                )
                # The clock separating the baseline from the continuation
                _t0 = self._load_params["time"]
            analysis_folder = batch_run_folder
            os.makedirs(analysis_folder, exist_ok=True)
        else:
            _t0 = None
            if explicit_out is not None:
                analysis_folder = explicit_out
                os.makedirs(analysis_folder, exist_ok=True)
            else:
                alg_for_folder = analyse_params.get("algorithm", (getattr(self, "_simulate_params", None) or {}).get("algorithm", "ce"))
                grid_active    = alg_for_folder != "ce"
                if grid_active:
                    analysis_folder = self._make_run_folder("analysis_data", analyse_params, grid_params)
                else:
                    analysis_folder = self._make_run_folder("analysis_data", analyse_params)

        total     = len(self.users_list)

        # ------------------------------------------------------------------
        # Pass 1 (sequential, cheap): extract plain DataFrames from whatever
        # form each user is currently in (live User, dict, or tuple loaded
        # from raw_data) and resolve grid_df. This is just attribute/column
        # access, not the bottleneck. user_history_df_list / venue_history_df_list
        # are also kept around afterwards for the batch-stats computation, so we
        # don't need to re-read them from disk.
        # ------------------------------------------------------------------
        alg_key     = analyse_params.get("algorithm", (getattr(self, "_simulate_params", None) or {}).get("algorithm", "ce"))
        grid_active = alg_key != "ce"

        user_history_df_list  = []
        venue_history_df_list = []
        grid_df_list           = []

        for i, user_entry in enumerate(self.users_list):
            if isinstance(user_entry, model.user.User):
                user_history_df  = pd.DataFrame(
                    user_entry.user_history[1:],  columns=user_entry.user_history[0])
                venue_history_df = pd.DataFrame(
                    user_entry.venue_history[1:], columns=user_entry.venue_history[0])
            elif isinstance(user_entry, dict):
                user_history_df  = user_entry["user_history"]
                venue_history_df = user_entry["venue_history"]
            else:
                user_history_df, venue_history_df = user_entry

            user_history_df_list.append(user_history_df)
            venue_history_df_list.append(venue_history_df)

            grid_df = None
            if grid_active:
                if isinstance(user_entry, model.user.User) and user_entry.has_grid:
                    grid_df = pd.DataFrame(user_entry.grid.matrix)
                elif isinstance(user_entry, dict) and user_entry.get("grid") is not None:
                    grid_df = user_entry["grid"]
                if grid_df is not None:
                    grid_df.columns = grid_df.columns.astype(str)
            grid_df_list.append(grid_df)

        # ------------------------------------------------------------------
        # Pass 2 (parallel): the actual analysis. compute_stats calls
        # compute_real_entropy, an O(n^2) pure-Python loop over the
        # trajectory -- this single function is what dominates analysis
        # wall-clock time. Every user's computation is fully independent
        # of every other user's, so this scales close to linearly with the
        # number of CPU cores. Workers only receive plain DataFrames (never
        # a live User or the Simulation) and write their own parquet files
        # directly, so nothing heavy needs to travel back through pickle.
        # ------------------------------------------------------------------
        n_workers = _default_n_workers(total)
        logger.debug(f"Running analysis for {total} users using {n_workers} worker process(es)...")

        tasks = [
            (i, user_history_df_list[i], venue_history_df_list[i], grid_df_list[i],
             analysis_folder, _t0, SAVE_PER_USER_GRIDS)
            for i in range(total)
        ]

        completed = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_analyze_one_user, t): t[0] for t in tasks}
            prog = model.utils.Progress("analysis", total, "individuals")
            for future in model.utils.completed_with_progress(futures, prog):
                future.result()  # re-raises any worker exception in the main process
                completed += 1
                prog.update(completed)

        # ------------------------------------------------------------------
        # Batch-level summary stats and aggregate grids.
        # ------------------------------------------------------------------
        # Batch metrics are now driven by the uniform registry in model.utils:
        # every quantity (gini, means, Moran, and the new
        # collective-entropy metrics) is one registry entry computed from the
        # on-disk batch artifacts.  `metrics` selects which to compute:
        #   None  -> all (fresh analysis, full overwrite);
        #   list  -> only these, MERGED into the existing batch_stats.parquet.
        # The batch aggregate grids are (re)built in memory from the per-user
        # grid matrices and saved once, so grid metrics keep working even when
        # per-user grid parquets are not persisted (SAVE_PER_USER_GRIDS=False).
        _grid_active = analyse_params.get(
            "algorithm", (getattr(self, "_simulate_params", None) or {}).get("algorithm", "ce")) != "ce"
        _cl = grid_params.get("cell_lenght") if grid_params else None
        _cn = grid_params.get("cell_number") if grid_params else None

        def _compute_and_save_batch_stats(target_folder, grid_matrices):
            if _grid_active:
                _va, _ea = model.utils.aggregates_from_matrices(grid_matrices)
                if _va is not None:
                    _vadf = pd.DataFrame(_va); _vadf.columns = _vadf.columns.astype(str)
                    _vadf.to_parquet(os.path.join(target_folder, "aggregate_grid.parquet"), index=False)
                    _eadf = pd.DataFrame(_ea); _eadf.columns = _eadf.columns.astype(str)
                    _eadf.to_parquet(os.path.join(target_folder, "aggregate_grid_exploration.parquet"), index=False)
                    model.utils.rollup_param_aggregates(target_folder)

            ctx = model.utils.BatchContext(
                target_folder, cell_lenght=_cl, cell_number=_cn, n_users=total)
            rows = model.utils.compute_batch_metrics(
                ctx, metrics=metrics, grid_active=_grid_active)
            model.utils.write_batch_stats(
                target_folder, rows, merge=(metrics is not None))
            logger.debug(f"Analysis data saved to: {target_folder}")

        def _save_pop_snapshots(target_folder, t0):
            """Per-cell popularity at init / end-ge(t0) / final, saved as
            popularity_snapshots.parquet (used by the rank_persistence plots).
            Only meaningful for continuations (grid active, t0 known)."""
            if not _grid_active or t0 is None:
                return
            from collections import Counter
            c_init, c_t0, c_final = Counter(), Counter(), Counter()
            for uh in user_history_df_list:
                if uh is None or len(uh) == 0:
                    continue
                clk = np.asarray(uh["Clock"].to_numpy(), dtype=float)
                dst = uh["Destination"].to_numpy()
                dep = uh["Departure"].to_numpy()
                start = dep[int(np.argmin(clk))]
                c_init[start] += 1; c_t0[start] += 1; c_final[start] += 1
                for cl, de in zip(clk, dst):
                    if de is None or (isinstance(de, float) and np.isnan(de)):
                        continue
                    c_final[de] += 1
                    if cl <= t0:
                        c_t0[de] += 1
            cells = sorted(set(c_init) | set(c_t0) | set(c_final), key=str)
            snap_df = pd.DataFrame({
                "cell":  cells,
                "init":  [c_init.get(c, 0)  for c in cells],
                "at_t0": [c_t0.get(c, 0)    for c in cells],
                "final": [c_final.get(c, 0) for c in cells],
            })
            snap_df.to_parquet(
                os.path.join(target_folder, "popularity_snapshots.parquet"), index=False)
            logger.debug(f"Popularity snapshots saved to: {target_folder}")

        def _save_pop_timeseries(target_folder, t0, step=POP_SNAPSHOT_STEP):
            """Per-cell CUMULATIVE visit counts sampled every `step` clock units,
            saved as popularity_timeseries.parquet, plus the per-cell CUMULATIVE
            number of distinct explorers, saved as exploration_timeseries.parquet.

            One column per snapshot time t_j (j = step, 2*step, ... up to the last
            multiple <= max clock), named 't_{t_j}', holding the number of
            destination-visits to each cell with Clock <= t_j (plus 1 for each
            user whose START cell is that cell, mirroring _save_pop_snapshots).
            This single per-batch (per-run) artifact is the source for the
            time-resolved venue plots (accumulation velocity, top-k turnover/
            stability and Kendall-tau-over-time) and, for grid runs, for the
            aggregate-grid snapshot heatmaps.  It is a within-realization
            quantity: it is NEVER summed across batches at this stage — each run
            writes its own file and the plots aggregate the per-run curves with a
            mean +/- error band, exactly like the rank_persistence pipeline.

            Grid runs only (t0 known, cells are 'Cell_(x,y)').  Cheap single pass:
            each jump is bucketed into the first snapshot >= its clock, then a
            cumulative sum along the time axis turns per-interval increments into
            cumulative counts.
            """
            if not _grid_active or t0 is None:
                return
            # Visits AND explorers (distinct users) built in one vectorised
            # pass by the shared builder in model.utils.
            vis_df, exp_df = model.utils.build_popularity_timeseries(
                user_history_df_list, step=step)
            if vis_df is None:
                logger.warning("popularity_timeseries: no usable clock/visits — skipped.")
                return
            vis_df.to_parquet(
                os.path.join(target_folder, "popularity_timeseries.parquet"), index=False)
            exp_df.to_parquet(
                os.path.join(target_folder, "exploration_timeseries.parquet"), index=False)
            logger.debug(
                f"Popularity + exploration time series ({vis_df.shape[1] - 1} snapshots, "
                f"step={int(step)}) saved to: {target_folder}")

        _compute_and_save_batch_stats(analysis_folder, grid_df_list)
        if combined:
            _save_pop_snapshots(analysis_folder, _t0)
            _save_pop_timeseries(analysis_folder, _t0)

        # Drop the per-run report (run_info.txt + run.log) next to the analysis
        # data.
        _lp = getattr(self, "_load_params", None) or {}
        _seed = (getattr(self, "_simulate_params", None) or {}).get("seed")
        model.utils.write_run_report(
            analysis_folder,
            "combined_analysis run" if combined else "analysis run",
            params=({**analyse_params, "seed": _seed}
                    if _seed is not None and "seed" not in analyse_params
                    else analyse_params),
            grid_params=grid_params,
            extra_lines=([f"load {k} = {v}" for k, v in sorted(_lp.items(), key=str)]
                         if combined else None))

        return analysis_folder

    # ==========================================================================
    # PHASE 2b: BULK RE-ANALYSIS FROM DISK
    # ==========================================================================

    @staticmethod
    def _read_run_users(run_folder, want_grid=False):
        """
        Read every user_{i}_history/venue(/grid).parquet from a single run
        folder into the dict form accepted by run_analysis' Pass 1.
        """
        users = []
        i = 0
        while True:
            h = os.path.join(run_folder, f"user_{i}_history.parquet")
            if not os.path.exists(h):
                break
            uh = pd.read_parquet(h)
            v  = os.path.join(run_folder, f"user_{i}_venue.parquet")
            vh = pd.read_parquet(v) if os.path.exists(v) else None
            entry = {"user_history": uh, "venue_history": vh}
            if want_grid:
                g = os.path.join(run_folder, f"user_{i}_grid.parquet")
                if os.path.exists(g):
                    entry["grid"] = pd.read_parquet(g)
            users.append(entry)
            i += 1
        return users

    def analyse_all(self, grid_params=None, metrics=None):
        """
        Re-analyse EVERY run folder under raw_data/, ignoring parameters and any
        previous analysis (the old `_analyzed.lock` mechanism is gone).

        metrics
        -------
        None  -> FULL re-analysis: regenerate every per-user artifact (the
                 expensive per-user pass, incl. real-entropy) and every batch
                 metric.  Use for brand-new raw data or when a per-user value
                 changed.
        list  -> SELECTIVE re-analysis: (re)compute ONLY these batch metrics and
                 MERGE them into each existing batch_stats.parquet, WITHOUT the
                 per-user pass.  Valid because every listed name is a batch
                 metric whose inputs already sit on disk.  A name present is
                 recomputed/overwritten; a name absent is left untouched.

        For each raw run folder the matching parameter-folder name is mirrored
        under analysis_data/ and the per-user artifacts (now including
        rg_explorative) plus batch_stats are regenerated.  Idempotent: every
        analysis_data parameter folder it touches is wiped and rebuilt from the
        current raw runs, so each launch yields exactly one fresh analysis batch
        per raw batch.  raw_data/ is never modified.

        grid_params is only a fallback; the grid (cl/cn) and algorithm are read
        directly from each parameter-folder name when present.
        """
        import re, shutil
        project_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_root      = os.path.join(project_root, "raw_data")
        analysis_root = os.path.join(project_root, "analysis_data")

        # Selective (cheap) path: only (re)compute the listed batch metrics on
        # already-analysed data, merging into each batch_stats.parquet. No raw
        # reload, no per-user recompute.
        if metrics is not None:
            model.utils.reanalyze_batch_metrics(
                project_root, metrics, do_analysis=True, do_combined=False)
            return

        if not os.path.isdir(raw_root):
            logger.warning(f"analyse_all: no raw_data folder at {raw_root}")
            return

        param_folders = sorted(d for d in os.listdir(raw_root)
                               if os.path.isdir(os.path.join(raw_root, d)))
        logger.debug(f"analyse_all: found {len(param_folders)} parameter folder(s) under raw_data/")

        for pf in param_folders:
            raw_param_path = os.path.join(raw_root, pf)
            run_ids = sorted(d for d in os.listdir(raw_param_path)
                             if os.path.isdir(os.path.join(raw_param_path, d)))
            if not run_ids:
                continue

            m_alg = re.search(r"_alg_([A-Za-z]+)$", pf)
            alg   = m_alg.group(1) if m_alg else "ce"
            m_g   = re.search(r"_cl_(\d+)_cn_(\d+)", pf)
            gp    = ({"cell_lenght": int(m_g.group(1)), "cell_number": int(m_g.group(2))}
                     if m_g else (grid_params if alg != "ce" else None))
            analyse_params = {"algorithm": alg}

            out_param_path = os.path.join(analysis_root, pf)
            if os.path.isdir(out_param_path):
                shutil.rmtree(out_param_path)

            for rid in run_ids:
                users = self._read_run_users(os.path.join(raw_param_path, rid),
                                             want_grid=(alg != "ce"))
                if not users:
                    continue
                self.users_list = []
                self.run_analysis(
                    analyse_params, grid_params=gp, combined=False,
                    explicit_out=os.path.join(out_param_path, rid),
                    users_override=users,
                )
                logger.debug(f"analyse_all: {pf}/{rid} -> analysis_data ({len(users)} users)")
        logger.debug("analyse_all: done.")

    def analyse_all_combined(self, grid_params=None, metrics=None):
        """
        Re-analyse EVERY run folder under combined_raw_data/ (the data the `mp`
        plots consume), ignoring parameters and any previous analysis, without
        re-running the resume/recommender phase.

        For each combined run folder the matching parameter-folder name is
        mirrored under combined_analysis_data/ and run_analysis(combined=True)
        regenerates the analysis.
        The loaded/resumed clock boundary (t0) and the resume algorithm (alg1)
        are parsed from the parameter-folder name.  Idempotent: each touched
        combined_analysis_data parameter folder is wiped and rebuilt.
        combined_raw_data/ is never modified.
        """
        import re, shutil
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        craw_root    = os.path.join(project_root, "combined_raw_data")
        cana_root    = os.path.join(project_root, "combined_analysis_data")

        # Selective (cheap) path: merge the listed batch metrics into the
        # existing combined batch_stats.parquet, no per-user pass.
        if metrics is not None:
            model.utils.reanalyze_batch_metrics(
                project_root, metrics, do_analysis=False, do_combined=True)
            return

        if not os.path.isdir(craw_root):
            logger.warning(f"analyse_all_combined: no combined_raw_data folder at {craw_root}")
            return

        param_folders = sorted(d for d in os.listdir(craw_root)
                               if os.path.isdir(os.path.join(craw_root, d)))
        logger.debug(f"analyse_all_combined: found {len(param_folders)} parameter folder(s)")

        for pf in param_folders:
            craw_param = os.path.join(craw_root, pf)
            batch_ids = sorted(d for d in os.listdir(craw_param)
                               if os.path.isdir(os.path.join(craw_param, d)))
            if not batch_ids:
                continue

            m_t  = re.search(r"_t0_(\d+)_t1_(\d+)", pf)
            t0   = int(m_t.group(1)) if m_t else 0
            m_a  = re.search(r"_alg0_([A-Za-z]+)_alg1_([A-Za-z]+)$", pf)
            alg1 = m_a.group(2) if m_a else "ce"
            m_g  = re.search(r"_cl_(\d+)_cn_(\d+)", pf)
            gp   = ({"cell_lenght": int(m_g.group(1)), "cell_number": int(m_g.group(2))}
                    if m_g else (grid_params if alg1 != "ce" else None))
            analyse_params = {"algorithm": alg1, "time": t0}

            out_param = os.path.join(cana_root, pf)
            if os.path.isdir(out_param):
                shutil.rmtree(out_param)

            for bid in batch_ids:
                users = self._read_run_users(os.path.join(craw_param, bid),
                                             want_grid=(alg1 != "ce"))
                if not users:
                    continue
                self.users_list = []
                self.run_analysis(
                    analyse_params, grid_params=gp, combined=True,
                    explicit_out=os.path.join(out_param, bid),
                    explicit_t0=t0, users_override=users,
                )
                logger.debug(f"analyse_all_combined: {pf}/{bid} -> combined_analysis_data ({len(users)} users)")
        logger.debug("analyse_all_combined: done.")

    # ==========================================================================
    # PHASE 3: PLOT
    # ==========================================================================

    def load_analysis_data(self, plot_params, grid_params=None, plot_list=None,
                           reduce_venue="auto"):
        """
        Load pre-computed analysis data from analysis_data/ into memory.

        Parameters
        ----------
        plot_params : dict
            Must contain: "ro", "gamma", "a_param", "b_param", "time".
            Optional: "user_number" (positive = first N, negative = last N, absent = all),
            "algorithm" (str), "starting_position" (str).
        grid_params : dict, optional
            When provided (and algorithm implies a grid), must contain
            "cell_lenght" and "cell_number". Merged into the filter dict passed
            to the loader so the correct analysis folder is located.
        plot_list : list of str, optional
            The same plot_list that will be passed to plot(). When provided,
            only the files required for those plots are loaded from disk,
            and aggregate grids are loaded only when needed.
            When None (default), all files are loaded (backward-compatible).
        """
        # ------------------------------------------------------------------
        # Determine which per-user files and which aggregate data are needed
        # ------------------------------------------------------------------
        _PLOT_FILE_NEEDS = {
            "mobility":           {"user_history", "venue_history", "stats"},
            "rg_distribution":    {"stats"},
            "rg_e_dist":          {"stats"},
            "dist_distribution":  {"user_history"},
            "freq_distribution":  {"venue_history"},
            "two_radius":         {"stats"},
            "num_venues_distribution": {"stats"},
            "s_over_time":        {"user_history"},
            "rg_over_time":       {"user_history"},
            "aggregate_grid":     set(),
            "batch_grid":         set(),
            "aggregate_grid_exploration": set(),
            "batch_grid_exploration":     set(),
            "batch_stats":        set(),
            "rank_persistence":   set(),   # uses the popularity_snapshots artifact
            # Time-resolved venue dynamics: all read their own per-batch artifact
            # (popularity_timeseries / popularity_snapshots) at plot time, so no
            # per-user files are needed here.
            "venue_velocity_fixed_top":   set(),
            "venue_velocity_fixed_worst": set(),
            "venue_velocity_fixed_final": set(),
            "top10_turnover":          set(),
            "top10_rank_tracking":     set(),
            "top10_membership_panels": set(),
            "rank_origin":             set(),
            "grid_final_comparison":   set(),
            "kendall_over_time":       set(),
            "rg_by_ring":              set(),   # reads the histories itself (cached)
            "aggregate_grid_snapshots": set(),
            "srand":              {"stats"},
            "sunc":               {"stats"},
            "sreal":              {"stats"},
            "s_triple":           {"stats"},
            "pi_rand":            {"stats"},
            "pi_unc":             {"stats"},
            "pi_real":            {"stats"},
            "pi_triple":          {"stats"},
            "2rad_distrib":       {"stats"},
            "2rad_distrib_zoomed": {"stats"},
            "venue_visits":                 {"venue_history"},
            "venue_popularity":             {"venue_history"},
            "venue_visits_distribution":    {"venue_history"},
            "venue_explorers_distribution": {"venue_history"},
        }

        # Columns actually consumed from user_history per plot.  Reading only
        # these (Solution B: parquet column projection) avoids decoding the whole
        # wide trajectory when a plot needs a single column.  "mobility" needs
        # the raw coordinates, so it disables projection (full read).
        _PLOT_UHIST_COLS = {
            "dist_distribution":  {"Distance"},
            "s_over_time":        {"Clock", "s"},
            "rg_over_time":       {"Clock", "rg"},
            "mobility":           None,   # sentinel -> read every column
        }

        if plot_list is not None:
            needed_files = set()
            for p in plot_list:
                needed_files |= _PLOT_FILE_NEEDS.get(p, set())
            needs_user_data   = bool(needed_files)
            needs_grids       = any(p in ("aggregate_grid", "batch_grid") for p in plot_list)
            needs_grids_exploration = any(
                p in ("aggregate_grid_exploration", "batch_grid_exploration") for p in plot_list)
            needs_batch_stats = ("batch_stats" in plot_list
                                 or "metric_vs_k" in plot_list
                                 or "metric_summary" in plot_list)
        else:
            needed_files      = None   # load everything
            needs_user_data   = True
            needs_grids       = True
            needs_grids_exploration = True
            needs_batch_stats = True

        # ------------------------------------------------------------------
        # Column projection for user_history (Solution B).  Union the columns
        # the requested plots need; if any requested plot needs the full frame
        # (mobility) or plot_list is None, fall back to reading every column.
        # ------------------------------------------------------------------
        if plot_list is None:
            uhist_cols = None
        else:
            uhist_cols = set()
            full_read  = False
            for p in plot_list:
                spec = _PLOT_UHIST_COLS.get(p, set())
                if spec is None:            # mobility -> needs all columns
                    full_read = True
                    break
                uhist_cols |= spec
            uhist_cols = None if (full_read or not uhist_cols) else sorted(uhist_cols)
        # venue/stats are small -> always read in full (None projection).
        columns_map = {"user_history": uhist_cols} if uhist_cols is not None else None

        # ------------------------------------------------------------------
        # Whether to pre-reduce venue_history to per-sim per-venue arrays
        # (Solution A for venue plots).  freq_distribution needs the per-user
        # venue frames, so it forces the DataFrames to be kept.  "auto" reduces
        # whenever freq is not requested; mp passes reduce_venue=True because
        # multipleplot never draws freq.
        _venue_plots = ("venue_visits", "venue_popularity",
                        "venue_visits_distribution", "venue_explorers_distribution")
        _wants_venue_plots = (plot_list is None) or any(p in plot_list for p in _venue_plots)
        _wants_freq        = (plot_list is None) or ("freq_distribution" in plot_list)
        if reduce_venue is True:
            do_reduce_venue = _wants_venue_plots
        elif reduce_venue is False:
            do_reduce_venue = False
        else:  # "auto"
            do_reduce_venue = _wants_venue_plots and not _wants_freq
        self._venue_reduced = do_reduce_venue

        logger.debug("Loading analysis data...")
        if plot_list is not None:
            logger.debug(f"Plots requested: {plot_list}")
            if needs_user_data:
                logger.debug(f"Per-user files needed: {sorted(needed_files)}")
            else:
                logger.debug("No per-user files required — skipping per-user load")
            if needs_grids:
                logger.debug("Aggregate grids will be loaded")
            else:
                logger.debug("No grid plots requested — skipping aggregate grid load")

        combined = plot_params.get("combined", False)

        # Derive whether a grid was used from the algorithm field.
        # For continuations the key is "algorithm_0" (the base run);
        # for normal params (p mode) the key is "algorithm".
        if combined:
            alg0        = plot_params.get("algorithm_0", "ce")
            grid_active = alg0 != "ce"
        else:
            alg         = plot_params.get("algorithm", "ce")
            grid_active = alg != "ce"

        filters = {**plot_params}
        if grid_active and grid_params:
            filters.update(grid_params)


        # gamma needed by freq_distribution plot
        self.gamma = plot_params.get("gamma")

        self.rg_list = []
        self.two_rg_list = []
        self.rg_e_list = []
        self.venue_history_list = []
        # Distances are histogrammed at load time onto these fixed, shared log
        # edges (Solution A): only per-sim count vectors are kept, never the raw
        # per-jump values.  Bounds come from the power-law sampling support so
        # every jump falls inside; values are clipped to the edges as a safety
        # net so nothing is silently dropped.
        _smin = float(plot_params.get("spatial_min") or _PL["spatial_min"])
        _smax = float(plot_params.get("spatial_max") or _PL["spatial_max"])
        _smin = max(_smin, 1e-9)
        _DIST_BPD = 8   # bins per decade — matches plot_dist_distribution
        _n_dist_bins = max(1, int(np.ceil((np.log10(_smax) - np.log10(_smin)) * _DIST_BPD)))
        self.dist_edges = np.logspace(np.log10(_smin), np.log10(_smax), _n_dist_bins + 1)
        self.dist_hist_per_sim = []          # list over sims of int count vectors
        # Pre-reduced venue arrays (per sim): total visits / distinct users per
        # venue.  Populated only when do_reduce_venue; otherwise stay empty and
        # the per-user venue_history frames are kept instead (freq path).
        self.venue_visits_per_sim = []
        self.venue_users_per_sim  = []
        self._venue_reduce_acc    = []       # list over sims of {vid: [visits, users]}
        self.all_rg_over_time = []
        self.all_s_over_time = []
        self.mean_dist_list = []
        self.s_rand_list = []
        self.s_unc_list  = []
        self.s_real_list = []
        self.pi_rand_list = []
        self.pi_unc_list  = []
        self.pi_real_list = []
        self.two_rg_ratio_list = []
        self.num_venues_list = []   # per-user S = distinct venues explored

        # ------------------------------------------------------------------
        # Per-simulation (per-batch) groupings.
        #
        # Each entry is a list over simulations; element i holds the data of
        # simulation i only.  These are what the band/error-bar machinery in
        # utils_plot consumes: the unit of independent replication is the
        # simulation, NOT the individual user (users within one simulation are
        # correlated), so uncertainty must be estimated ACROSS simulations.
        #
        # The flat lists above are kept for the central estimate and for full
        # backward compatibility; the *_per_sim lists below carry the structure
        # needed to draw bands.
        # ------------------------------------------------------------------
        self.rg_list_per_sim          = []
        self.two_rg_list_per_sim      = []
        self.rg_e_list_per_sim        = []
        self.venue_history_per_sim    = []
        self.all_rg_over_time_per_sim = []
        self.all_s_over_time_per_sim  = []
        self.mean_dist_per_sim        = []
        self.s_rand_per_sim           = []
        self.s_unc_per_sim            = []
        self.s_real_per_sim           = []
        self.pi_rand_per_sim          = []
        self.pi_unc_per_sim           = []
        self.pi_real_per_sim          = []
        self.two_rg_ratio_per_sim     = []
        self.num_venues_per_sim       = []

        # ------------------------------------------------------------------
        # Per-user data load (skipped when only grid plots are requested)
        #
        # return_run_folders=True tags every loaded user with the batch folder
        # it came from, so we can recover the simulation boundaries below.
        # ------------------------------------------------------------------
        loaded = []
        if needs_user_data:
            # Activate parquet column projection for the duration of the load,
            # then restore the previous setting (mirrors the LOAD_THREADS global).
            _prev_cols = model.utils._READ_COLUMNS
            model.utils._READ_COLUMNS = columns_map
            if columns_map:
                logger.debug(f"Column projection active: {columns_map}")
            try:
                if combined:
                    loaded = model.utils.load_combined_analysis_parquet(
                        filters=filters, needed_files=needed_files, return_run_folders=True)
                else:
                    loaded = model.utils.load_analysis_parquet(
                        filters=filters, needed_files=needed_files, return_run_folders=True)
            finally:
                model.utils._READ_COLUMNS = _prev_cols

        # Keep a copy of the full filters with the actual user count for titles/file names.
        # _plot_grid_params excludes user_number (irrelevant for grid/aggregate plots).
        self._plot_params = {**filters, "user_number": len(loaded)}
        self._plot_grid_params = {k: v for k, v in filters.items() if k != "user_number"}
        self._grid_active = grid_active  # saved for use in plot()

        # Map each batch folder to the index of its per-sim bucket, in order of
        # first appearance, so the per-sim lists stay aligned and deterministic.
        sim_index_of = {}

        def _sim_bucket(run_folder):
            """Return (and lazily create) the per-sim bucket index for a batch."""
            if run_folder not in sim_index_of:
                sim_index_of[run_folder] = len(sim_index_of)
                self.rg_list_per_sim.append([])
                self.two_rg_list_per_sim.append([])
                self.rg_e_list_per_sim.append([])
                self.venue_history_per_sim.append([])
                self.all_rg_over_time_per_sim.append([])
                self.all_s_over_time_per_sim.append([])
                self.mean_dist_per_sim.append([])
                self.dist_hist_per_sim.append(None)      # int count vector, lazily created
                self._venue_reduce_acc.append({})        # {vid: [visits, users]}
                self.s_rand_per_sim.append([])
                self.s_unc_per_sim.append([])
                self.s_real_per_sim.append([])
                self.pi_rand_per_sim.append([])
                self.pi_unc_per_sim.append([])
                self.pi_real_per_sim.append([])
                self.two_rg_ratio_per_sim.append([])
                self.num_venues_per_sim.append([])
            return sim_index_of[run_folder]

        for wrapped in loaded:
            # return_run_folders=True wraps each user as {"data":..., "run_folder":...}.
            # Fall back gracefully if a plain entry ever slips through.
            if isinstance(wrapped, dict) and "data" in wrapped and "run_folder" in wrapped:
                entry      = wrapped["data"]
                run_folder = wrapped["run_folder"]
            else:
                entry      = wrapped
                run_folder = "__single__"   # one anonymous simulation

            si = _sim_bucket(run_folder)

            stats_df         = entry.get("stats")
            venue_history_df = entry.get("venue_history")
            user_history_df  = entry.get("user_history")

            rg_over_time_df = None
            s_over_time_df  = None
            if user_history_df is not None:
                if "rg" in user_history_df.columns:
                    rg_over_time_df = user_history_df[["Clock", "rg"]].rename(
                        columns={"Clock": "clock"})
                if "s" in user_history_df.columns:
                    s_rows = user_history_df[["Clock", "s"]].dropna(subset=["s"])
                    s_over_time_df = s_rows.rename(columns={"Clock": "clock"})

            self.last_user_history  = user_history_df
            self.last_venue_history = venue_history_df
            self.last_stats         = stats_df

            if stats_df is not None:
                # Build an O(1) metric->value map once per user instead of
                # scanning the DataFrame with .loc for every single metric
                # (the old _sget did a full linear filter per metric per user,
                # which dominated the post-load processing time).
                stats_map = dict(zip(stats_df['metric'], stats_df['value']))
                def _sget(m):
                    v = stats_map.get(m, float("nan"))
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return float("nan")

                rg_val  = _sget("radius of gyration")
                two_val = _sget("radius of gyration two")
                rg_e_val = _sget("rg_explorative")
                sr_val  = _sget("s_rand")
                su_val  = _sget("s_unc")
                sre_val = _sget("s_real")
                pir_val  = _sget("pi_rand")
                piu_val  = _sget("pi_unc")
                pire_val = _sget("pi_real")
                nv_val   = _sget("num_venues")   # S = distinct venues explored

                # Derived on the fly from metrics already in the parquet, so
                # no re-analysis is needed for the 2rad_distrib plot alone.
                ratio_val = (two_val / rg_val
                             if (np.isfinite(rg_val) and rg_val > 0 and np.isfinite(two_val))
                             else float("nan"))

                self.rg_list.append(rg_val)
                self.two_rg_list.append(two_val)
                self.rg_e_list.append(rg_e_val)

                self.s_rand_list.append(sr_val)
                self.s_unc_list.append(su_val)
                self.s_real_list.append(sre_val)
                self.pi_rand_list.append(pir_val)
                self.pi_unc_list.append(piu_val)
                self.pi_real_list.append(pire_val)
                self.two_rg_ratio_list.append(ratio_val)
                self.num_venues_list.append(nv_val)

                self.rg_list_per_sim[si].append(rg_val)
                self.two_rg_list_per_sim[si].append(two_val)
                self.rg_e_list_per_sim[si].append(rg_e_val)
                self.s_rand_per_sim[si].append(sr_val)
                self.s_unc_per_sim[si].append(su_val)
                self.s_real_per_sim[si].append(sre_val)
                self.pi_rand_per_sim[si].append(pir_val)
                self.pi_unc_per_sim[si].append(piu_val)
                self.pi_real_per_sim[si].append(pire_val)
                self.two_rg_ratio_per_sim[si].append(ratio_val)
                self.num_venues_per_sim[si].append(nv_val)

            # -- distances: histogram on shared fixed edges at load time --
            # (Solution A) instead of keeping every raw delta_r in a Python list.
            if user_history_df is not None and "Distance" in user_history_df.columns:
                d = user_history_df["Distance"].to_numpy()
                d = d[np.isfinite(d) & (d > 0)]
                if d.size:
                    # Clip into the fixed support so out-of-range jumps land in the
                    # extreme bins rather than being dropped by np.histogram.
                    np.clip(d, self.dist_edges[0], self.dist_edges[-1], out=d)
                    counts, _ = np.histogram(d, bins=self.dist_edges)
                    if self.dist_hist_per_sim[si] is None:
                        self.dist_hist_per_sim[si] = counts.astype(np.int64)
                    else:
                        self.dist_hist_per_sim[si] += counts

            # -- venue history: either reduce to per-venue arrays now (Solution A
            #    for venue plots) or keep the per-user frames (freq path) --
            if venue_history_df is not None:
                if do_reduce_venue:
                    acc = self._venue_reduce_acc[si]
                    id_col = venue_history_df.columns[0]
                    vids = venue_history_df[id_col].to_numpy()
                    nvis = venue_history_df["Number of visits"].to_numpy()
                    seen = set()   # dedupe distinct-user credit within this user
                    for vid, nv in zip(vids, nvis):
                        slot = acc.get(vid)
                        if slot is None:
                            slot = [0.0, 0]
                            acc[vid] = slot
                        slot[0] += float(nv)          # total visits
                        if vid not in seen:
                            slot[1] += 1              # one distinct user
                            seen.add(vid)
                else:
                    self.venue_history_list.append(venue_history_df)
                    self.venue_history_per_sim[si].append(venue_history_df)
            if rg_over_time_df is not None:
                self.all_rg_over_time.append(rg_over_time_df)
                self.all_rg_over_time_per_sim[si].append(rg_over_time_df)
            if s_over_time_df is not None:
                self.all_s_over_time.append(s_over_time_df)
                self.all_s_over_time_per_sim[si].append(s_over_time_df)

        # Finalize the per-sim venue reduction: turn each {vid: [visits, users]}
        # accumulator into two flat arrays (one entry per venue in that sim).
        if do_reduce_venue:
            for acc in self._venue_reduce_acc:
                if acc:
                    vals = np.fromiter((s[0] for s in acc.values()),
                                       dtype=float, count=len(acc))
                    usrs = np.fromiter((s[1] for s in acc.values()),
                                       dtype=float, count=len(acc))
                else:
                    vals = np.array([], dtype=float)
                    usrs = np.array([], dtype=float)
                self.venue_visits_per_sim.append(vals)
                self.venue_users_per_sim.append(usrs)
        # The accumulator dicts and the raw loaded frames are no longer needed.
        self._venue_reduce_acc = []
        try:
            del loaded
        except NameError:
            pass

        n_sims = len(sim_index_of)
        self._n_sims = n_sims
        if n_sims <= 1:
            logger.debug(
                "load_analysis_data: only %d simulation(s) detected — no error "
                "bands will be drawn (bands need >=2 batches). If you expected "
                "several batches, make sure user_number is left empty/None so "
                "every batch is loaded instead of just the first N users.", n_sims
            )
        else:
            logger.debug("load_analysis_data: %d simulations detected — "
                        "error bands will be drawn across simulations.", n_sims)

        # ------------------------------------------------------------------
        # Aggregate grid load (skipped when no grid plots are requested)
        # ------------------------------------------------------------------
        if needs_grids:
            if combined:
                self.aggregate_grid, self.aggregate_grid_user_count, self.batch_grids = \
                    model.utils.load_combined_all_aggregate_grids(filters)
            else:
                self.aggregate_grid, self.aggregate_grid_user_count, self.batch_grids = \
                    model.utils.load_all_aggregate_grids(filters)
        else:
            self.aggregate_grid            = None
            self.aggregate_grid_user_count = 0
            self.batch_grids               = []

        # ------------------------------------------------------------------
        # Exploration aggregate grid load (distinct users per cell).
        # Skipped unless an exploration grid plot is requested.
        # ------------------------------------------------------------------
        if needs_grids_exploration:
            if combined:
                (self.aggregate_grid_exploration,
                 self.aggregate_grid_exploration_user_count,
                 self.batch_grids_exploration) = \
                    model.utils.load_combined_all_aggregate_grids_exploration(filters)
            else:
                (self.aggregate_grid_exploration,
                 self.aggregate_grid_exploration_user_count,
                 self.batch_grids_exploration) = \
                    model.utils.load_all_aggregate_grids_exploration(filters)
        else:
            self.aggregate_grid_exploration            = None
            self.aggregate_grid_exploration_user_count = 0
            self.batch_grids_exploration               = []

        # ------------------------------------------------------------------
        # Batch stats load (skipped when batch_stats is not in plot_list)
        # ------------------------------------------------------------------
        if needs_batch_stats:
            if combined:
                self.batch_stats_list = model.utils.load_combined_all_batch_stats(filters)
            else:
                self.batch_stats_list = model.utils.load_all_batch_stats(filters)
        else:
            self.batch_stats_list = []

    def plot(self, plot_list, show=False, parallel=True, max_workers=None):
        """
        Generate all plots listed in plot_list.

        Must be called after load_analysis_data().

        Parameters
        ----------
        plot_list : list of str
            Accepted values: "mobility", "rg_distribution", "freq_distribution",
            "two_radius", "2rad_distrib", "s_over_time", "rg_over_time",
            "srand", "sunc", "sreal", "s_triple",
            "pi_rand", "pi_unc", "pi_real", "pi_triple".
        """
        p  = {**self._plot_params}
        pg = {**self._plot_grid_params}

        title = model.utils_plot.build_plot_title
        up = model.utils_plot

        # Single-series per-sim payloads: one (label, per_sim_data) pair with an
        # empty label (so the plot stays single-colour with no legend, but a band
        # is drawn when >=2 simulations are present).
        def one(per_sim):
            return [("", per_sim)]

        # ------------------------------------------------------------------
        # Build the task list.  Each task is (name, callable, kwargs) and is
        # fully self-contained so it can run in a separate process.
        # ------------------------------------------------------------------
        tasks = []
        def add(name, func, **kwargs):
            tasks.append((name, func, kwargs))

        if "mobility" in plot_list:
            add("mobility", up.plot_mobility_visualization,
                user_history=self.last_user_history, venue_history=self.last_venue_history,
                stats=self.last_stats, filters=p, title=title("User Mobility", p), show=show)

        if "rg_distribution" in plot_list:
            add("rg_distribution", up.plot_rg_distribution,
                rg_list=self.rg_list, per_sim_series=one(self.rg_list_per_sim),
                filters=p, title=title("Radius of gyration", p), show=show)
            add("rg_rank", up.plot_rg_rank,
                per_sim_series=one(self.rg_list_per_sim),
                filters=p, title=title("Radius of gyration, rank plot", p), show=show)

        if "rg_e_dist" in plot_list:
            add("rg_e_dist", up.plot_rg_e_distribution,
                rg_list=self.rg_e_list, per_sim_series=one(self.rg_e_list_per_sim),
                filters=p, title=title("Exploration radius of gyration", p), show=show)

        if "dist_distribution" in plot_list:
            add("dist_distribution", up.plot_dist_distribution,
                mean_dist_list=None,
                per_sim_hist=(self.dist_edges, one(self.dist_hist_per_sim)),
                filters=p, title=title("Travel distance", p), show=show)

        if "freq_distribution" in plot_list:
            add("freq_distribution", up.plot_frequency_distribution,
                venue_history_list=self.venue_history_list, gamma=self.gamma,
                per_sim_list=self.venue_history_per_sim,
                filters=p, title=title("Frequency Distribution", p), show=show)

        if "two_radius" in plot_list:
            # Scatter of the joint (rg, 2rg) cloud: pooling points does not bias
            # the cloud, so this stays a pooled scatter (no band).
            add("two_radius", up.plot_two_radius_vs_radius,
                rg_list=self.rg_list, two_rg_list=self.two_rg_list,
                filters=p, title=title("Two-RG vs RG", p), show=show)

        if "s_over_time" in plot_list:
            add("s_over_time", up.plot_S_over_time_from_df,
                s_over_time_list=self.all_s_over_time, per_sim_series=one(self.all_s_over_time_per_sim),
                bins_per_decade=10, filters=p, title=title("S over time", p), show=show)

        if "rg_over_time" in plot_list:
            add("rg_over_time", up.plot_r_gyr_over_time_from_df,
                rg_over_time_list=self.all_rg_over_time, per_sim_series=one(self.all_rg_over_time_per_sim),
                bins_per_decade=10, filters=p, title=title("Radius of Gyration over time", p), show=show)

        if "aggregate_grid" in plot_list:
            if self.aggregate_grid is not None:
                pg_with_n = {**pg, "u_n": self.aggregate_grid_user_count}
                add("aggregate_grid", up.plot_aggregate_grid,
                    grid_matrix=self.aggregate_grid, filters=pg_with_n,
                    title=title("Aggregate Grid", pg_with_n),
                    mask_center=self._grid_active and p.get("starting_position", "core_periphery") == "centered",
                    show=show)
            else:
                logger.warning("No param-level aggregate grid available to plot.")

        if "batch_grid" in plot_list:
            if self.batch_grids:
                for batch_id, grid, batch_user_count in self.batch_grids:
                    pg_with_n = {**pg, "u_n": batch_user_count}
                    add(f"batch_grid[{batch_id}]", up.plot_aggregate_grid,
                        grid_matrix=grid, filters=pg_with_n,
                        title=title(f"Grid - {batch_id}", pg_with_n), batch_id=batch_id,
                        mask_center=self._grid_active and p.get("starting_position", "core_periphery") == "centered",
                        show=show)
            else:
                logger.warning("No batch-level aggregate grids available to plot.")

        if "aggregate_grid_exploration" in plot_list:
            if self.aggregate_grid_exploration is not None:
                pg_with_n = {**pg, "u_n": self.aggregate_grid_exploration_user_count}
                add("aggregate_grid_exploration", up.plot_aggregate_grid,
                    grid_matrix=self.aggregate_grid_exploration, filters=pg_with_n,
                    title=title("Aggregate Grid - distinct visitors", pg_with_n),
                    mask_center=self._grid_active and p.get("starting_position", "core_periphery") == "centered",
                    cbar_label="Number of distinct visitors",
                    name_override="plot_aggregate_grid_exploration",
                    show=show)
            else:
                logger.warning("No param-level exploration grid available to plot.")

        if "batch_grid_exploration" in plot_list:
            if self.batch_grids_exploration:
                for batch_id, grid, batch_user_count in self.batch_grids_exploration:
                    pg_with_n = {**pg, "u_n": batch_user_count}
                    add(f"batch_grid_exploration[{batch_id}]", up.plot_aggregate_grid,
                        grid_matrix=grid, filters=pg_with_n,
                        title=title(f"Grid distinct visitors - {batch_id}", pg_with_n),
                        batch_id=batch_id,
                        mask_center=self._grid_active and p.get("starting_position", "core_periphery") == "centered",
                        cbar_label="Number of distinct visitors",
                        name_override="plot_aggregate_grid_exploration",
                        show=show)
            else:
                logger.warning("No batch-level exploration grids available to plot.")

        if "num_venues_distribution" in plot_list:
            add("num_venues_distribution", up.plot_num_venues_distribution,
                num_venues_list=self.num_venues_list,
                per_sim_series=one(self.num_venues_per_sim),
                filters=p, title=title("Number of distinct locations", p), show=show)

        if "srand" in plot_list:
            add("srand", up.plot_entropy_distribution,
                entropy_list=self.s_rand_list, entropy_type="srand", per_sim_series=one(self.s_rand_per_sim),
                filters=p, title=title("Random entropy", p), show=show)
        if "sunc" in plot_list:
            add("sunc", up.plot_entropy_distribution,
                entropy_list=self.s_unc_list, entropy_type="sunc", per_sim_series=one(self.s_unc_per_sim),
                filters=p, title=title("Uncorrelated entropy", p), show=show)
        if "sreal" in plot_list:
            add("sreal", up.plot_entropy_distribution,
                entropy_list=self.s_real_list, entropy_type="sreal", per_sim_series=one(self.s_real_per_sim),
                filters=p, title=title("Real entropy", p), show=show)
        if "s_triple" in plot_list:
            add("s_triple", up.plot_entropy_triple,
                s_rand_list=self.s_rand_list, s_unc_list=self.s_unc_list, s_real_list=self.s_real_list,
                per_sim_triple=(self.s_rand_per_sim, self.s_unc_per_sim, self.s_real_per_sim),
                filters=p, title=title("Entropy Distributions", p), show=show)

        if "pi_rand" in plot_list:
            add("pi_rand", up.plot_predictability_distribution,
                pi_list=self.pi_rand_list, pi_type="pi_rand",
                per_sim_series=one(self.pi_rand_per_sim),
                filters=p, title=title("Random predictability", p), show=show)
        if "pi_unc" in plot_list:
            add("pi_unc", up.plot_predictability_distribution,
                pi_list=self.pi_unc_list, pi_type="pi_unc",
                per_sim_series=one(self.pi_unc_per_sim),
                filters=p, title=title("Uncorrelated predictability", p), show=show)
        if "pi_real" in plot_list:
            add("pi_real", up.plot_predictability_distribution,
                pi_list=self.pi_real_list, pi_type="pi_real",
                per_sim_series=one(self.pi_real_per_sim),
                filters=p, title=title("Maximum predictability", p), show=show)
        if "pi_triple" in plot_list:
            add("pi_triple", up.plot_predictability_triple,
                pi_rand_list=self.pi_rand_list, pi_unc_list=self.pi_unc_list,
                pi_real_list=self.pi_real_list,
                per_sim_triple=(self.pi_rand_per_sim, self.pi_unc_per_sim,
                                self.pi_real_per_sim),
                filters=p, title=title("Max Predictability Distributions", p), show=show)

        if "2rad_distrib" in plot_list:
            add("2rad_distrib", up.plot_two_rg_ratio_distribution,
                ratio_list=self.two_rg_ratio_list,
                per_sim_series=one(self.two_rg_ratio_per_sim),
                filters=p, title=title("2-radius ratio", p), show=show)

        if "2rad_distrib_zoomed" in plot_list:
            add("2rad_distrib_zoomed", up.plot_two_rg_ratio_distribution_zoomed,
                ratio_list=self.two_rg_ratio_list,
                per_sim_series=one(self.two_rg_ratio_per_sim),
                filters=p, title=title("2-radius ratio (zoom)", p), show=show)

        # Venue plots: when venue_history was pre-reduced at load, feed the
        # per-sim per-venue arrays (per_sim_values); otherwise fall back to the
        # per-user DataFrames (per_sim_series).
        _vred = getattr(self, "_venue_reduced", False)
        if "venue_visits" in plot_list:
            _kw = ({"per_sim_values": one(self.venue_visits_per_sim)} if _vred
                   else {"per_sim_series": one(self.venue_history_per_sim)})
            add("venue_visits", up.plot_venue_ranking,
                venue_history_list=self.venue_history_list, mode="visits",
                filters=p, title=title("Venue Visits Ranking", p), show=show,
                **_kw)
        if "venue_popularity" in plot_list:
            _kw = ({"per_sim_values": one(self.venue_users_per_sim)} if _vred
                   else {"per_sim_series": one(self.venue_history_per_sim)})
            add("venue_popularity", up.plot_venue_ranking,
                venue_history_list=self.venue_history_list, mode="popularity",
                filters=p, title=title("Venue Popularity Ranking", p), show=show,
                **_kw)
        if "venue_visits_distribution" in plot_list:
            _kw = ({"per_sim_values": one(self.venue_visits_per_sim)} if _vred
                   else {"per_sim_series": one(self.venue_history_per_sim)})
            add("venue_visits_distribution", up.plot_venue_count_distribution,
                venue_history_list=self.venue_history_list, mode="visits",
                filters=p, title=title("Visit popularity", p), show=show, **_kw)
        if "venue_explorers_distribution" in plot_list:
            _kw = ({"per_sim_values": one(self.venue_users_per_sim)} if _vred
                   else {"per_sim_series": one(self.venue_history_per_sim)})
            add("venue_explorers_distribution", up.plot_venue_count_distribution,
                venue_history_list=self.venue_history_list, mode="explorers",
                filters=p, title=title("Exploration popularity", p), show=show, **_kw)

        # Dispatch all figure tasks (serial when show=True, else across processes).
        _run_plot_tasks(tasks, parallel=parallel, max_workers=max_workers, show=show)

        # batch_stats only prints a text table to stdout -> keep it on the main
        # process (no figure, nothing to parallelise).
        if "batch_stats" in plot_list:
            if self.batch_stats_list:
                up.print_batch_stats(self.batch_stats_list, filters=pg,
                                     title=title("Batch Stats", pg))
            else:
                logger.warning("No batch stats available to print.")

        if "rank_persistence" in plot_list:
            snap_list = (model.utils.load_combined_popularity_snapshots(self._plot_params)
                         if self._plot_params.get("combined") else [])
            if snap_list:
                lbl = str(self._plot_params.get("algorithm_1", ""))
                pparams = {**self._plot_params}
                rec = [(lbl, _snapshot_pairs(snap_list, "at_t0", "final"))]
                cold = [(lbl + " cold-start", _snapshot_pairs(snap_list, "init", "at_t0"))]
                up.plot_retention(rec, transition="recommender", filters=pparams, show=show)
                up.plot_retention(cold, transition="coldstart", filters=pparams, show=show)
            else:
                logger.warning("No popularity snapshots available for rank_persistence.")

        # ------------------------------------------------------------------
        # Time-resolved venue dynamics (single dataset): see
        # _draw_time_resolved for the list and the per-run logic.
        # ------------------------------------------------------------------
        if any(pl in plot_list for pl in TIME_RESOLVED_PLOTS):
            if self._plot_params.get("combined"):
                _draw_time_resolved(
                    plot_list,
                    [(str(self._plot_params.get("algorithm_1", "")), self._plot_params)],
                    filters={**self._plot_params},
                    show=show, single_dataset=True,
                    mask_center=self._grid_active and
                    self._plot_params.get("starting_position", "core_periphery") == "centered")
            else:
                logger.warning("Time-resolved venue plots need continuations (t0, t1): "
                               "skipped for baselines.")

        # r_g and centre of mass by ring (reads the histories, cached)
        if "rg_by_ring" in plot_list:
            if self._plot_params.get("combined"):
                up.plot_rg_by_ring(
                    [(str(self._plot_params.get("algorithm_1", "")), self._plot_params)],
                    filters={**self._plot_params}, show=show)
            else:
                logger.warning("rg_by_ring needs continuations (t0, t1): skipped for baselines.")

    # ==========================================================================
    # MULTIPLEPLOT  (several series overlaid)
    # ==========================================================================

    @staticmethod
    def _expand_multiple_plot_params(mp_params):
        """
        Expand a multiple_plot_params dict that may contain list/tuple values
        into a list of plain dicts (one per combination) and a matching list
        of label strings.

        Rules
        -----
        * The key ``"label"`` must be a list/tuple of strings — one per
          expected combination.  It is popped before building combinations.
        * The key ``"combined"`` is scalar and is copied verbatim to every
          expansion.
        * If the key ``"series"`` is present, it must be a list of dicts, one
          per series.  The cartesian product is then BYPASSED entirely: each
          dict is merged over the shared scalar keys and zipped with ``label``.
          This is the way to express COUPLED parameters that a cartesian product
          cannot represent.
        * Otherwise, every key whose value is a list/tuple participates in the
          cartesian product; scalar values are shared across all combinations.
          The number of labels must equal the number of combinations produced.

        Returns
        -------
        list of (label_str, params_dict)
        """
        import itertools

        params = dict(mp_params)
        labels = params.pop("label")
        if isinstance(labels, str):
            labels = [labels]

        # ------------------------------------------------------------------
        # Explicit per-series specification: bypass the cartesian product.
        # "series" is a list of dicts (one per series); each is layered on top
        # of the remaining (shared) keys.  Use this for coupled parameters.
        # ------------------------------------------------------------------
        if "series" in params:
            series_list = params.pop("series")
            if not isinstance(series_list, (list, tuple)):
                raise ValueError(
                    "multiple_plot_params: 'series' must be a list of dicts.")
            if len(labels) != len(series_list):
                raise ValueError(
                    f"multiple_plot_params: 'label' has {len(labels)} entries "
                    f"but 'series' has {len(series_list)} entries.")
            shared = dict(params)   # every remaining key is shared across series
            result = []
            for label, overrides in zip(labels, series_list):
                if not isinstance(overrides, dict):
                    raise ValueError(
                        "multiple_plot_params: each 'series' entry must be a dict.")
                result.append((label, {**shared, **overrides}))
            return result

        # Separate list/scalar values (skip "combined" — always scalar)
        list_keys   = [k for k, v in params.items()
                       if isinstance(v, (list, tuple)) and k != "combined"]
        scalar_keys = [k for k in params if k not in list_keys]

        if not list_keys:
            # No expansion needed — single combination
            return [(labels[0] if labels else "", params)]

        # Build cartesian product of all list values
        value_combinations = list(itertools.product(*[params[k] for k in list_keys]))
        scalar_part = {k: params[k] for k in scalar_keys}

        if len(labels) != len(value_combinations):
            raise ValueError(
                f"multiple_plot_params: 'label' has {len(labels)} entries but "
                f"the cartesian product of list values yields "
                f"{len(value_combinations)} combinations.\n"
                f"List keys: {list_keys}"
            )

        result = []
        for label, combo in zip(labels, value_combinations):
            combo_dict = {**scalar_part, **dict(zip(list_keys, combo))}
            result.append((label, combo_dict))
        return result

    def load_multiple_plot_data(self, mp_params, grid_params=None, plot_list=None,
                                load_parallel=True, load_workers=None):
        """
        Load one analysis dataset per combination encoded in *mp_params* and
        store each as a named series in self._mp_series for use by
        multipleplot().

        The series are independent, so by default they are loaded in PARALLEL
        across processes (each process has its own GIL, which is what actually
        speeds up the parquet decode — thread-level parallelism cannot, because
        the GIL is held during DataFrame construction).

        Parameters
        ----------
        mp_params : dict
            Like combined_plot_params but with:
            - ``"label"``  : list of strings, one per series
            - List/tuple values on any other key trigger cartesian expansion.
            - ``"series"`` : list of per-series dicts -> bypasses the cartesian
              product (use for coupled params).
            - ``"combined": True`` must be present (mp works on combined data).
        grid_params : dict, optional
            Shared grid settings.  Grid keys given per series (``cell_number`` /
            ``cell_lenght`` inside a ``series`` entry or an expanded combination)
            OVERRIDE this shared dict for that series, so different series can
            load from different-sized grids (e.g. 15 vs 35).
        plot_list : list of str, optional
        load_parallel : bool, optional (default True)
            Load the series concurrently across processes.  Set False on a slow
            spinning disk, where concurrent heavy reads thrash the disk and can
            be slower than serial.  Override via env AIEPR_SERIES_WORKERS=1.
        load_workers : int, optional
            Number of series loaded at once.  Defaults to
            min(n_series, max(1, n_cpu // 4)) so each worker still gets a few
            cores for its own parquet reads.  Env AIEPR_SERIES_WORKERS wins.
        """
        self._mp_series = []
        expansions = self._expand_multiple_plot_params(mp_params)
        n = len(expansions)
        logger.debug(f"multipleplot: loading {n} series")

        # Per-series grid_params: overlay this series' own grid keys onto the
        # shared grid_params so a per-series cell_number/cell_lenght is not
        # clobbered by the shared dict (load_analysis_data lets grid_params win
        # over plot_params).  Without this, all series would load from the same
        # grid regardless of their individual cell_number.
        def _grid_for(p_dict):
            g = dict(grid_params) if grid_params else {}
            for gk in ("cell_number", "cell_lenght"):
                if p_dict.get(gk) is not None:
                    g[gk] = p_dict[gk]
            return g or None

        env_workers = os.environ.get("AIEPR_SERIES_WORKERS")
        if env_workers is not None:
            load_workers = int(env_workers)
        if load_workers is None:
            # Conservative default: each worker transiently holds a full series'
            # user_history in memory, so we cap concurrency to protect RAM.
            # Raise it via load_workers=... or AIEPR_SERIES_WORKERS if you have
            # headroom and a fast disk.
            load_workers = min(n, 4)

        # Cap inner read-threads per worker so total threads ≈ n_cpu.
        inner_threads = max(1, (os.cpu_count() or 4) // max(1, load_workers))

        use_parallel = load_parallel and load_workers > 1 and n > 1

        if not use_parallel:
            for label, p_dict in expansions:
                logger.debug(f"  loading series '{label}' (serial) with params: {p_dict}")
                tmp = Simulation()
                tmp.load_analysis_data(p_dict, grid_params=_grid_for(p_dict), plot_list=plot_list,
                                       reduce_venue=True)
                self._mp_series.append(_series_dict_from_sim(label, tmp))
                _log_series_loaded(self._mp_series[-1])
            return

        logger.debug(f"  loading series across {load_workers} processes "
                    f"({inner_threads} read-threads each)")
        try:
            ctx = __import__("multiprocessing").get_context("fork")
        except ValueError:
            ctx = None

        if ctx is None:
            # Fork unavailable -> serial fallback.
            for label, p_dict in expansions:
                tmp = Simulation()
                tmp.load_analysis_data(p_dict, grid_params=_grid_for(p_dict), plot_list=plot_list,
                                       reduce_venue=True)
                self._mp_series.append(_series_dict_from_sim(label, tmp))
                _log_series_loaded(self._mp_series[-1])
            return

        work = [(label, p_dict, _grid_for(p_dict), plot_list, inner_threads)
                for label, p_dict in expansions]
        results = [None] * n
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=load_workers, mp_context=ctx
        ) as ex:
            futs = {ex.submit(_load_one_series_worker, w): i for i, w in enumerate(work)}
            prog = model.utils.Progress("loading", n, "series")
            for fut in model.utils.completed_with_progress(futs, prog):
                i = futs[fut]
                series = fut.result()        # re-raises worker exceptions here
                results[i] = series
                _log_series_loaded(series)
                prog.update(prog.done + 1)
        self._mp_series = results

    def multipleplot(self, plot_list, show=False, parallel=True, max_workers=None):
        """
        Generate multi-series overlay plots for every entry in plot_list that
        supports it.  When a series carries >=2 simulations, each overlaid curve
        gets its own cross-simulation error band.

        "batch_stats" is printed once per series (each series gets its own table).
        """
        if not self._mp_series:
            logger.warning("multipleplot() called but _mp_series is empty. "
                           "Call load_multiple_plot_data() first.")
            return

        # Baselines have no t0/t1: drop the figures that exist only for
        # continuations (time-resolved dynamics, rank persistence, final grids,
        # r_g by ring, statistics vs a parameter of the recommenders).
        if not all(s["_plot_params"].get("combined") for s in self._mp_series):
            _cont_only = set(TIME_RESOLVED_PLOTS) | {
                "rank_persistence", "grid_final_comparison", "rg_by_ring", "metric_vs_k"}
            _skipped = [pl for pl in plot_list if pl in _cont_only]
            if _skipped:
                logger.warning(f"baselines: figures only for continuations skipped: {_skipped}")
            plot_list = [pl for pl in plot_list if pl not in _cont_only]

        # Reclaim any transient load-time garbage before forking the plot
        # workers: forked children share the parent's pages copy-on-write, but
        # CPython's refcount writes progressively copy them, so a smaller live
        # set at fork time means less duplicated RSS across workers.
        gc.collect()

        # Filters used for the titles: those of the first series.
        first_p = dict(self._mp_series[0]["_plot_params"])

        title = model.utils_plot.build_plot_title
        up = model.utils_plot

        def ps(key):
            """Per-sim series payload across all series for a given attribute."""
            return [(s["label"], s[key]) for s in self._mp_series]

        _UNSUPPORTED = {"mobility", "aggregate_grid", "batch_grid",
                        "aggregate_grid_exploration", "batch_grid_exploration",
                        "aggregate_grid_snapshots",   # physical-grid show, per dataset
                        "s_triple", "pi_triple"}
        for pl in plot_list:
            if pl in _UNSUPPORTED:
                logger.warning(f"multipleplot: '{pl}' is not supported in a multiplot — skipped.")

        tasks = []
        def add(name, func, **kwargs):
            tasks.append((name, func, kwargs))

        if "rg_distribution" in plot_list:
            add("rg_distribution", up.plot_rg_distribution, rg_list=None,
                per_sim_series=ps("rg_list_per_sim"), filters=first_p,
                title=title("Radius of gyration (multi)", first_p), show=show)
            add("rg_rank", up.plot_rg_rank,
                per_sim_series=ps("rg_list_per_sim"), filters=first_p,
                title=title("Radius of gyration, rank plot (multi)", first_p), show=show)

        if "rg_e_dist" in plot_list:
            add("rg_e_dist", up.plot_rg_e_distribution, rg_list=None,
                per_sim_series=ps("rg_e_list_per_sim"), filters=first_p,
                title=title("Exploration radius of gyration (multi)", first_p), show=show)

        if "dist_distribution" in plot_list:
            # All series share the same fixed edges (built from the same
            # spatial_min/max), so any series' edges will do.
            _edges = self._mp_series[0].get("dist_edges")
            _hist_series = [(s["label"], s.get("dist_hist_per_sim", [])) for s in self._mp_series]
            add("dist_distribution", up.plot_dist_distribution, mean_dist_list=None,
                per_sim_hist=(_edges, _hist_series), filters=first_p,
                title=title("Travel distance (multi)", first_p), show=show)

        if "rg_over_time" in plot_list:
            add("rg_over_time", up.plot_r_gyr_over_time_from_df, rg_over_time_list=None,
                per_sim_series=ps("all_rg_over_time_per_sim"), bins_per_decade=10,
                filters=first_p, title=title("RG over time (multi)", first_p), show=show)

        if "s_over_time" in plot_list:
            add("s_over_time", up.plot_S_over_time_from_df, s_over_time_list=None,
                per_sim_series=ps("all_s_over_time_per_sim"), bins_per_decade=10,
                filters=first_p, title=title("S over time (multi)", first_p), show=show)

        if "two_radius" in plot_list:
            # Pooled scatter per series (no band) — unchanged.
            add("two_radius", up.plot_two_radius_vs_radius, rg_list=None, two_rg_list=None,
                series_list=[(s["label"], s["rg_list"], s["two_rg_list"]) for s in self._mp_series],
                filters=first_p, title=title("Two-RG vs RG (multi)", first_p), show=show)

        if "num_venues_distribution" in plot_list:
            add("num_venues_distribution", up.plot_num_venues_distribution, num_venues_list=None,
                per_sim_series=ps("num_venues_per_sim"), filters=first_p,
                title=title("Number of distinct locations (multi)", first_p), show=show)

        if "srand" in plot_list:
            add("srand", up.plot_entropy_distribution, entropy_list=None, entropy_type="srand",
                per_sim_series=ps("s_rand_per_sim"), filters=first_p,
                title=title("Random entropy (multi)", first_p), show=show)
        if "sunc" in plot_list:
            add("sunc", up.plot_entropy_distribution, entropy_list=None, entropy_type="sunc",
                per_sim_series=ps("s_unc_per_sim"), filters=first_p,
                title=title("Uncorrelated entropy (multi)", first_p), show=show)
        if "sreal" in plot_list:
            add("sreal", up.plot_entropy_distribution, entropy_list=None, entropy_type="sreal",
                per_sim_series=ps("s_real_per_sim"), filters=first_p,
                title=title("Real entropy (multi)", first_p), show=show)

        if "pi_rand" in plot_list:
            add("pi_rand", up.plot_predictability_distribution, pi_list=None, pi_type="pi_rand",
                per_sim_series=ps("pi_rand_per_sim"), filters=first_p,
                title=title("Random predictability (multi)", first_p), show=show)
        if "pi_unc" in plot_list:
            add("pi_unc", up.plot_predictability_distribution, pi_list=None, pi_type="pi_unc",
                per_sim_series=ps("pi_unc_per_sim"), filters=first_p,
                title=title("Uncorrelated predictability (multi)", first_p), show=show)
        if "pi_real" in plot_list:
            add("pi_real", up.plot_predictability_distribution, pi_list=None, pi_type="pi_real",
                per_sim_series=ps("pi_real_per_sim"), filters=first_p,
                title=title("Maximum predictability (multi)", first_p), show=show)

        if "2rad_distrib" in plot_list:
            add("2rad_distrib", up.plot_two_rg_ratio_distribution, ratio_list=None,
                per_sim_series=ps("two_rg_ratio_per_sim"), filters=first_p,
                title=title("2-radius ratio (multi)", first_p), show=show)

        if "2rad_distrib_zoomed" in plot_list:
            add("2rad_distrib_zoomed", up.plot_two_rg_ratio_distribution_zoomed, ratio_list=None,
                per_sim_series=ps("two_rg_ratio_per_sim"), filters=first_p,
                title=title("2-radius ratio (zoom) (multi)", first_p), show=show)

        # Venue plots: mp always loads with venue reduction on, so feed the
        # pre-reduced per-sim per-venue arrays.  Fall back to per-user frames
        # only if some series were (unexpectedly) not reduced.
        _all_reduced = all(s.get("venue_reduced", False) for s in self._mp_series)
        if "venue_visits" in plot_list:
            _kw = ({"per_sim_values": ps("venue_visits_per_sim")} if _all_reduced
                   else {"per_sim_series": ps("venue_history_per_sim")})
            add("venue_visits", up.plot_venue_ranking, venue_history_list=None, mode="visits",
                filters=first_p, title=title("Venue Visits Ranking (multi)", first_p),
                show=show, **_kw)
        if "venue_popularity" in plot_list:
            _kw = ({"per_sim_values": ps("venue_users_per_sim")} if _all_reduced
                   else {"per_sim_series": ps("venue_history_per_sim")})
            add("venue_popularity", up.plot_venue_ranking, venue_history_list=None, mode="popularity",
                filters=first_p, title=title("Venue Popularity Ranking (multi)", first_p),
                show=show, **_kw)
        if "venue_visits_distribution" in plot_list:
            _kw = ({"per_sim_values": ps("venue_visits_per_sim")} if _all_reduced
                   else {"per_sim_series": ps("venue_history_per_sim")})
            add("venue_visits_distribution", up.plot_venue_count_distribution,
                venue_history_list=None, mode="visits",
                filters=first_p, title=title("Visit popularity (multi)", first_p),
                show=show, **_kw)
        if "venue_explorers_distribution" in plot_list:
            _kw = ({"per_sim_values": ps("venue_users_per_sim")} if _all_reduced
                   else {"per_sim_series": ps("venue_history_per_sim")})
            add("venue_explorers_distribution", up.plot_venue_count_distribution,
                venue_history_list=None, mode="explorers",
                filters=first_p, title=title("Exploration popularity (multi)", first_p),
                show=show, **_kw)

        # Jensen-Shannon tables: every distribution figure writes its own
        # <folder>/js/<plot>.tsv from inside its (possibly parallel) worker;
        # start from an empty js/ so js_summary.tsv reflects THIS run only.
        if not show:
            import shutil
            _js_dir = up._resolve_plot_dir(subfolder="js")
            shutil.rmtree(_js_dir, ignore_errors=True)

        _run_plot_tasks(tasks, parallel=parallel, max_workers=max_workers, show=show)

        if not show:
            up.merge_js_tables()

        if "batch_stats" in plot_list:
            for s in self._mp_series:
                if s["batch_stats_list"]:
                    sp = {**s["_plot_params"]}
                    up.print_batch_stats(
                        s["batch_stats_list"], filters=sp,
                        title=title(f"Batch Stats [{s['label']}]", sp))

        if "metric_vs_k" in plot_list:
            # Batch statistics of MF as a function of the number of latent
            # factors f (one point per f, mean +/- SEM over the runs); the
            # other series (ge, uk, ...) are drawn as horizontal references.
            # The same statistics are drawn against k = top_k (one curve per
            # recommender) when the series differ in top_k or score_k;
            # each plot is skipped when its parameter does not vary.
            stats, stats_tk = [], []
            for s in self._mp_series:
                pparams = s["_plot_params"]
                stats.append((s["label"], str(pparams.get("algorithm_1", "")),
                              pparams.get("num_latent_factors"), s["batch_stats_list"]))
                stats_tk.append((s["label"], str(pparams.get("algorithm_1", "")),
                                 up.sweep_value(pparams), s["batch_stats_list"]))
            for metric_name, ylabel in METRICS_VS_K.items():
                sc = METRICS_VS_K_SCALE.get(metric_name, 1.0)
                up.plot_metric_vs_latent_factors(stats, metric_name, ylabel,
                                                 filters=first_p, show=show, scale=sc)
                up.plot_metric_vs_top_k(stats_tk, metric_name, ylabel,
                                        filters=first_p, show=show, scale=sc)

        if "metric_summary" in plot_list:
            # One point per series (mean +/- std across runs) for each batch
            # statistic in METRICS_SUMMARY (e.g. the collective Gini).
            _ss = [(s["label"], s["batch_stats_list"]) for s in self._mp_series]
            for metric_name, ylabel in METRICS_SUMMARY.items():
                up.plot_metric_summary(_ss, metric_name, ylabel, filters=first_p,
                                       show=show,
                                       scale=METRICS_VS_K_SCALE.get(metric_name, 1.0))

        if "rank_persistence" in plot_list:
            # Overlay every series' rank persistence, computed PER BATCH (the
            # loader returns one snapshot per batch; metrics are aggregated with
            # a mean+/-SEM band, never by summing cells across batches).
            rec_series, cold_series = [], []
            for s in self._mp_series:
                snap_list = model.utils.load_combined_popularity_snapshots(s["_plot_params"])
                if not snap_list:
                    continue
                rec_batches = _snapshot_pairs(snap_list, "at_t0", "final")
                if rec_batches:
                    rec_series.append((s["label"], rec_batches))
                if s["label"] == "ge":                 # shared baseline -> cold-start ref
                    cold_batches = _snapshot_pairs(snap_list, "init", "at_t0")
                    if cold_batches:
                        cold_series.append(("ge cold-start", cold_batches))
            if rec_series:
                sp = {**self._mp_series[0]["_plot_params"]}
                up.plot_retention(rec_series, transition="recommender", filters=sp, show=show)
                try:
                    _df = up.rank_persistence_stats(rec_series, null_label="ge")
                    logger.debug("Rank-persistence (recommender):\n" + _df.round(4).to_string())
                except Exception as _e:
                    logger.warning(f"rank_persistence_stats failed: {_e}")
                if cold_series:
                    up.plot_retention(cold_series, transition="coldstart", filters=sp, show=show)
            else:
                logger.warning("rank_persistence: no popularity snapshots found for any series.")

        # ------------------------------------------------------------------
        # Time-resolved venue dynamics (multi-series overlay): see
        # _draw_time_resolved.  Each series loads its own per-batch artifacts.
        # ------------------------------------------------------------------
        _all_cont = all(s["_plot_params"].get("combined") for s in self._mp_series)
        if any(pl in plot_list for pl in TIME_RESOLVED_PLOTS) and not _all_cont:
            logger.warning("Time-resolved venue plots need continuations (t0, t1): "
                           "skipped for baselines.")
        elif any(pl in plot_list for pl in TIME_RESOLVED_PLOTS):
            sp = {**self._mp_series[0]["_plot_params"]}
            _sp0 = self._mp_series[0]["_plot_params"].get("starting_position", "core_periphery")
            _draw_time_resolved(
                plot_list,
                [(s["label"], s["_plot_params"]) for s in self._mp_series],
                filters=sp, show=show, mask_center=(_sp0 == "centered"),
                single_dataset=False)

        # r_g and centre of mass by ring (reads the histories, cached)
        if "rg_by_ring" in plot_list and _all_cont:
            up.plot_rg_by_ring(
                [(s["label"], s["_plot_params"]) for s in self._mp_series],
                filters={**self._mp_series[0]["_plot_params"]},
                show=show)

    # ==========================================================================
    # UTILITY
    # ==========================================================================

    def _make_run_folder(self, data_type, params, grid_params=None):
        """
        Create and return a timestamped run folder under a data_type root.

        Folder hierarchy:
            <project_root>/<data_type>/
                ro_{ro}_g_{gamma}_a_{a}_b_{b}_t_{time}_[cl_{cl}_cn_{cn}]_sp_{sp}_alg_{algorithm}/
                    {YYYYMMDD_HHMMSS_ffffff}/

        Parameters
        ----------
        data_type : str
            e.g. "raw_data" or "analysis_data".
        params : dict
            Must contain "ro", "gamma", "a_param", "b_param", "time", "algorithm".
            Optional: "starting_position" (str, default "core_periphery").
        grid_params : dict, optional
            When provided, must contain "cell_lenght" and "cell_number".

        Returns
        -------
        str
            Path to the newly created timestamped run folder.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        param_folder = os.path.join(project_root, data_type,
                                    model.utils.build_param_folder_name(params, grid_params))
        run_folder = os.path.join(param_folder, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        os.makedirs(run_folder, exist_ok=True)
        return run_folder

    def _make_combined_run_folder(self, data_type, base_params, resume_params, grid_params=None, batch_ts=None):
        """
        Create and return a timestamped run folder under a two-level parameter
        hierarchy for chained (load → resume) simulations.

        Folder hierarchy
        ----------------
        <project_root>/<data_type>/
            ro_{ro}_g_{gamma}_a_{a}_b_{b}_t0_{t0}_t1_{t1}
            [_cl_{cl}_cn_{cn}]_sp_{sp}_alg0_{alg0}_alg1_{alg1}/
                {YYYYMMDD_HHMMSS_ffffff}/

        Parameters
        ----------
        data_type : str
            Root directory name, e.g. "combined_raw_data" or
            "combined_analysis_data".
        base_params : dict
            Parameters of the original loaded batch.
            Must contain "ro", "gamma", "a_param", "b_param", "time", "algorithm".
        resume_params : dict
            Parameters used for the resume phase.
            Must contain "ro", "gamma", "a_param", "b_param", "time", "algorithm".
        grid_params : dict, optional
            When provided, must contain "cell_lenght" and "cell_number".
        batch_ts : str, optional
            Timestamp string in the format "YYYYMMDD_HHMMSS_ffffff".
            When provided, this exact string is used as the batch folder name
            so that combined_raw_data, combined_analysis_data and the check
            folder all share the same ID.  When None, a fresh timestamp is
            generated.

        Returns
        -------
        str
            Path to the newly created timestamped run folder.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_folder  = os.path.join(project_root, data_type)

        # The baseline fixes ro/gamma/a/b, t0, the starting position, the
        # population and the power-law bounds; the continuation fixes t1, the
        # recommender and its knobs (top_k, score_k, cf_updates, f).
        filters = {k: resume_params[k] for k in
                   ("top_k", "score_k", "cf_updates", "num_latent_factors")
                   if k in resume_params}
        filters.update({k: base_params[k] for k in
                        ("ro", "gamma", "a_param", "b_param", "spatial_min",
                         "spatial_max", "temporal_min", "temporal_max")
                        if k in base_params})
        filters.update(
            t0=base_params["time"], t1=resume_params["time"],
            starting_position=base_params.get("starting_position", "core_periphery"),
            n_users=base_params.get("user_number"),
            algorithm_0=base_params.get("algorithm", "ce"),
            algorithm_1=resume_params.get("algorithm", "ce"))
        param_folder_name = model.utils._build_combined_param_folder_name(
            filters, grid_params)

        if batch_ts is None:
            batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_folder = os.path.join(
            base_folder,
            param_folder_name,
            batch_ts,
        )
        os.makedirs(run_folder, exist_ok=True)
        return run_folder