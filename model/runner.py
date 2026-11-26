"""
model/runner.py
===============
The operations behind main.py: simulate, continue, reanalyse, plot and export
the statistics.  main.py only holds the configuration and calls these.

Vocabulary
----------
baseline       Phase I: a run from t = 0 to t0 with an EPR model (ge, ce, de).
               Saved in raw_data/ and analysis_data/.
continuation   Phase II: a baseline continued from t0 to t1, with a
               recommender (uk, ik, mf, ra) or with EPR (ge).
               Saved in combined_raw_data/ and combined_analysis_data/.
realisation    one baseline together with its continuations.

Parameters
----------
Every function takes the dictionary P of main.py (PARAMS): the values shared
by all runs.  A continuation (or a plotted series) is given either by the name
of its algorithm ("uk") or by a dictionary with the algorithm and the
parameters that differ from P, e.g. {"algorithm": "mf", "num_latent_factors": 5}.
"""
import hashlib
import json
import logging
import os
import re
import time

import model.simulation
import model.utils
import model.utils_plot

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY = 86400

# Parameters shared by the baseline and the continuation of a run.
SHARED_KEYS = ("ro", "gamma", "a_param", "b_param",
               "spatial_min", "spatial_max", "temporal_min", "temporal_max")
# Keys a continuation (or a plotted series) may change with respect to P.
CONTINUATION_KEYS = ("t1", "top_k", "score_k", "cf_updates", "num_latent_factors")
# Keys that change the baseline: allowed only in plotted series.
BASELINE_KEYS = ("t0", "n_users", "starting_position", "baseline_algorithm")
EPR_ALGORITHMS = ("ce", "ge", "de")
RECOMMENDERS = ("uk", "ik", "mf", "ra")

# Every figure the plot dispatchers know about (described in main.py, PLOT).
ALL_PLOTS = [
    "rg_distribution", "rg_e_dist", "rg_over_time", "two_radius", "2rad_distrib",
    "2rad_distrib_zoomed", "dist_distribution", "s_over_time", "num_venues_distribution",
    "freq_distribution", "mobility",
    "srand", "sunc", "sreal", "s_triple", "pi_rand", "pi_unc", "pi_real", "pi_triple",
    "venue_visits", "venue_popularity", "venue_visits_distribution",
    "venue_explorers_distribution",
    "aggregate_grid", "aggregate_grid_exploration", "batch_grid", "batch_grid_exploration",
    "grid_final_comparison", "aggregate_grid_snapshots",
    "kendall_over_time", "top10_turnover", "top10_rank_tracking",
    "top10_membership_panels", "rank_origin", "rank_persistence",
    "venue_velocity_fixed_top", "venue_velocity_fixed_worst", "venue_velocity_fixed_final",
    "rg_by_ring", "batch_stats", "metric_vs_k", "metric_summary",
]
# Marks of main.py: [1] one series only, [M] several series only, [C]
# continuations only.  plots = "all" takes only the figures that the data
# asked can have (and not "batch_stats", a long printout with no figure);
# a figure named explicitly and not available is skipped with a warning.
ONE_SERIES_ONLY = {"freq_distribution", "mobility", "s_triple", "pi_triple",
                   "aggregate_grid", "aggregate_grid_exploration", "batch_grid",
                   "batch_grid_exploration", "aggregate_grid_snapshots"}
MANY_SERIES_ONLY = {"metric_vs_k", "metric_summary"}
CONTINUATION_ONLY = set(model.simulation.TIME_RESOLVED_PLOTS) | {
    "rank_persistence", "grid_final_comparison", "rg_by_ring", "metric_vs_k"}


def all_plots(data, n_series):
    """The figures of plots = "all" for these data."""
    skip = {"batch_stats"}
    skip |= ONE_SERIES_ONLY if n_series > 1 else MANY_SERIES_ONLY
    if data == "baseline":
        skip |= CONTINUATION_ONLY
    return [p for p in ALL_PLOTS if p not in skip]


# =============================================================================
# Parameters
# =============================================================================
def grid_of(P, algorithm=None):
    """Grid parameters, or None for the grid-less classic EPR ("ce")."""
    if (algorithm or P.get("baseline_algorithm", "ge")) == "ce":
        return None
    return {"cell_lenght": P["cell_lenght"], "cell_number": P["cell_number"]}


def as_spec(item):
    """"uk" -> {"algorithm": "uk"};  a dict is copied and checked."""
    if isinstance(item, str):
        return {"algorithm": item}
    if not isinstance(item, dict) or "algorithm" not in item:
        raise ValueError(f"a series/continuation must be an algorithm name or a dict "
                         f"with the key 'algorithm', not {item!r}")
    bad = set(item) - {"algorithm", "label"} - set(CONTINUATION_KEYS) - set(BASELINE_KEYS)
    if bad:
        raise ValueError(f"unknown keys {sorted(bad)} in {item!r}; allowed: algorithm, "
                         f"label, {', '.join(CONTINUATION_KEYS + BASELINE_KEYS)}")
    return dict(item)


def label_of(spec):
    """Label of a series in legends and colours: the explicit "label", else the
    algorithm plus what differs (mf5, uk_ktop10, uk_k5, ik_ra, ...)."""
    if spec.get("label"):
        return spec["label"]
    alg = spec["algorithm"]
    lab = alg
    if "num_latent_factors" in spec:
        lab = f"{alg}{spec['num_latent_factors']}"
    if "top_k" in spec:
        lab += f"_ktop{spec['top_k']}"
    if "score_k" in spec:
        lab += f"_k{spec['score_k']}"
    if "starting_position" in spec:
        lab += f"_{model.utils.sp_abbr(spec['starting_position'])}"
    for k in ("t0", "t1", "cf_updates", "n_users", "baseline_algorithm"):
        if k in spec:
            lab += f"_{k}{spec[k]}"
    return lab


def baseline_params(P, seed=None):
    """simulate_params of a baseline."""
    p = {k: P[k] for k in SHARED_KEYS}
    p.update(time=P["t0"], user_number=P["n_users"],
             starting_position=P["starting_position"],
             algorithm=P.get("baseline_algorithm", "ge"))
    if seed is not None:
        p["seed"] = int(seed)
    return p


def continuation_params(P, spec, seed=None):
    """resume_params of a continuation."""
    p = {k: P[k] for k in SHARED_KEYS}
    p.update(time=spec.get("t1", P["t1"]), algorithm=spec["algorithm"],
             top_k=spec.get("top_k", P["top_k"]),
             score_k=spec.get("score_k", P.get("score_k", 25)),
             cf_updates=spec.get("cf_updates", P["cf_updates"]),
             num_latent_factors=spec.get("num_latent_factors", P["num_latent_factors"]))
    if seed is not None:
        p["seed"] = int(seed)
    return p


def derived_seed(*parts):
    """A stable seed in [0, 2^31) from any list of values."""
    h = hashlib.sha1("|".join(str(x) for x in parts).encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31)


def _check_continuations(P, specs):
    base_alg = P.get("baseline_algorithm", "ge")
    for spec in specs:
        if spec["algorithm"] not in EPR_ALGORITHMS + RECOMMENDERS:
            raise ValueError(f"unknown algorithm {spec['algorithm']!r}: use one of "
                             f"{', '.join(EPR_ALGORITHMS + RECOMMENDERS)}")
        bad = [k for k in BASELINE_KEYS if k in spec]
        if bad:
            raise ValueError(f"{spec}: {bad} would change the baseline; set them in "
                             f"PARAMS instead")
        if spec["algorithm"] in EPR_ALGORITHMS and spec["algorithm"] != base_alg:
            raise ValueError(f"an EPR continuation ({spec['algorithm']}) is meaningful only "
                             f"on a baseline of the same model ({base_alg})")


# =============================================================================
# Simulate / continue
# =============================================================================
def baseline_folder(P):
    """raw_data parameter folder of the baselines described by P."""
    bp = baseline_params(P)
    return os.path.join(ROOT, "raw_data",
                        model.utils.build_param_folder_name(bp, grid_of(P)))


def list_batches(folder):
    """Batch ids (timestamps) in a parameter folder, oldest first."""
    if not os.path.isdir(folder):
        return []
    return sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))


class _Phases:
    """Duration of the phases of one run, for its final line."""

    def __init__(self):
        self.t0 = self.t = time.time()
        self.parts = []

    def mark(self, name):
        now = time.time()
        self.parts.append(f"{name} {model.utils.fmt_duration(now - self.t)}")
        self.t = now

    def __str__(self):
        return (f"{model.utils.fmt_duration(time.time() - self.t0)} "
                f"({', '.join(self.parts)})")


def run_baseline(P, seed=None):
    """Simulate and analyse one baseline.  Returns (batch_id, seed)."""
    model.utils.reset_run_log_buffer()
    bp = baseline_params(P, seed)
    grid = grid_of(P)
    logger.info(f"baseline {bp['algorithm']}, {bp['user_number']} individuals")
    ph = _Phases()
    sim = model.simulation.Simulation()
    sim.run(bp, grid_params=grid)
    ph.mark("simulation")
    raw = sim.save_raw_data()
    ph.mark("saving")
    sim.run_analysis(bp, raw_run_folder=raw, grid_params=grid)
    ph.mark("analysis")
    bid = os.path.basename(raw)
    used = sim._simulate_params.get("seed")
    logger.info(f"baseline {bid} done in {ph}, seed {used}")
    return bid, used


def run_continuation(P, batch_id, spec, seed=None):
    """Continue the baseline batch_id with one algorithm, then analyse."""
    model.utils.reset_run_log_buffer()
    grid = grid_of(P)
    logger.info(f"continuation {label_of(spec)} of baseline {batch_id}")
    ph = _Phases()
    sim = model.simulation.Simulation()
    sim._reconstruct_users_from_raw(dict(baseline_params(P), batch_id=batch_id),
                                    grid_params=grid)
    ph.mark("reconstruction")
    rp = continuation_params(P, spec, seed)
    sim.resume(rp, grid_params=grid)
    ph.mark("simulation")
    folder = sim.save_combined_raw_data()
    ph.mark("saving")
    sim.run_analysis(rp, raw_run_folder=folder, grid_params=grid, combined=True)
    ph.mark("analysis")
    logger.info(f"continuation {label_of(spec)} done in {ph}")
    return os.path.basename(folder)


def simulate(P, continuations=(), cycles=1, seed=None):
    """`cycles` realisations: each is a NEW baseline, analysed, then continued
    (and analysed) with every algorithm in `continuations`.
    seed None = random; an int s = realisation i uses the seed s + i for the
    baseline and a seed derived from it for each continuation."""
    specs = [as_spec(c) for c in continuations]
    _check_continuations(P, specs)
    for i in range(int(cycles)):
        logger.info(f"=== realisation {i + 1}/{cycles} ===")
        bseed = None if seed is None else int(seed) + i
        bid, used = run_baseline(P, bseed)
        for spec in specs:
            run_continuation(P, bid, spec, derived_seed(used, label_of(spec)))


def select_baselines(P, which):
    """Baselines of P on disk: an int N = the N oldest, "all", or a list of ids."""
    ids = list_batches(baseline_folder(P))
    if not ids:
        raise SystemExit(f"no baseline with these parameters:\n  {baseline_folder(P)}\n"
                         + closest_message(baseline_folder(P)))
    if which == "all":
        return ids
    if isinstance(which, int):
        if which > len(ids):
            raise SystemExit(f"asked for {which} baselines, only {len(ids)} on disk")
        return ids[:which]
    missing = [b for b in which if b not in ids]
    if missing:
        raise SystemExit(f"baselines not found: {missing}")
    return list(which)


def continue_runs(P, baselines, continuations, seed=None):
    """Continue existing baselines (see select_baselines) with every algorithm
    in `continuations`.  seed None = random; an int = a seed derived from it,
    from the baseline id and from the algorithm."""
    specs = [as_spec(c) for c in continuations]
    _check_continuations(P, specs)
    ids = select_baselines(P, baselines)
    logger.info(f"continuing {len(ids)} baseline(s) with {[label_of(s) for s in specs]}")
    for bid in ids:
        for spec in specs:
            run_continuation(P, bid, spec,
                             None if seed is None else derived_seed(seed, bid, label_of(spec)))


# =============================================================================
# Reanalyse / export
# =============================================================================
def reanalyse(what="all", metrics=None):
    """Redo the analysis of every run on disk from its raw data.
    what: "baseline" | "continuation" | "all".  metrics: None = everything;
    a list of batch metrics = only those, merged into batch_stats (fast)."""
    sim = model.simulation.Simulation()
    if what in ("baseline", "all"):
        sim.analyse_all(metrics=metrics)
    if what in ("continuation", "all"):
        sim.analyse_all_combined(metrics=metrics)
    if what not in ("baseline", "continuation", "all"):
        raise ValueError("what must be 'baseline', 'continuation' or 'all'")


def export_stats():
    written = model.utils.export_all_stats()
    print("\nExported statistics:")
    for kind, path in written.items():
        print(f"  {kind:<13s} -> {path if path else '(no data found)'}")
    return written


# =============================================================================
# Plot
# =============================================================================
def _data_root(data):
    return os.path.join(ROOT, "analysis_data" if data == "baseline"
                        else "combined_analysis_data")


def plot_params(P, data, specs, first_run_only=False):
    """(common params, per-series dicts) in the format of the loaders."""
    shared = {k: P[k] for k in SHARED_KEYS}
    slice_ = P["n_users"] if first_run_only else None
    if data == "continuation":
        common = dict(shared, t0=P["t0"], t1=P["t1"], user_number=slice_,
                      n_users=P["n_users"], top_k=P["top_k"],
                      score_k=P.get("score_k", 25), cf_updates=P["cf_updates"],
                      num_latent_factors=P["num_latent_factors"],
                      algorithm_0=P.get("baseline_algorithm", "ge"), combined=True)
        series = []
        for spec in specs:
            d = {"algorithm_1": spec["algorithm"],
                 "starting_position": spec.get("starting_position", P["starting_position"])}
            for k in ("t0", "t1", "n_users", "top_k", "score_k", "cf_updates",
                      "num_latent_factors"):
                if k in spec:
                    d[k] = spec[k]
            if "baseline_algorithm" in spec:
                d["algorithm_0"] = spec["baseline_algorithm"]
            series.append(d)
    elif data == "baseline":
        # (no "combined" key: the baseline loader matches every key it gets
        # against the tokens of the folder names)
        common = dict(shared, time=P["t0"], user_number=slice_, n_users=P["n_users"])
        series = []
        for spec in specs:
            bad = [k for k in CONTINUATION_KEYS if k in spec]
            if bad:
                raise ValueError(f"{spec}: {bad} do not apply to a baseline")
            d = {"algorithm": spec["algorithm"],
                 "starting_position": spec.get("starting_position", P["starting_position"])}
            if "t0" in spec:
                d["time"] = spec["t0"]
            if "n_users" in spec:
                d["n_users"] = spec["n_users"]
            series.append(d)
    else:
        raise ValueError("data must be 'baseline' or 'continuation'")
    return common, series


def series_folder(P, data, common, ser):
    """Name of the analysis parameter folder of one series."""
    f = {**common, **ser}
    if data == "continuation":
        return model.utils._build_combined_param_folder_name(
            f, grid_of(P, f.get("algorithm_0")))
    return model.utils.build_param_folder_name(f, grid_of(P, f.get("algorithm")))


def count_runs(data, folder_name):
    path = os.path.join(_data_root(data), folder_name)
    return len(list_batches(path))


def closest_message(path, n=5):
    """The parameter folders next to `path` closest to it, with the tokens in
    which they differ: what to change to point at data that exists."""
    root, name = os.path.dirname(path), os.path.basename(path)
    if not os.path.isdir(root):
        return f"  ({os.path.relpath(root, ROOT)}/ does not exist)"

    defaults = {t: str(v) for t, v in model.utils.powerlaw_token_defaults().items()}

    def toks(s):
        p = s.split("_")
        return {**defaults, **dict(zip(p[0::2], p[1::2]))} if len(p) % 2 == 0 else {}

    want = toks(name)
    scored = []
    for other in os.listdir(root):
        have = toks(other)
        if not have:
            continue
        diff = [f"{k}: {want.get(k, '-')} -> {have.get(k, '-')}"
                for k in sorted(set(want) | set(have)) if want.get(k) != have.get(k)]
        scored.append((len(diff), other, diff))
    scored.sort()
    lines = ["  closest existing data (what differs: asked -> on disk):"]
    for _n, other, diff in scored[:n]:
        lines.append(f"    {len(list_batches(os.path.join(root, other)))} run(s)  "
                     + "; ".join(diff))
    return "\n".join(lines) if scored else "  (no data at all)"


def check_series(P, data, common, series, labels, verbose=True):
    """Number of runs of every series; prints them.  True if all have data."""
    ok = True
    for lab, ser in zip(labels, series):
        name = series_folder(P, data, common, ser)
        n = count_runs(data, name)
        if verbose:
            print(f"   {lab:14s} {n:3d} run(s)")
        if n == 0:
            ok = False
            print(f"   !! no data for '{lab}':\n      {name}\n"
                  + closest_message(os.path.join(_data_root(data), name)))
    return ok


def plot(P, data="continuation", series=("ge", "uk", "ik", "mf"), plots="all",
         output=None, first_run_only=False, show=False, out_root=None,
         time_series_metrics=None, verbose=True):
    """One series -> the single-dataset figures; several -> overlaid
    (multiplot).  Figures go in plots/<output>/ (output None = a name made from
    the series and a hash of the parameters), or in out_root/<output> when
    out_root is given.  Returns the output folder, or None if data is missing."""
    up = model.utils_plot
    specs = [as_spec(s) for s in series]
    labels = [label_of(s) for s in specs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"two series have the same label: {labels}; give them a 'label'")
    plot_list = all_plots(data, len(specs)) if plots == "all" else list(plots)
    common, sers = plot_params(P, data, specs, first_run_only)

    if verbose:
        print(f"\n{data}, {len(sers)} series:")
    if not check_series(P, data, common, sers, labels, verbose=verbose):
        return None

    if output is None:
        sig = json.dumps([common, sers, labels], sort_keys=True, default=str)
        readable = re.sub(r"[^A-Za-z0-9.]+", "_", "-".join(labels)).strip("_")
        output = (f"{data}_{readable}"[:120]
                  + "__" + hashlib.sha1(sig.encode()).hexdigest()[:8])
    out = os.path.join(out_root or os.path.join(ROOT, "plots"), output)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "plot_info.txt"), "w") as fh:
        fh.write(f"data = {data}\n")
        for k in sorted(common):
            fh.write(f"{k} = {common[k]}\n")
        for lab, ser in zip(labels, sers):
            fh.write(f"series {lab} = {ser}\n")

    n_cpu = os.cpu_count() or 1
    old_metrics = model.simulation.TIME_SERIES_METRICS
    if time_series_metrics is not None:
        model.simulation.TIME_SERIES_METRICS = tuple(time_series_metrics)
    up.PLOT_OUTPUT_DIR = out
    up.FIRST_RUN_ONLY = bool(first_run_only)
    up._ACTIVE_LABELS = labels
    t = time.time()
    try:
        sim = model.simulation.Simulation()
        logger.info(f"loading {len(sers)} series ...")
        if len(sers) == 1:
            p = {**common, **sers[0]}
            sim.load_analysis_data(p, grid_params=grid_of(P, p.get("algorithm_0",
                                                                  p.get("algorithm"))),
                                   plot_list=plot_list)
            logger.info(f"  series {labels[0]} loaded: {getattr(sim, '_n_sims', 1)} run(s), "
                        f"{len(sim.rg_list)} individuals")
            logger.info("drawing the figures ...")
            sim.plot(plot_list, show=show, parallel=True, max_workers=n_cpu)
        else:
            mp = dict(common, label=labels, series=sers)
            sim.load_multiple_plot_data(mp, grid_params=grid_of(P), plot_list=plot_list,
                                        load_parallel=True,
                                        load_workers=max(1, min(len(sers), n_cpu)))
            logger.info("drawing the figures ...")
            sim.multipleplot(plot_list, show=show, parallel=True, max_workers=n_cpu)
    finally:
        up.PLOT_OUTPUT_DIR = None
        up.FIRST_RUN_ONLY = False
        model.simulation.TIME_SERIES_METRICS = old_metrics
    n_pdf = sum(f.endswith(".pdf") for _r, _d, fs in os.walk(out) for f in fs)
    logger.info(f"{n_pdf} figures in {os.path.relpath(out, ROOT)}/ "
                f"({model.utils.fmt_duration(time.time() - t)})")
    return out