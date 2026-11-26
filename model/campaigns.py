"""
model/campaigns.py
==================
The simulation campaigns of the thesis, behind the commands

    python main.py simulate_thesis <campaign ...> [--cycles N] [--check]
    python main.py plot_thesis     <campaign ...> [--check]

Campaigns
---------
long             main campaign (Chapter 4): new baselines (t0 = 4 days),
                 continued with g-EPR, User-kNN, Item-kNN, MF up to 64 days.
f                MF with f = 3, 5, 10, 15, 30 latent factors (f = 20 is the
                 main campaign), on the baselines of "long".
initialization   as "long", with random initial positions (new baselines).
k_top            length of the lists k_top = 5, 10, 15, 50 (25 is the main
                 campaign), User-kNN / Item-kNN / MF, on the baselines of "long".
k                number of terms in the score k = 5, 10, 15, 50 (25 is the main
                 campaign), User-kNN / Item-kNN, on the baselines of "long".
baseline_length  baselines of 32 days instead of 4 (new baselines), then the
                 same 60 days of continuation as "long".
"sensitivity" = every campaign but "long";  "all" = every campaign.

Realisations
------------
A realisation is one baseline with all the continuations of the campaign.
--cycles N means "reach N realisations": what is already on disk is found
(from the "load batch_id" written in the run_info.txt of the continuations)
and only what is missing is simulated, so a campaign can be split over
several launches and resumed after an interruption.  The campaigns on the
baselines of "long" use its first N realisations (those already used by the
campaign first), so "long" must have at least N realisations.

Seeds: the baseline of realisation i of a campaign gets a seed derived from
(campaign, i); each continuation a seed derived from (campaign, series,
seed of its baseline).  Same code + same seeds -> same data.
"""
import logging
import os
import time

import model.runner as R
import model.simulation
import model.utils
import model.utils_plot

logger = logging.getLogger(__name__)
DAY = R.DAY

# Parameters of the thesis (Table 3.1 and Section 3.2.4).
THESIS_PARAMS = {
    "ro": 0.6, "gamma": 0.21, "a_param": 0.55, "b_param": 0.8,
    "spatial_min": 1000, "spatial_max": 100000,          # 1 km, 100 km
    "temporal_min": 3600, "temporal_max": 61200,          # 1 h, 17 h
    "cell_lenght": 1400, "cell_number": 51,
    "n_users": 6000,
    "starting_position": "core_periphery",
    "baseline_algorithm": "ge",
    "t0": 4 * DAY, "t1": 64 * DAY,
    "top_k": 25, "score_k": 25, "cf_updates": 14, "num_latent_factors": 20,
}
FOUR = ["ge", "uk", "ik", "mf"]
F_VALUES = [3, 5, 10, 15, 30]          # + 20 of the main campaign
K_VALUES = [5, 10, 15, 50]             # + 25 of the main campaign

# Figures of the Results chapter, made for every campaign with the four dynamics.
RESULTS_PLOTS = ["rg_distribution", "dist_distribution", "2rad_distrib", "rg_over_time",
                 "venue_explorers_distribution", "grid_final_comparison",
                 "kendall_over_time", "top10_turnover", "top10_membership_panels",
                 "rank_origin", "venue_velocity_fixed_top", "venue_velocity_fixed_worst",
                 "rg_by_ring"]


class Campaign:
    def __init__(self, name, params, continuations, cycles, reuse_long=False,
                 description=""):
        self.name = name
        self.P = dict(THESIS_PARAMS, **params)
        self.specs = [R.as_spec(c) for c in continuations]
        self.cycles = cycles                     # default number of realisations
        self.reuse_long = reuse_long             # continue the baselines of "long"
        self.description = description


CAMPAIGNS = {
    "long": Campaign("long", {}, FOUR, 10,
                     description="main campaign: g-EPR, User-kNN, Item-kNN, MF"),
    "f": Campaign("f", {}, [{"algorithm": "mf", "num_latent_factors": f}
                            for f in F_VALUES], 10, reuse_long=True,
                  description="MF, f = 3, 5, 10, 15, 30"),
    "initialization": Campaign("initialization", {"starting_position": "random"}, FOUR, 10,
                               description="random initial positions"),
    "k_top": Campaign("k_top", {}, [{"algorithm": a, "top_k": v}
                                    for v in K_VALUES for a in ("uk", "ik", "mf")],
                      5, reuse_long=True, description="k_top = 5, 10, 15, 50"),
    "k": Campaign("k", {}, [{"algorithm": a, "score_k": v}
                            for v in K_VALUES for a in ("uk", "ik")],
                  5, reuse_long=True, description="k = 5, 10, 15, 50"),
    "baseline_length": Campaign("baseline_length", {"t0": 32 * DAY, "t1": 92 * DAY}, FOUR, 10,
                                description="baselines of 32 days"),
}
ORDER = list(CAMPAIGNS)                       # "long" first: the others may need it
SENSITIVITY = [c for c in ORDER if c != "long"]


def resolve(names):
    """Command-line names -> ordered list of campaign names."""
    out = []
    for n in names or ["all"]:
        if n == "all":
            out += ORDER
        elif n == "sensitivity":
            out += SENSITIVITY
        elif n in CAMPAIGNS:
            out.append(n)
        else:
            raise SystemExit(f"unknown campaign '{n}'; choose among "
                             f"{ORDER + ['sensitivity', 'all']}")
    return [c for c in ORDER if c in out]


# =============================================================================
# What is on disk
# =============================================================================
def _continuation_folder(P, spec):
    """combined_analysis_data parameter folder of one continuation."""
    common, sers = R.plot_params(P, "continuation", [spec])
    return os.path.join(R.ROOT, "combined_analysis_data",
                        R.series_folder(P, "continuation", common, sers[0]))


def baselines_done(P, spec):
    """Baselines (batch ids) already continued with spec and analysed."""
    folder = _continuation_folder(P, spec)
    done = set()
    for run in R.list_batches(folder):
        b = model.utils.read_run_info_value(os.path.join(folder, run), "load batch_id")
        if b:
            done.add(b)
    return done


def status(camp):
    """{spec label: set of baselines done} and the complete realisations."""
    done = {R.label_of(s): baselines_done(camp.P, s) for s in camp.specs}
    used = set().union(*done.values()) if done else set()
    complete = sorted(b for b in used if all(b in d for d in done.values()))
    return done, sorted(used), complete


def long_baselines():
    return status(CAMPAIGNS["long"])[2]


def plan(camp, n):
    """[(baseline id or None for a new one, [specs to run])], to reach n."""
    done, used, complete = status(camp)
    if camp.reuse_long:
        pool = long_baselines()
        if len(pool) < n:
            raise SystemExit(f"{camp.name}: needs {n} realisations of 'long', there are "
                             f"{len(pool)}: run first  python main.py simulate_thesis long "
                             f"--cycles {n}")
        ordered = [b for b in pool if b in used] + [b for b in pool if b not in used]
        chosen = ordered[:n]
        return [(b, [s for s in camp.specs if b not in done[R.label_of(s)]])
                for b in chosen if any(b not in done[R.label_of(s)] for s in camp.specs)]
    need = n - len(complete)
    if need <= 0:
        return []
    partial = [b for b in used if b not in complete][:need]
    out = [(b, [s for s in camp.specs if b not in done[R.label_of(s)]]) for b in partial]
    out += [(None, list(camp.specs))] * (need - len(partial))
    return out


# =============================================================================
# simulate_thesis
# =============================================================================
def _baseline_seed(P, batch_id):
    folder = R.baseline_folder(P)
    return model.utils.read_run_info_value(os.path.join(folder, batch_id), "seed")


def simulate_thesis(names, cycles=None, check=False):
    chosen = resolve(names)
    print()
    total = 0
    plans = {}
    for name in chosen:
        camp = CAMPAIGNS[name]
        n = cycles or camp.cycles
        done, used, complete = status(camp)
        n_have = len(complete) if not camp.reuse_long else \
            sum(1 for b in long_baselines()[:n] if all(b in d for d in done.values()))
        print(f"== {name} ({camp.description}): {n_have} realisation(s) complete, "
              f"target {n}")
        try:
            p = plan(camp, n)
        except SystemExit as exc:
            if "long" in chosen and camp.reuse_long:
                print(f"   to do: up to {n * len(camp.specs)} continuation(s) on the "
                      f"baselines of 'long' (made first, in this launch)")
                p = None
            else:
                print(f"   !! {exc}")
                continue
        plans[name] = (n, p)
        if p:
            n_new = sum(b is None for b, _ in p)
            n_cont = sum(len(s) for _, s in p)
            total += n_cont
            print(f"   to do: {n_new} new baseline(s), {n_cont} continuation(s)")
        elif p is not None:
            print("   nothing to do")
    if check:
        print(f"\nOnly a check.  To simulate:  python main.py simulate_thesis "
              f"{' '.join(names or ['all'])}"
              + (f" --cycles {cycles}" if cycles else ""))
        print("(one continuation of 6000 individuals for 60 days takes ~15 minutes)\n")
        return

    model.utils.install_run_log_buffer()
    t_all = time.time()
    for name in chosen:
        if name not in plans:
            continue
        camp = CAMPAIGNS[name]
        n, p = plans[name]
        if p is None:                       # depends on "long", simulated just now
            p = plan(camp, n)
        logger.info(f"=== {name}: {len(p)} realisation(s) to complete ===")
        n_base = len(status(camp)[1])
        for k, (bid, specs) in enumerate(p, 1):
            logger.info(f"=== {name}: realisation {k}/{len(p)} ===")
            if bid is None:
                seed = R.derived_seed(name, "baseline", n_base)
                n_base += 1
                bid, bseed = R.run_baseline(camp.P, seed)
            else:
                bseed = _baseline_seed(camp.P, bid) or bid
            for spec in specs:
                R.run_continuation(camp.P, bid, spec,
                                   R.derived_seed(name, R.label_of(spec), bseed))
    logger.info(f"simulate_thesis done in {model.utils.fmt_duration(time.time() - t_all)}")


# =============================================================================
# plot_thesis
# =============================================================================
OUT_ROOT = os.path.join(R.ROOT, "thesis_plot")


def plot_passes(name):
    """[(output subfolder, P, series, plots, options)] of a campaign."""
    camp = CAMPAIGNS[name]
    P = camp.P
    if name == "long":
        return [("main_campaign", P, FOUR, RESULTS_PLOTS, {})]
    if name == "initialization":
        return [("sensitivity/initialization", P, FOUR, RESULTS_PLOTS, {})]
    if name == "baseline_length":
        days0 = P["t0"] / DAY
        return [("sensitivity/baseline_length", P, FOUR, RESULTS_PLOTS,
                 {"zoom": (days0 - 4, days0 + 12)})]
    if name == "f":
        fs = sorted(set(F_VALUES + [P["num_latent_factors"]]))   # + f of the main campaign
        series = ["ge"] + [{"algorithm": "mf", "num_latent_factors": f, "label": f"mf{f}"}
                           for f in fs]
        grids = ["ge"] + [f"mf{f}" for f in fs if f != 15]       # the 6 panels of the thesis
        return [("sensitivity/f", P, series, ["grid_final_comparison", "metric_vs_k"],
                 {"grids": grids})]
    if name in ("k_top", "k"):
        key = "top_k" if name == "k_top" else "score_k"
        algs = ("uk", "ik", "mf") if name == "k_top" else ("uk", "ik")
        symbol = r"$k_{\mathrm{top}}$" if name == "k_top" else "$k$"
        suffix = "ktop" if name == "k_top" else "k"
        out = []
        for v in K_VALUES:
            series = ["ge"] + [{"algorithm": a, key: v, "label": a} for a in algs]
            out.append((f"sensitivity/{name}/{name}_{v}", P, series, RESULTS_PLOTS,
                        {"tag": f"{symbol} = {v}"}))
        values = sorted(set(K_VALUES + [P[key]]))                # + value of the main campaign
        summary = ["ge"] + [{"algorithm": a, key: v, "label": f"{a}_{suffix}{v}"}
                            for a in algs for v in values]
        out.append((f"sensitivity/{name}/summary", P, summary, ["metric_vs_k"],
                    {"sweep": name}))
        return out
    raise ValueError(name)


def plot_thesis(names, check=False):
    chosen = resolve(names)
    up = model.utils_plot
    passes = [(name, *p) for name in chosen for p in plot_passes(name)]
    print(f"\nfigures in {os.path.relpath(OUT_ROOT, R.ROOT)}/")
    ok = []
    for name, sub, P, series, plots, opt in passes:
        specs = [R.as_spec(s) for s in series]
        common, sers = R.plot_params(P, "continuation", specs)
        print(f"\n== {sub}")
        if R.check_series(P, "continuation", common, sers, [R.label_of(s) for s in specs]):
            ok.append((name, sub, P, series, plots, opt))
        else:
            print(f"   -> {sub}: data missing, SKIPPED")
    if check:
        print(f"\nOnly a check.  To make the figures:  python main.py plot_thesis "
              f"{' '.join(names or ['all'])}\n")
        return

    model.utils.install_run_log_buffer()
    t_all = time.time()
    for name, sub, P, series, plots, opt in ok:
        up.PANEL_TAG = opt.get("tag", "")
        up.GRID_FINAL_SERIES = opt.get("grids")
        up.SWEEP = opt.get("sweep")
        old_zoom = up.ZOOM_WINDOW
        if "zoom" in opt:
            up.ZOOM_WINDOW = opt["zoom"]
        logger.info(f"=== {sub} ===")
        try:
            R.plot(P, "continuation", series, plots, output=sub, out_root=OUT_ROOT,
                   time_series_metrics=("explorers",), verbose=False)
        except Exception as exc:
            logger.exception(f"{sub} FAILED: {exc}")
        finally:
            up.PANEL_TAG, up.GRID_FINAL_SERIES, up.SWEEP = "", None, None
            up.ZOOM_WINDOW = old_zoom
    logger.info(f"plot_thesis done in {model.utils.fmt_duration(time.time() - t_all)}")