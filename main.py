"""
main.py  
====================================================

    python main.py                 run MODE, with the settings below
    python main.py <mode>          the same, with another mode (simulate,
                                   continue, reanalyse, plot, export_stats,
                                   simulate_thesis, plot_thesis, test)

    python main.py test            check that the whole code works (a few
                                   minutes, in a temporary copy: the data of
                                   the project are not touched)

    python main.py simulate_thesis <campaigns> [--cycles N] [--check]
    python main.py plot_thesis     <campaigns> [--check]
        repeat the simulations / remake the figures of the thesis.  The same
        as MODE = "simulate_thesis" / "plot_thesis" with the block THESIS
        below; what is written on the command line wins over THESIS.
        campaigns: long, f, initialization, k_top, k, baseline_length,
                   sensitivity (= all but long), all (default).
        With --check they only print what is on disk and what is missing,
        without simulating / plotting.
        --cycles N: reach N realisations per campaign (default: those of the
        thesis, 10, and 5 for k_top and k).  Data in the usual folders,
        figures in thesis_plot/main_campaign/ and thesis_plot/sensitivity/.

    --verbose (with any mode): every detail on the terminal, as VERBOSE = True.

Vocabulary
    baseline       Phase I: a run from t = 0 to t0 with an EPR model.
    continuation   Phase II: a baseline continued from t0 to t1 with a
                   recommender (uk, ik, mf, ...) or with EPR (ge).
Every run is analysed right after it is simulated.  Simulated data go in
raw_data/ (baselines) and combined_raw_data/ (continuations), their analysis
in analysis_data/ and combined_analysis_data/.
"""
import os
import sys

import model.campaigns
import model.runner
import model.selftest
import model.utils

# =============================================================================
# MODE
# =============================================================================
MODE = "plot"
#   "simulate"         new baselines (+ their continuations), see SIMULATE
#   "continue"         continue baselines already on disk, see CONTINUE
#   "reanalyse"        redo the analysis of the data on disk, see REANALYSE
#   "plot"             figures of one or more datasets, see PLOT
#   "export_stats"     statistics of every dataset (mean, std, sem over the
#                      runs) in stats_export/all_stats_baseline.tsv and
#                      stats_export/all_stats_continuation.tsv
#   "simulate_thesis"  repeat the simulations of the thesis, see THESIS
#   "plot_thesis"      remake the figures of the thesis, see THESIS
#   "test"             check that the whole code works (a few minutes, in a
#                      temporary copy: the data of the project are not touched)

VERBOSE = False
#   False: the terminal shows a summary (one line per phase, one per update
#          of the recommender, a progress line every minute in the long
#          phases, warnings and errors).  Every detail goes anyway to the
#          run.log in the folder of each run.
#   True:  every detail on the terminal too.

# =============================================================================
# PARAMETERS  (used by every mode; these are the values of the thesis,
# main campaign)
# =============================================================================
PARAMS = {
    # g-EPR (Table 3.1)
    "ro": 0.6,                  # prefactor of the exploration probability (default 0.6)
    "gamma": 0.21,              # decay of the exploration probability (default 0.21)
    "a_param": 0.55,            # alpha, tail exponent of the jump lengths (default 0.55)
    "b_param": 0.8,             # beta, tail exponent of the waiting times (default 0.8)
    "spatial_min": 1000,        # minimum jump length [m] (default 1000 = 1 km)
    "spatial_max": 100000,      # maximum jump length [m] (default 100000 = 100 km)
    "temporal_min": 3600,       # minimum waiting time [s] (default 3600 = 1 h)
    "temporal_max": 61200,      # maximum waiting time [s] (default 61200 = 17 h)
    "cell_lenght": 1400,        # side of a cell [m] (default 1400)
    "cell_number": 51,          # cells per side (default 51)
    "n_users": 6000,            # number of individuals (default 6000)
    "starting_position": "core_periphery",
                                # "core_periphery" | "random" | "centered"
                                # (default "core_periphery")
    # baseline (Phase I)
    "baseline_algorithm": "ge", # "ge" grid EPR | "ce" classic EPR (no grid) | "de" d-EPR
                                # (default "ge")
    "t0": 345600,               # end of the baseline [s] (default 345600 = 4 days)
    # continuation (Phase II)
    "t1": 5529600,              # end of the continuation [s] (default 5529600 = 64 days)
    "top_k": 25,                # k_top, length of the recommendation lists (default 25)
    "score_k": 25,              # k, terms in the score: peer group of uk, visited
                                # venues of ik (default 25)
    "cf_updates": 14,           # the recommenders are rebuilt cf_updates + 1 times
                                # (default 14: every 4 days over 60 days)
    "num_latent_factors": 20,   # f, latent factors of MF (default 20)
}

# Continuations are given by the name of the algorithm:
#   "ge" g-EPR (no recommender)   "uk" User-kNN   "ik" Item-kNN   "mf" MF
#   "ra" random recommender
# or by a dictionary with the algorithm and what differs from PARAMS, e.g.
#   {"algorithm": "mf", "num_latent_factors": 5}
#   {"algorithm": "uk", "top_k": 10}
# (allowed keys: t1, top_k, score_k, cf_updates, num_latent_factors, and "label")

# =============================================================================
# SIMULATE: `cycles` new baselines, each continued with every algorithm
# =============================================================================
SIMULATE = {
    "continuations": ["ge", "uk", "ik", "mf"],   # [] = baselines only
    "cycles": 1,
    "seed": None,          # None = random; an int = reproducible runs
}

# =============================================================================
# CONTINUE: baselines already on disk (those of PARAMS), continued again
# =============================================================================
CONTINUE = {
    "baselines": 10,       # N = the N oldest | "all" | ["20261021_120421_680290", ...]
    "continuations": ["uk", "ik", "mf"],
    "seed": None,          # None = random; an int = reproducible runs
}

# =============================================================================
# REANALYSE: redo the analysis from the simulated data
# =============================================================================
REANALYSE = {
    "what": "all",         # "baseline" | "continuation" | "all"
    "metrics": None,       # None = everything (slow); a list of batch metrics =
                           # only those, merged into batch_stats (fast), e.g.
                           # ["collective_entropy_visits", "gini_collective"]
}

# =============================================================================
# PLOT: one series = figures of one dataset; several = overlaid (multiplot)
# =============================================================================
PLOT = {
    "data": "continuation",    # "continuation" | "baseline"
    "series": ["ge", "uk", "ik", "mf"],
    # Each series is a dataset: all the parameters of PARAMS, plus what the
    # series changes.  A series is written in one of two ways:
    #   "uk"                                  short form: only the algorithm,
    #                                         i.e. {"algorithm": "uk"}
    #   {"algorithm": "uk", "top_k": 10}      the algorithm and what differs
    #                                         from PARAMS
    # Keys a series can change: t1, top_k, score_k, cf_updates,
    # num_latent_factors, t0, n_users, starting_position, baseline_algorithm,
    # and "label" (name in the legend; automatic if absent).
    # For baselines the series are EPR models: ["ge"], ["ge", "de"], ...
    # Examples:
    #   ["mf"]
    #       the figures of MF alone (single dataset)
    #   ["ge", "uk", "ik", "mf"]
    #       the four dynamics overlaid 
    #   ["ge", {"algorithm": "uk", "starting_position": "random"}]
    #       g-EPR (core-periphery) and User-kNN with random initial positions
    #   ["ge", {"algorithm": "mf", "num_latent_factors": 5},
    #          {"algorithm": "mf", "num_latent_factors": 30}]
    #       MF with f = 5 and f = 30 against g-EPR (labels mf5, mf30)
    #   with "data": "baseline":
    #   ["ge", {"algorithm": "ge", "t0": 2764800, "label": "ge 32 days"}]
    #       baselines of 4 and of 32 days
    "plots": "all",            # "all" | a list, e.g. ["rg_distribution", "rg_by_ring"]
    # "all" = every figure these data can have (by the marks below), without
    # "batch_stats"; a figure in a list that the data cannot have is skipped
    # with a warning.
    # The figures.  [1] = one series only, [M] = several series only,
    # [C] = continuations only (they need t0 and t1); unmarked = always.
    # The ones not available for the data asked are skipped with a warning.
    #  Individual mobility
    #   "rg_distribution"      distribution of the radius of gyration r_g, and
    #                          r_g against the rank of the individual
    #   "rg_e_dist"            the same for r_g of the explorations only
    #   "rg_over_time"         r_g(t): relative to the final one, in km, and (with
    #                          several series) difference from g-EPR
    #   "two_radius"           r_g of the two most visited places against r_g
    #   "2rad_distrib"         distribution of r_g(2) / r_g
    #   "2rad_distrib_zoomed"  the same, zoomed
    #   "dist_distribution"    distribution of the jump lengths
    #   "s_over_time"          S(t), number of distinct places visited
    #   "num_venues_distribution"  distribution of the number of places visited
    #   "freq_distribution"    [1] visit frequency against the rank of the place
    #   "mobility"             [1] trajectory of one individual
    #  Entropy and predictability of the trajectories
    #   "srand", "sunc", "sreal"        random, uncorrelated, real entropy
    #   "s_triple"                      [1] the three entropies together
    #   "pi_rand", "pi_unc", "pi_real"  the corresponding predictabilities
    #   "pi_triple"                     [1] the three together
    #  Popularity of the places (venues)
    #   "venue_visits"                   visits of each place against its rank
    #   "venue_popularity"               distinct visitors against the rank
    #   "venue_visits_distribution"      distribution of the visits per place n^V
    #   "venue_explorers_distribution"   distribution of the explorers per place n^E
    #  Grids
    #   "aggregate_grid"               [1] visits per cell, all the runs together
    #   "aggregate_grid_exploration"   [1] explorers per cell, all the runs together
    #   "batch_grid", "batch_grid_exploration"  [1] the same, one figure per run
    #   "grid_final_comparison"        [C] final grids of the series side by side
    #                                  (mean over the runs, and one single run)
    #   "aggregate_grid_snapshots"     [1][C] grid at every snapshot (a folder)
    #  Dynamics of the ranking of the places after t0  [C]
    #   "kendall_over_time"        Kendall tau of the top-10 between snapshots
    #   "top10_turnover"           places entering the top-10 per interval
    #   "top10_rank_tracking"      rank over time of the top-10 at t0 and at the end
    #   "top10_membership_panels"  exit of the top-10 of t0, entry of the final one
    #   "rank_origin"              rank at t0 of the place that ends at each rank
    #   "rank_persistence"         top-K retention: share of the top-K kept
    #                              from t0 to the end (and from start to t0)
    #   "venue_velocity_fixed_top"    growth of the top-10 of t0
    #   "venue_velocity_fixed_worst"  growth of the least popular places of t0
    #   "venue_velocity_fixed_final"  growth of the final top-10
    #  Other
    #   "rg_by_ring"       [C] r_g growth and radial shift of the centre of mass,
    #                      by distance of the individual from the centre
    #   "batch_stats"      prints the statistics of every run (no figure)
    #   "metric_vs_k"      [M] statistics against f, k or k_top
    #   "metric_summary"   [M] one statistic per series (collective Gini)
    "output": None,
    # Name of the folder inside plots/ where the figures go.
    #   None    an automatic name: the data and the series, plus a code made
    #           from all the parameters, e.g.
    #           plots/continuation_ge-uk-ik-mf__2e4fe6cc/
    #           Different plots never overwrite each other; the same plot done
    #           again replaces its own figures.
    #   "name"  the figures go in plots/name/ (replacing those with the same
    #           file name already there).
    # Each folder also gets plot_info.txt, with the parameters of every series.
    "first_run_only": False,   # True = one realisation per series (quick look)
}

# =============================================================================
# THESIS: modes "simulate_thesis" and "plot_thesis"
# =============================================================================
THESIS = {
    "campaigns": ["all"],
    #   "long"             main campaign
    #   "f"                MF with f = 3, 5, 10, 15, 30 latent factors
    #   "initialization"   random initial positions
    #   "k_top"            k_top = 5, 10, 15, 50
    #   "k"                k = 5, 10, 15, 50
    #   "baseline_length"  baselines of 32 days
    #   "sensitivity"      every campaign but "long";  "all" = every campaign
    "cycles": None,        # simulate_thesis: reach this many realisations per
                           # campaign (None = those of the thesis: 10, and 5
                           # for k_top and k)
    "check": False,        # True = only print what is on disk and what is
                           # missing, without simulating / plotting
}

# =============================================================================
def _thesis_args(args):
    """Campaign names, --cycles N, --check: from the command line, and from the
    block THESIS for what the command line does not say."""
    names, cycles, check = [], THESIS["cycles"], THESIS["check"]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--check":
            check = True
        elif a == "--cycles":
            cycles = int(args[i + 1])
            i += 1
        elif a.startswith("--cycles="):
            cycles = int(a.split("=", 1)[1])
        else:
            names.append(a)
        i += 1
    return names or list(THESIS["campaigns"]), cycles, check


def main():
    args = sys.argv[1:]
    verbose = VERBOSE or "--verbose" in args
    args = [a for a in args if a != "--verbose"]
    model.utils.setup_logging(verbose)
    mode = args[0] if args else MODE
    model.utils.install_run_log_buffer()
    R = model.runner

    if mode in ("-h", "--help", "help"):
        print(__doc__)
    elif mode == "simulate":
        R.simulate(PARAMS, SIMULATE["continuations"], SIMULATE["cycles"], SIMULATE["seed"])
    elif mode == "continue":
        R.continue_runs(PARAMS, CONTINUE["baselines"], CONTINUE["continuations"],
                        CONTINUE["seed"])
    elif mode == "reanalyse":
        R.reanalyse(REANALYSE["what"], REANALYSE["metrics"])
    elif mode == "plot":
        R.plot(PARAMS, PLOT["data"], PLOT["series"], PLOT["plots"],
               output=PLOT["output"], first_run_only=PLOT["first_run_only"])
    elif mode == "export_stats":
        R.export_stats()
    elif mode == "test":
        ok = model.selftest.run(os.path.dirname(os.path.abspath(__file__)))
        sys.exit(0 if ok else 1)
    elif mode == "simulate_thesis":
        names, cycles, check = _thesis_args(args[1:])
        model.campaigns.simulate_thesis(names, cycles=cycles, check=check)
    elif mode == "plot_thesis":
        names, _cycles, check = _thesis_args(args[1:])
        model.campaigns.plot_thesis(names, check=check)
    else:
        raise SystemExit(f"unknown mode '{mode}': python main.py help")


if __name__ == "__main__":
    main()