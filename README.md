# Effects of Algorithmic Recommendation on Human Mobility

Code of the Master's thesis in Physics *Effects of Algorithmic Recommendation on Human Mobility: a Feedback-Loop Analysis of the EPR Model*.

- **Author:** Jacopo Resasco
- **Supervisors:** Prof. Luca Pappalardo, Prof. Giovanni Mauro
- **Internal supervisor:** Prof. Riccardo Mannella

The code implements **grid-EPR**, a discrete
version of EPR (Exploration and Preferential Return) in which a population moves on a common
grid of venues, and couples it with collaborative-filtering recommenders (User-kNN, Item-kNN,
Matrix Factorisation). With it you can:

- simulate a population with an EPR-type model (grid-EPR, classic EPR, d-EPR), in parallel
  on many cores;
- continue the same population with a recommender that chooses where each individual
  explores and is periodically retrained on the visits it has induced;
- analyse every run, with individual measures (such as radius of gyration, entropies and predictability) and collective ones (such as popularity and ranking of the venues, Gini coefficient and Moran's I);
- compare several dynamics in the same figures, averaged over many realisations;
- reproduce the simulation campaigns and the figures of the thesis, one command each.

Everything is driven by a single file, `main.py`, where every setting is written next to its
default value.

## How a simulation works

Every run has two phases.

1. **Baseline** (from t = 0 to t0): the population moves with an EPR model only
   (`ge` grid-EPR, the default; `ce` classic EPR without a grid; `de` d-EPR).
2. **Continuation** (from t0 to t1): the same individuals continue from where the baseline
   stopped, and the choice of the new venue during exploration is made by one of
   - `ge`: no recommender (the reference),
   - `uk` User-kNN, `ik` Item-kNN, `mf` Matrix Factorisation, `ra` random recommender.

   The recommender is rebuilt `cf_updates + 1` times on the visits made so far.

Several continuations (one per algorithm) start from the same baseline, so the algorithms are
compared on identical histories. A baseline with its continuations is a *realisation*; the
statistics are averaged over realisations (mean ± standard error).

## Main parameters

All set in `PARAMS` in `main.py`; the defaults are the values of the thesis.

| parameter | symbol | meaning | default |
|---|---|---|---|
| `ro`, `gamma` | ρ, γ | exploration probability ρ S^(−γ), S = places visited so far | 0.6, 0.21 |
| `a_param` | α | tail exponent of the jump lengths | 0.55 |
| `b_param` | β | tail exponent of the waiting times | 0.8 |
| `spatial_min`, `spatial_max` | | range of the jump lengths | 1 km – 100 km |
| `temporal_min`, `temporal_max` | | range of the waiting times | 1 h – 17 h |
| `cell_lenght`, `cell_number` | | side of a cell, cells per side | 1400 m, 51 |
| `n_users` | N | number of individuals | 6000 |
| `starting_position` | | initial positions: `core_periphery`, `random`, `centered` | `core_periphery` |
| `baseline_algorithm` | | model of the baseline: `ge` grid-EPR, `ce` classic EPR, `de` d-EPR | `ge` |
| `t0`, `t1` | | end of the baseline, end of the continuation | 4 days, 64 days |
| `top_k` | k_top | length of the recommendation lists | 25 |
| `score_k` | k | terms in the score of User-kNN and Item-kNN | 25 |
| `cf_updates` | | the recommender is rebuilt `cf_updates + 1` times | 14 (every 4 days) |
| `num_latent_factors` | f | latent factors of Matrix Factorisation | 20 |

## Installation

The code needs Python ≥ 3.11 and seven packages: numpy, pandas, scipy, matplotlib, pyarrow,
libpysal and esda.

With conda:

```bash
conda env create -f environment.yml
conda activate aiepr
```

or with pip:

```bash
pip install -r requirements.txt
```

Then check that everything works (a few minutes; it runs in a temporary copy and does not
touch any data):

```bash
python main.py test
```

It must end with `EVERYTHING OK`.

## Usage

Everything goes through `main.py`. Every setting is described next to its value, and
the default values are those of the thesis.

```bash
python main.py                  # run MODE, with the settings written in main.py
python main.py <mode>           # the same, with another mode
python main.py <mode> --verbose # every detail on the terminal
```

| mode | what it does |
|---|---|
| `simulate` | new baselines and their continuations (block `SIMULATE`) |
| `continue` | continue baselines already on disk with more algorithms (block `CONTINUE`) |
| `reanalyse` | redo the analysis of the data on disk (block `REANALYSE`) |
| `plot` | figures of one or more datasets (block `PLOT`: the list of figures is in the comments) |
| `export_stats` | statistics of every dataset in `stats_export/` |
| `simulate_thesis` | repeat the simulation campaigns of the thesis |
| `plot_thesis` | remake the figures of the thesis |
| `test` | check that the code works |

A first small experiment: in `main.py` set for instance `"n_users": 500` and
`"cell_number": 21` in `PARAMS`, then

```bash
python main.py simulate         # 1 baseline + continuations with ge, uk, ik, mf
python main.py plot             # figures in plots/
```

### The thesis

```bash
python main.py simulate_thesis long --check   # what is on disk and what is missing
python main.py simulate_thesis long           # simulate it
python main.py plot_thesis long               # figures in thesis_plot/main_campaign/
```

| campaign | content |
|---|---|
| `long` | main campaign: g-EPR, User-kNN, Item-kNN, MF (10 realisations) |
| `f` | MF with f = 3, 5, 10, 15, 30 latent factors |
| `initialization` | as `long`, with random initial positions |
| `k_top` | length of the recommendation lists k_top = 5, 10, 15, 50 |
| `k` | number of terms in the score k = 5, 10, 15, 50 |
| `baseline_length` | baselines of 32 days instead of 4 |
| `sensitivity` / `all` | every campaign but `long` / every campaign |

`--cycles N` sets the number of realisations to reach. What is already on disk is found and
only what is missing is simulated, so a campaign can be split over several launches and
resumed after an interruption. Every run has its own seed, derived from the campaign, so the
same code reproduces the same data.

The full campaigns were run on a 24-core server (16 worker processes), where
one continuation of 6000 individuals over 60 days takes about 15 minutes, and the data of all the campaigns take
about 270 GB.

## Output

```
raw_data/                 baselines, as simulated        (one folder per parameter set,
analysis_data/            baselines, analysed             with one subfolder per run)
combined_raw_data/        continuations, as simulated
combined_analysis_data/   continuations, analysed
plots/                    figures of the mode "plot"
thesis_plot/              figures of the mode "plot_thesis"
stats_export/             tables of the mode "export_stats"
.cache/                   intermediate results, rebuilt when missing
```

The name of a parameter folder lists the parameters of its runs, e.g.
`ro_0.6_g_0.21_a_0.55_b_0.8_t0_345600_t1_5529600_cl_1400_cn_51_sp_cp_un_6000_tk_25_cfu_14_alg0_ge_alg1_uk`.
Every run folder contains a `run_info.txt` (parameters, seed, main diagnostics) and a `run.log`
with the complete log of the run.

## Code

```
main.py              the only entry point, with all the settings
model/runner.py      simulate / continue / reanalyse / plot / export_stats
model/campaigns.py   the campaigns of the thesis
model/simulation.py  the Simulation class: simulation, analysis, figures
model/user.py        one individual (EPR and recommender-driven decisions)
model/grid.py        the grid of venues
model/utils_cf.py    the recommender systems
model/utils.py       measurements, file names, loading and saving
model/utils_plot.py  the figures
model/selftest.py    the self-test (python main.py test)
```

## Platforms and resources

- **Linux**: the platform the code was developed and run on.
- **macOS**: expected to work (the self-test passes with the process start method of macOS);
  not tested on a real Mac.
- **Windows**: the folder names are long and Windows refuses paths longer than 260
  characters. If a path is too long the code stops at once and explains what to do: put the
  project in a short folder (e.g. `C:\aiepr`), enable the long paths of Windows, or use
  [WSL](https://learn.microsoft.com/windows/wsl/). On Windows the figures are drawn one at a
  time instead of in parallel; WSL avoids both issues.
- **Processes**: all the available cores are used, up to 16. With little memory, limit them
  with the environment variable `AIEPR_MAX_WORKERS`, e.g. `AIEPR_MAX_WORKERS=4 python main.py simulate`.

## License

MIT, see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the thesis:

```
J. Resasco, "Effects of Algorithmic Recommendation on Human Mobility:
a Feedback-Loop Analysis of the EPR Model", Master's thesis in Physics,
University of Pisa, 2026.
```