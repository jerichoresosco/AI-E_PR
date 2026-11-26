"""
model/selftest.py  --  python main.py test
===========================================
Checks that the whole code works, in a few minutes, WITHOUT touching the data
of the project: the code is copied in a temporary folder and every mode of
main.py is exercised there on a small system (60-80 individuals, 15 x 15
cells, a few days), then the folder is deleted.

Checks:
  1 simulate        2 realisations, g-EPR baseline + ge, uk, ik, mf
  2 seeds           same seed -> identical trajectories
  3 continue        existing baselines continued with ra and MF (f = 3)
  4 reanalyse       full reanalysis of the baselines, fast one (one metric)
                    of the continuations
  5 plot            continuation: several series and one; baseline: one and
                    several; every figure, rg_by_ring included
  6 export_stats    the two tables
  7 simulate_thesis check, 2 realisations of "long", "f" and "k_top" on them,
                    then a second launch that must find nothing to do
  8 plot_thesis     long, f, k_top
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

SMALL = dict(cell_number=15, n_users=60, cf_updates=1, num_latent_factors=5,
             t1=8 * 86400)


def _p():
    import model.campaigns as C
    return dict(C.THESIS_PARAMS, **SMALL)


def _runs(root, part):
    return sum(len([b for b in os.listdir(d) if os.path.isdir(os.path.join(d, b))])
               for d in glob.glob(os.path.join(root, "*")) if part in os.path.basename(d))


def _pdfs(folder):
    return len(glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True)) if folder else 0


# =============================================================================
# The steps (run inside the temporary copy)
# =============================================================================
def step_simulate():
    import model.runner as R
    R.simulate(_p(), ["ge", "uk", "ik", "mf"], cycles=2, seed=1)
    n_b = len(R.list_batches(R.baseline_folder(_p())))
    n_c = _runs(os.path.join(R.ROOT, "combined_analysis_data"), "alg0_ge")
    assert n_b == 2, f"{n_b} baselines instead of 2"
    assert n_c == 8, f"{n_c} analysed continuations instead of 8"
    return "2 baselines, 8 continuations"


def step_seeds():
    import pandas as pd
    import model.runner as R
    P = dict(_p(), n_users=40)
    a, _ = R.run_baseline(P, seed=7)
    b, _ = R.run_baseline(P, seed=7)
    folder = R.baseline_folder(P)
    for f in sorted(glob.glob(os.path.join(folder, a, "user_*_history.parquet"))):
        g = os.path.join(folder, b, os.path.basename(f))
        assert pd.read_parquet(f).equals(pd.read_parquet(g)), f"{os.path.basename(f)} differs"
    return "same seed, same trajectories"


def step_continue():
    import model.runner as R
    R.continue_runs(_p(), 2, ["ra", {"algorithm": "mf", "num_latent_factors": 3}], seed=3)
    root = os.path.join(R.ROOT, "combined_analysis_data")
    assert _runs(root, "alg1_ra") == 2 and _runs(root, "nlf_3_") == 2
    return "2 baselines continued with ra and mf (f=3)"


def step_reanalyse():
    import model.runner as R
    R.reanalyse("baseline")
    R.reanalyse("continuation", metrics=["gini_collective"])
    return "baselines (full) and continuations (gini_collective)"


def step_plot():
    import model.runner as R
    P = _p()
    out = []
    o = R.plot(P, "continuation", ["ge", "uk", "ik", "mf"], output="t_cont_multi")
    assert _pdfs(o) >= 30, f"only {_pdfs(o)} figures"
    assert os.path.isfile(os.path.join(o, "plot_rg_by_ring_ratio.pdf")), "no rg_by_ring"
    out.append(f"continuation x4: {_pdfs(o)}")
    o = R.plot(P, "continuation", ["uk"], output="t_cont_one")
    assert _pdfs(o) >= 15, f"only {_pdfs(o)} figures"
    out.append(f"x1: {_pdfs(o)}")
    o = R.plot(P, "continuation", ["ge", {"algorithm": "mf", "num_latent_factors": 3}],
               output="t_cont_f", first_run_only=True)
    assert _pdfs(o) >= 10
    out.append(f"quick: {_pdfs(o)}")
    o = R.plot(P, "baseline", ["ge"], output="t_base_one")
    assert _pdfs(o) >= 5
    out.append(f"baseline x1: {_pdfs(o)}")
    o = R.plot(P, "baseline", ["ge", {"algorithm": "ge", "n_users": 40, "label": "ge40"}],
               output="t_base_multi")
    assert _pdfs(o) >= 5
    out.append(f"x2: {_pdfs(o)}")
    assert R.plot(P, "continuation", ["ge", {"algorithm": "mf", "num_latent_factors": 7}]) \
        is None, "missing data not detected"
    out.append("missing data detected")
    return "figures: " + ", ".join(out)


def step_export():
    import model.runner as R
    w = R.export_stats()
    for k, path in w.items():
        assert path and os.path.getsize(path) > 0, f"no table {k}"
    return "all_stats_baseline.tsv, all_stats_continuation.tsv"


def _small_campaigns():
    import model.campaigns as C
    import model.runner as R
    for c in C.CAMPAIGNS.values():
        c.P.update(SMALL)
    C.CAMPAIGNS["f"].specs = [R.as_spec({"algorithm": "mf", "num_latent_factors": 3})]
    C.F_VALUES[:] = [3]
    C.CAMPAIGNS["k_top"].specs = [R.as_spec({"algorithm": a, "top_k": 5})
                                  for a in ("uk", "ik", "mf")]
    C.K_VALUES[:] = [5]
    return C


def step_simulate_thesis():
    C = _small_campaigns()
    C.simulate_thesis(["long", "f", "k_top"], cycles=2, check=True)
    C.simulate_thesis(["long", "f", "k_top"], cycles=2)
    for name in ("long", "f", "k_top"):
        p = C.plan(C.CAMPAIGNS[name], 2)
        assert not p, f"{name}: still {len(p)} realisation(s) to do"
    return "long, f, k_top: 2 realisations each, nothing left to do"


def step_plot_thesis():
    import model.runner as R
    C = _small_campaigns()
    C.plot_thesis(["long", "f", "k_top"])
    out = os.path.join(R.ROOT, "thesis_plot")
    n = {s: _pdfs(os.path.join(out, s)) for s in
         ("main_campaign", "sensitivity/f", "sensitivity/k_top")}
    assert n["main_campaign"] >= 20 and n["sensitivity/f"] >= 4 \
        and n["sensitivity/k_top"] >= 20, n
    return ", ".join(f"{k}: {v}" for k, v in n.items())


STEPS = [("simulate", step_simulate), ("seeds", step_seeds), ("continue", step_continue),
         ("reanalyse", step_reanalyse), ("plot", step_plot), ("export_stats", step_export),
         ("simulate_thesis", step_simulate_thesis), ("plot_thesis", step_plot_thesis)]


def inside():
    """Run every step in the current (temporary) project; one line per step."""
    import logging
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)
    import warnings
    warnings.filterwarnings("ignore")
    import model.utils
    model.utils.install_run_log_buffer()
    ok = True
    for name, fn in STEPS:
        t = time.time()
        try:
            msg = fn()
            print(f"@@ OK      {name:16s} {time.time() - t:5.0f} s   {msg}", flush=True)
        except Exception as exc:
            ok = False
            print(f"@@ ERRORE  {name:16s} {time.time() - t:5.0f} s   {exc!r}", flush=True)
            print("".join(traceback.format_exc()), file=sys.stderr, flush=True)
    sys.exit(0 if ok else 1)


def run(project_root):
    """python main.py test: copy the code into a temporary folder, run the
    steps there, print the result, delete the folder."""
    tmp = tempfile.mkdtemp(prefix="selftest_")
    try:
        shutil.copy2(os.path.join(project_root, "main.py"), tmp)
        shutil.copytree(os.path.join(project_root, "model"), os.path.join(tmp, "model"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        print(f"\nTest of the code in a temporary copy ({tmp}); the data of the project "
              f"are not touched.\nIt takes a few minutes.\n", flush=True)
        t = time.time()
        log = os.path.join(tmp, "selftest.log")
        with open(log, "w") as fh:
            proc = subprocess.Popen([sys.executable, "-c",
                                     "import model.selftest as s; s.inside()"],
                                    cwd=tmp, stdout=subprocess.PIPE, stderr=fh, text=True)
            for line in proc.stdout:
                if line.startswith("@@ "):
                    print("  " + line[3:], end="", flush=True)
            proc.wait()
        print(f"\ntotal {(time.time() - t) / 60:.1f} min")
        if proc.returncode == 0:
            print("EVERYTHING OK\n")
        else:
            kept = os.path.join(project_root, "selftest_errors.log")
            shutil.copy2(log, kept)
            print(f"THERE ARE ERRORS: the details are in {os.path.basename(kept)}\n")
        return proc.returncode == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)