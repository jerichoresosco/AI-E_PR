import numpy as np
import pandas as pd
import os
import re
import inspect
import math
from libpysal.weights import lat2W
from esda.moran import Moran
import time
import logging
from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------------------------
# Per-run log capture + report
#
# Instead of a separate `check/` tree, every run drops two small files INTO its
# own data folder:
#   run_info.txt : parameters + a digest of the key per-run diagnostics
#   run.log      : the terminal log lines emitted during this run
#
# A single buffering handler is attached to the root logger (once) and simply
# accumulates formatted records in memory; the save methods dump the buffer
# next to the data. One process == one run, so the buffer holds exactly that
# run's log.
# ---------------------------------------------------------------------------
class _RunLogBuffer(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
        self.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S"))

    def emit(self, record):
        try:
            self.records.append(self.format(record))
        except Exception:
            pass


_RUN_LOG_BUFFER = None

# Substrings that mark the "important" per-run diagnostic lines to echo into
# run_info.txt (the full stream still goes to run.log).
_RUN_INFO_MARKERS = (
    "CF interval", "CF data built", "singular values",
    "recommender health", "sim matrix", "Resuming simulation",
    "Analysis data saved", "Combined raw data saved", "Raw data saved",
)


def install_run_log_buffer():
    """Attach the process-wide buffering handler to the root logger (idempotent)."""
    global _RUN_LOG_BUFFER
    if _RUN_LOG_BUFFER is None:
        _RUN_LOG_BUFFER = _RunLogBuffer()
        logging.getLogger().addHandler(_RUN_LOG_BUFFER)
    return _RUN_LOG_BUFFER


def get_run_log_lines():
    """All formatted log lines captured so far this process (empty if uninstalled)."""
    return list(_RUN_LOG_BUFFER.records) if _RUN_LOG_BUFFER is not None else []


def reset_run_log_buffer():
    """
    Clear the captured log lines so the next run starts fresh.

    Multi-run drivers (e.g. the batch sweep) produce one data folder per run in
    a single process; call this at the start of each run so its run.log / run
    diagnostics contain only that run's lines instead of everything since
    process start. No-op if the buffer was never installed.
    """
    if _RUN_LOG_BUFFER is not None:
        _RUN_LOG_BUFFER.records.clear()


def read_run_info_value(folder, key):
    """Value of the first line "key = value" of folder/run_info.txt (as a
    string), or None when the file or the key is missing.  E.g. "seed" (the
    run's own seed) or "load batch_id" (the baseline a continuation continued)."""
    path = os.path.join(folder, "run_info.txt")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        for line in fh:
            k, sep, v = line.partition(" = ")
            if sep and k.strip() == key:
                return v.strip()
    return None


def write_run_report(folder, title, params=None, grid_params=None,
                     extra_lines=None, include_log=True):
    """
    Write run_info.txt (and, when available, run.log) into a run's own data
    folder. Best-effort: never raises into the caller.
    """
    try:
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"# {title}", f"# written: {stamp}", ""]
        if params:
            lines.append("[parameters]")
            for k in sorted(params, key=str):
                lines.append(f"{k} = {params[k]}")
            lines.append("")
        if grid_params:
            lines.append("[grid]")
            for k in sorted(grid_params, key=str):
                lines.append(f"{k} = {grid_params[k]}")
            lines.append("")
        if extra_lines:
            lines.extend(str(x) for x in extra_lines)
            lines.append("")

        log_lines = get_run_log_lines() if include_log else []
        if log_lines:
            digest = [ln for ln in log_lines
                      if any(mark in ln for mark in _RUN_INFO_MARKERS)]
            if digest:
                lines.append("[run diagnostics]")
                lines.extend(digest)
                lines.append("")

        with open(os.path.join(folder, "run_info.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")

        if include_log and log_lines:
            with open(os.path.join(folder, "run.log"), "w") as fh:
                fh.write("\n".join(log_lines) + "\n")
    except Exception as e:
        logger.warning(f"write_run_report failed for {folder}: {e}")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terminal output
#
# Two levels.  The terminal gets a short summary (INFO): one line per phase,
# one line per recommender update, a progress line every PROGRESS_EVERY
# seconds during the long phases, warnings and errors.  The details (DEBUG)
# go to the run.log of every run (see the buffer above), or to the terminal
# too with VERBOSE = True / --verbose in main.py.
# ---------------------------------------------------------------------------
PROGRESS_EVERY = 60          # seconds between two progress lines


class _ConsoleFormatter(logging.Formatter):
    """"HH:MM:SS  message"; the level only for warnings and errors."""

    def format(self, record):
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        t = self.formatTime(record, "%H:%M:%S")
        if record.levelno >= logging.WARNING:
            return f"{t}  {record.levelname}: {msg}"
        return f"{t}  {msg}"


def setup_logging(verbose=False):
    """Terminal output of main.py: the summary (INFO) of this code, or every
    detail (DEBUG) with verbose.  Other libraries: warnings only."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if type(h) is logging.StreamHandler:
            root.removeHandler(h)
    root.setLevel(logging.WARNING)
    console = logging.StreamHandler()
    console.setFormatter(_ConsoleFormatter())
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(console)
    logging.getLogger("model").setLevel(logging.DEBUG)


def fmt_duration(seconds):
    """45 s, 3.2 min, 1.5 h."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _in_main_process():
    import multiprocessing
    return multiprocessing.current_process().name == "MainProcess"


class Progress:
    """Progress of a long loop over `total` items.

    update(done) after each item: a line on the terminal at most every
    PROGRESS_EVERY seconds (none at all if the loop is quicker), with the
    share done and an estimate of the time left; the 10% milestones go to the
    run.log.  tick() prints the same line without new items (used while
    waiting for parallel workers).  In worker processes everything goes to
    the run.log only, so parallel workers do not mix their lines."""

    def __init__(self, what, total, unit=""):
        self.what, self.total, self.unit = what, max(int(total), 1), unit
        self.done = 0
        self.t0 = self.last = time.time()
        self.step = max(1, self.total // 10)
        self.main = _in_main_process()

    def update(self, done):
        self.done = done
        if done % self.step == 0 or done == self.total:
            logger.debug(f"{self.what} progress: {100 * done / self.total:.0f}% "
                         f"({done}/{self.total} {self.unit})")
        self.tick()

    def tick(self):
        now = time.time()
        if not self.main or now - self.last < PROGRESS_EVERY or self.done >= self.total:
            return
        self.last = now
        line = f"  ... {self.what}: {self.done}/{self.total} {self.unit}".rstrip()
        if self.done:
            left = (now - self.t0) * (self.total - self.done) / self.done
            line += (f" ({100 * self.done / self.total:.0f}%, about "
                     f"{fmt_duration(left)} left)")
        else:
            line += f" (running for {fmt_duration(now - self.t0)})"
        logger.info(line)


def completed_with_progress(futures, progress):
    """concurrent.futures.as_completed that keeps progress.tick() going while
    it waits, so a long wait still prints a line every PROGRESS_EVERY s."""
    import concurrent.futures as cf
    pending = set(futures)
    while pending:
        done, pending = cf.wait(pending, timeout=5, return_when=cf.FIRST_COMPLETED)
        for fut in done:
            yield fut
        progress.tick()


# ---------------------------------------------------------------------------
# Path length on Windows
#
# Windows refuses paths longer than 259 characters (folders: 247) unless the
# "long paths" option of the system is on.  The parameter folders are long, so
# every folder name built here is checked on Windows: better a clear message
# at once than a failure in the middle of a simulation.  Nothing happens on
# Linux / macOS.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IS_WINDOWS = os.name == "nt"
_MAX_FILE_PATH, _MAX_DIR_PATH = 259, 247
# The longest pieces that go below a parameter folder.
_LONGEST_DATA_ROOT = "combined_analysis_data"
_SAMPLE_BATCH_ID = "20260101_120000_000000"     # same length as every batch id
_LONGEST_DATA_FILE = "aggregate_grid_exploration.parquet"
_long_paths_cache = []


def _windows_long_paths_enabled():
    """True if the LongPathsEnabled option of Windows is on (read once)."""
    if not _long_paths_cache:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
                _long_paths_cache.append(winreg.QueryValueEx(k, "LongPathsEnabled")[0] == 1)
        except OSError:
            _long_paths_cache.append(False)
    return _long_paths_cache[0]


def check_path_length(path, is_dir=False):
    """On Windows without long paths, stop with an explanation when `path`
    is too long.  No effect elsewhere."""
    if not _IS_WINDOWS:
        return
    limit = _MAX_DIR_PATH if is_dir else _MAX_FILE_PATH
    if len(path) <= limit or _windows_long_paths_enabled():
        return
    raise RuntimeError(
        f"Path too long for Windows ({len(path)} characters, at most {limit}):\n"
        f"  {path}\n"
        f"The project folder ({PROJECT_ROOT}) takes {len(PROJECT_ROOT)} of them. "
        f"Either\n"
        f"  - move the project to a shorter folder (e.g. C:\\aiepr), or\n"
        f"  - turn on the long paths of Windows once (PowerShell as administrator:\n"
        f"    New-ItemProperty -Path HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem "
        f"-Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force), or\n"
        f"  - run the code in WSL (Linux inside Windows).")


def _check_param_folder(name):
    """Check the longest path that will be written below parameter folder
    `name` (see check_path_length)."""
    if not _IS_WINDOWS:
        return
    batch = os.path.join(PROJECT_ROOT, _LONGEST_DATA_ROOT, name, _SAMPLE_BATCH_ID)
    check_path_length(batch, is_dir=True)
    check_path_length(os.path.join(batch, _LONGEST_DATA_FILE))


# Initial positions of the users (starting_position) and their two-letter
# folder token "_sp_XX".
STARTING_POSITIONS = {
    "core_periphery": "cp",   # Gaussian around the centre (the thesis default)
    "random":         "ra",   # uniform over the grid
    "centered":       "ce",   # everybody in the central cell
}
SP_FULL = {v: k for k, v in STARTING_POSITIONS.items()}


def sp_abbr(starting_position):
    """Two-letter folder token of a starting_position ("cp", "ra", "ce")."""
    try:
        return STARTING_POSITIONS[starting_position]
    except KeyError:
        raise ValueError(
            f"unknown starting_position {starting_position!r}: use one of "
            f"{list(STARTING_POSITIONS)}") from None


def build_param_folder_name(filters, grid_params=None):
    """
    Name of the parameter folder of a BASELINE run (raw_data / analysis_data):
        ro_{ro}_g_{gamma}_a_{a}_b_{b}_t_{time}[_cl_{cl}_cn_{cn}]_sp_{sp}
        [_un_{N}][_tk_{K}][_smin_..][_smax_..][_tmin_..][_tmax_..]_alg_{algorithm}
    The grid tokens come from grid_params when given, else from filters
    ("cell_lenght"/"cell_number"), and are omitted when neither has them.
    Single source of truth for the write side (Simulation) and the read side
    (loaders).
    """
    sp  = filters.get("starting_position", "core_periphery")
    alg = filters.get("algorithm", "ce")
    name = (f"ro_{filters['ro']}_g_{filters['gamma']}"
            f"_a_{filters['a_param']}_b_{filters['b_param']}_t_{filters['time']}")
    g = grid_params if grid_params is not None else filters
    if g.get("cell_lenght") is not None and g.get("cell_number") is not None:
        name += f"_cl_{g['cell_lenght']}_cn_{g['cell_number']}"
    name += (f"_sp_{sp_abbr(sp)}{_user_number_suffix(filters)}"
             f"{_top_k_suffix(filters, alg)}{_powerlaw_suffix(filters)}_alg_{alg}")
    _check_param_folder(name)
    return name


# Algorithms whose behaviour actually depends on top_k. Only these emit the
# "_tk_" folder token; EPR and the random recommender ignore top_k, so tagging
# them would pointlessly lengthen (and fragment) their folder names.
_CF_TOPK_ALGORITHMS = ("uk", "ik", "mf")


def _user_number_suffix(filters):
    """
    Parameter-folder token for the population size (number of simulated users).

    Returns "_un_{N}" when a positive population size is available, otherwise ""
    (so legacy folders produced before this token existed keep loading).

    The value is read from "n_users" first (used on the plot/read side, where
    the plain "user_number" key already means a first/last-N *slice* selector),
    falling back to "user_number" (used on the simulate/save/reconstruct side,
    where it means the population size). Both feed the SAME "_un_" token so that
    a folder written from simulate_params matches a name rebuilt from plot
    filters. Inserted before the algorithm token to keep the regex-anchored
    loaders (analyse_all / analyse_all_combined) working.
    """
    if not isinstance(filters, dict):
        return ""
    n = filters.get("n_users", None)
    if n is None:
        n = filters.get("user_number", None)
    try:
        n = int(n)
    except (TypeError, ValueError):
        return ""
    return f"_un_{n}" if n > 0 else ""


def _top_k_suffix(filters, algorithm):
    """
    Parameter-folder token for the CF neighbourhood size top_k.

    Returns "_tk_{K}" only when `algorithm` is a CF variant that consumes top_k
    and a positive value is present; otherwise "" (EPR and random runs).
    Inserted before the algorithm token, like the other optional suffixes.
    """
    if algorithm not in _CF_TOPK_ALGORITHMS:
        return ""
    k = filters.get("top_k", 10) if isinstance(filters, dict) else 10
    return f"_tk_{int(k)}"


# Algorithms whose score is a sum over k terms that can be set apart from the
# list length top_k (resume parameter "score_k", default 25).
_SCORE_K_ALGORITHMS = ("uk", "ik")


def _score_k_suffix(filters, algorithm):
    """
    Parameter-folder token for the number k of terms in the score: "_ks_{K}".

    Emitted only when `filters` (resume_params of a run, or the filters of a
    loader) contains a "score_k" different from the default 25, i.e. only for
    the runs of the sensitivity on k, where k differs from the value
    used everywhere else.  All the other runs, including the sensitivity on the
    list length (k fixed, top_k varied), keep their names unchanged.
    """
    if algorithm not in _SCORE_K_ALGORITHMS or not isinstance(filters, dict):
        return ""
    k = filters.get("score_k")
    if k is None or int(k) == 25:
        return ""
    return f"_ks_{int(k)}"


# Algorithms that actually use Matrix-Factorization latent factors. Only these
# emit the "_nlf_" token (latent factors are only meaningful under mf).
_MF_ALGORITHMS = ("mf",)

# Bounds of the power laws of the jump length [m] and of the waiting time [s]
# (the values of the thesis, as in PARAMS of main.py).  A folder name carries
# a bound only when it differs from these: "_smin_500" appears, "_smin_1000"
# does not.  Every reader of folder names treats a missing token as this value.
POWERLAW_DEFAULTS = {"spatial_min": 1000, "spatial_max": 100000,
                     "temporal_min": 3600, "temporal_max": 61200}
# Parameter key -> folder token, in the order they appear in the name.
_POWERLAW_TOKENS = (
    ("spatial_min",  "smin"),
    ("spatial_max",  "smax"),
    ("temporal_min", "tmin"),
    ("temporal_max", "tmax"),
)

# Default fall-backs for the resume-phase knobs, matching run()/resume().
_CF_UPDATES_DEFAULT = 6
_NUM_LATENT_FACTORS_DEFAULT = 20


def _powerlaw_suffix(filters):
    """
    Power-law bound tokens "_smin_/_smax_/_tmin_/_tmax_", only for the bounds
    that differ from POWERLAW_DEFAULTS (keeps the names short: the paths must
    stay under the 260 characters of Windows).  Values are interpolated as-is
    (like ro/gamma) so ints and floats both round-trip.
    """
    if not isinstance(filters, dict):
        filters = {}
    out = ""
    for key, tok in _POWERLAW_TOKENS:
        v = filters.get(key, POWERLAW_DEFAULTS[key])
        if float(v) != float(POWERLAW_DEFAULTS[key]):
            out += f"_{tok}_{v}"
    return out


def powerlaw_token_defaults():
    """{folder token: default value}, for the readers of folder names."""
    return {tok: POWERLAW_DEFAULTS[key] for key, tok in _POWERLAW_TOKENS}


def _cf_updates_suffix(filters, algorithm):
    """
    cf_updates token (A1): "_cfu_{N}".

    Always emitted for CF algorithms (cf_updates governs the CF rebuild loop);
    never emitted for EPR/random, where it is inert. Uses the default
    when not overridden.
    """
    if algorithm not in _CF_TOPK_ALGORITHMS:
        return ""
    v = filters.get("cf_updates", _CF_UPDATES_DEFAULT) if isinstance(filters, dict) else _CF_UPDATES_DEFAULT
    return f"_cfu_{int(v)}"


def _num_latent_factors_suffix(filters, algorithm):
    """
    num_latent_factors token (A2): "_nlf_{K}".

    Always emitted for MF; never emitted otherwise, where
    latent factors are meaningless. Uses the default when not overridden.
    """
    if algorithm not in _MF_ALGORITHMS:
        return ""
    v = filters.get("num_latent_factors", _NUM_LATENT_FACTORS_DEFAULT) if isinstance(filters, dict) else _NUM_LATENT_FACTORS_DEFAULT
    return f"_nlf_{int(v)}"


# Number of worker threads used to read per-user parquet files in parallel.
# Reading parquet (pyarrow) releases the GIL during decompression, so threads
# give a real speed-up on the I/O-bound load step. Override via the
# AIEPR_LOAD_THREADS environment variable; set to 1 to disable.
LOAD_THREADS = int(os.environ.get("AIEPR_LOAD_THREADS", min(24, (os.cpu_count() or 4))))


# Optional per-file column projection, keyed by file_map key
# (e.g. {"user_history": ["Clock", "Distance"]}).  When a value is a non-empty
# list, only those columns are read from that parquet -- a large memory/IO win
# when a plot needs one column out of a wide user_history.  A value of None (or
# a missing key) means "read every column".  Set by callers (load_analysis_data)
# right before invoking a loader and reset afterwards, mirroring LOAD_THREADS.
_READ_COLUMNS = None


def _read_user_entry(run_path, idx, file_map):
    """Read all parquet files for a single user index into an entry dict.

    Returns the entry dict, or None when the primary file is missing/unreadable.
    Used both by the serial path and the threaded loader below.

    Honors the module-level ``_READ_COLUMNS`` projection map: when it requests a
    subset of columns for a file, only those columns are decoded.  If a requested
    column is absent from the file, the read transparently falls back to the full
    parquet, so projection can never drop data it was not sure about.
    """
    entry = {}
    cols_map = _READ_COLUMNS
    for key, suffix in file_map.items():
        fpath = os.path.join(run_path, f"user_{idx}{suffix}")
        if not os.path.exists(fpath):
            entry[key] = None
            continue
        cols = cols_map.get(key) if cols_map else None
        try:
            if cols:
                try:
                    entry[key] = pd.read_parquet(fpath, columns=list(cols))
                except Exception:
                    # A requested column may not exist in this file -> read all.
                    entry[key] = pd.read_parquet(fpath)
            else:
                entry[key] = pd.read_parquet(fpath)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to read {fpath}: {exc}")
            entry[key] = None
    if entry[next(iter(file_map))] is None:
        return None
    return entry

def power_law_sampling(par, min_val, max_val, n_samples):
    """
    Core function used to calculate samplings from analytical power law distribution p(x) ∝ x^(-1-par),
    as done in the EPR paper "Modelling the scaling properties of human mobility".
    Samples are generated following the Inverse transform sampling theorem. 
    
    Parameters:
    ----------
    par : float
        Power law exponent in p(x) ∝ x^(-1-par)
    min_val : float
        Minimum value. Has to be >0
    max_val : float
        Maximum value.
    n_samples : int
        Number of samples to generate

    Returns:
    --------
    samples : numpy.ndarray
        Array of shape (n_samples,) containing the sampled values from
        power law distribution
    """



    u = np.random.uniform(0, 1, n_samples)

    #Using r values obtained from inversion of the cumulative distribution funtion
    if par != 1:
        r = (u * (min_val**(-par) - max_val**(-par)) + max_val**(-par))**(-1/par)

    #if alpha = 1 we have no more a power law
    else:
        r = u
    
    return r


def spatial_power_law_sampling(par = 0.55, min_val = 1, max_val = 100000, n_samples = 1):    
    """
    Wrapper funtion for power_law_sampling from a spatial distribution

    Parameters:
    ----------
    par : float, optional (default = 0.55)
        Power law exponent (0.55 in the paper for p(r) ∝ r^(-1-par))
    min_val : float, optional (default=5)
        Minimum value. Has to be >0. Chosen 5m arbitrarily as a unit-base for movement
    max_val : float, optional (default=100000)
        Maximum distance. Chosen 100km as in the paper for distance
    n_samples : int, optional (default=1)
        Number of samples to generate

    Returns:
    --------
    samples : numpy.ndarray
        Array of shape (n_samples,) containing the sampled distances from 
        the power law distribution p(r) ∝ r^(-1-par) 
    """
    return power_law_sampling(par = par, min_val = min_val, max_val = max_val, n_samples = n_samples)

def temporal_power_law_sampling( par = 0.8, min_val = 1, max_val = 61200, n_samples = 1):    
    """
    Wrapper funtion for power_law_sampling from a temporal distribution 

    Parameters:
    ----------
    par : float, optional (default = 0.8)
        Power law exponent (0.8 in the paper for p(t) ∝ t^(-1-par))
    min_val : float, optional (default=300)
        Minimum value. Has to be >0. Chosen 300s (5 minutes) arbitrarily as a unit-base for waiting-time
    max_val : float, optional (default=100000)
        Maximum waiting-time. Chosen 61200s (17h) as in the paper
    n_samples : int, optional (default=1)
        Number of samples to generate

    Returns:
    --------
    samples : numpy.ndarray
        Array of shape (n_samples,) containing the sampled waiting-times from 
        the power law distribution p(t) ∝ t^(-1-par)
    """
    return power_law_sampling(par = par, min_val = min_val, max_val = max_val, n_samples = n_samples)

def polar_to_cartesian (delta_r, angle, position=(0,0)):
    """
    Convert polar coordinates to Cartesian coordinates.

    This function transforms a distance and angle in polar coordinates
    to x and y coordinates in Cartesian space starting from a initial position.

    Parameters
    ----------
    delta_r : float
        The radial distance from the origin.
    angle : float
        The angle in radians from the positive x-axis.
    position : tuple of float, optional (default=(0,0))
        The initial position in Cartesian coordinates.

    Returns
    -------
    tuple of float
        A tuple (x, y) representing the Cartesian coordinates.
    """
    x = delta_r*np.cos(angle)
    y = delta_r*np.sin(angle)
    return (x + position[0], y + position[1])


def compute_cm_displacement(user_history_df, t0=None):
    """
    Per-user displacement of the trajectory centre of mass (CM) between the
    baseline (Clock <= t0) and the continuation (Clock > t0) of a run.

    Returns (dx, dy, mag) where
        (dx, dy) = CM_resumed - CM_loaded     (signed vector: the drift)
        mag      = || (dx, dy) ||             (Euclidean magnitude)

    The CM of a phase is the visit-weighted mean position over that phase's
    steps (the mean of the step destinations), consistent with the
    centre-of-mass definition used by compute_stats().

    The trajectory is split at Clock == t0; for a baseline run (t0 None) the
    metric is undefined -> (nan, nan, nan).

    The batch aggregates built from this are: cm_displacement_abs = mean over
    users of `mag` (unsigned typical drift), and cm_displacement_rel_x /
    cm_displacement_rel_y = mean over users of `dx` / `dy` (the signed mean
    drift vector, so opposite drifts cancel and only a net direction survives).
    """
    nan3 = (float("nan"), float("nan"), float("nan"))
    if user_history_df is None or len(user_history_df) == 0:
        return nan3

    xcol, ycol = "Destination_x_coord", "Destination_y_coord"
    if xcol not in user_history_df.columns or ycol not in user_history_df.columns:
        return nan3

    def _cm(df):
        if df is None or len(df) == 0:
            return None
        x = df[xcol].to_numpy(dtype=float)
        y = df[ycol].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if not m.any():
            return None
        return (float(x[m].mean()), float(y[m].mean()))

    if t0 is None:
        return nan3
    cm0 = _cm(user_history_df[user_history_df["Clock"] <= t0])
    cm1 = _cm(user_history_df[user_history_df["Clock"] >  t0])

    if cm0 is None or cm1 is None:
        return nan3
    dx = cm1[0] - cm0[0]
    dy = cm1[1] - cm0[1]
    return (dx, dy, float(np.hypot(dx, dy)))


def compute_stats(user_history_df, venue_history_df):
    """
    Compute trajectory statistics from a user_history DataFrame.

    Computes:
      - x/y centre of mass (full trajectory)
      - x/y centre of mass (top-2 most visited venues only)
      - radius of gyration (full)
      - radius of gyration (top-2 venues)
      - s_rand  : random entropy      log2(N)   [bits]
      - s_unc   : uncorrelated entropy           [bits]
      - s_real  : real entropy (Lempel-Ziv)      [bits]
      - pi_rand : max predictability from s_rand (degenerates to 1/N)
      - pi_unc  : max predictability from s_unc
      - pi_real : max predictability from s_real  (the Pi^max of Def. 6.9)

    Parameters
    ----------
    user_history_df : pd.DataFrame
        User history with columns 'Destination', 'Destination_x_coord',
        'Destination_y_coord'.
    venue_history_df : pd.DataFrame
        Venue visit counts with 'Number of visits' column.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame with columns ['metric', 'value'].
    """
    if len(user_history_df) == 0:
        return pd.DataFrame([
            {"metric": "x_center of mass",      "value": float("nan")},
            {"metric": "y_center of mass",       "value": float("nan")},
            {"metric": "x_center of mass two",   "value": float("nan")},
            {"metric": "y_center of mass two",   "value": float("nan")},
            {"metric": "radius of gyration",     "value": float("nan")},
            {"metric": "radius of gyration two", "value": float("nan")},
            {"metric": "rg_explorative",         "value": float("nan")},
            {"metric": "s_rand",                 "value": float("nan")},
            {"metric": "s_unc",                  "value": float("nan")},
            {"metric": "s_real",                 "value": float("nan")},
            {"metric": "pi_rand",                "value": float("nan")},
            {"metric": "pi_unc",                 "value": float("nan")},
            {"metric": "pi_real",                "value": float("nan")},
        ])

    x_traj = np.concatenate((user_history_df["Departure_x_coord"].values[0:1], user_history_df["Destination_x_coord"].values))
    y_traj = np.concatenate((user_history_df["Departure_y_coord"].values[0:1], user_history_df["Destination_y_coord"].values))

    x_cm = np.mean(x_traj)
    y_cm = np.mean(y_traj)
    r_gyr = np.sqrt(np.mean((x_traj - x_cm)**2 + (y_traj - y_cm)**2))

    # Top-2 most visited destinations
    top2 = user_history_df["Destination"].value_counts().head(2).index.tolist()
    mask = user_history_df["Destination"].isin(top2)
    x_two = np.concatenate(([0], user_history_df.loc[mask, "Destination_x_coord"].values))
    y_two = np.concatenate(([0], user_history_df.loc[mask, "Destination_y_coord"].values))

    if len(x_two) > 1:
        x_cm_two = np.mean(x_two)
        y_cm_two = np.mean(y_two)
        r_gyr_two = np.sqrt(np.mean((x_two - x_cm_two)**2 + (y_two - y_cm_two)**2))
    else:
        x_cm_two, y_cm_two, r_gyr_two = x_cm, y_cm, 0.0

    # --- exploration-only metric ---
    # Recommenders only ever act on exploration steps, so this isolates the
    # exploration "footprint" from the EPR return dynamics.
    rg_explorative = compute_rg_explorative(user_history_df)

    # --- entropy measures (all in bits) ---
    s_rand = compute_random_entropy(venue_history_df)
    s_unc  = compute_uncorrelated_entropy_frequentist(venue_history_df)
    s_real = compute_real_entropy(user_history_df)

    n_loc = len(venue_history_df)
    pi_rand = compute_max_predictability(s_rand, n_loc)
    pi_unc  = compute_max_predictability(s_unc,  n_loc)
    pi_real = compute_max_predictability(s_real, n_loc)

    return pd.DataFrame([
        {"metric": "x_center of mass",      "value": x_cm},
        {"metric": "y_center of mass",       "value": y_cm},
        {"metric": "x_center of mass two",   "value": x_cm_two},
        {"metric": "y_center of mass two",   "value": y_cm_two},
        {"metric": "radius of gyration",     "value": r_gyr},
        {"metric": "radius of gyration two", "value": r_gyr_two},
        {"metric": "rg_explorative",         "value": rg_explorative},
        {"metric": "s_rand",                 "value": s_rand},
        {"metric": "s_unc",                  "value": s_unc},
        {"metric": "s_real",                 "value": s_real},
        {"metric": "pi_rand",                "value": pi_rand},
        {"metric": "pi_unc",                 "value": pi_unc},
        {"metric": "pi_real",                "value": pi_real},
    ])


def compute_rg_explorative(user_history_df):
    """
    Radius of gyration computed over the exploration footprint only.

    Recommenders act exclusively on exploration steps, so this isolates the
    discovery process from the (algorithm-independent) EPR return dynamics.
    It is computed over the rows whose state is "E" -- the steps in which a
    brand-new venue is visited for the first time -- using the initial position
    plus the coordinates of every explored venue, each counted exactly once
    (there are no repeats among exploration destinations).

    This is therefore the *unweighted* radius of gyration of the set of
    discovered locations: the analogue of `radius of gyration`, but with the
    visit-frequency weighting (which the full r_g carries implicitly through
    the repeated rows of the event log) removed.

    Returns
    -------
    float
        rg_explorative.  0.0 for degenerate trajectories (< 2 points).
    """
    if len(user_history_df) == 0:
        return 0.0

    e_mask = user_history_df["State(E/R)"] == "E"
    e_df   = user_history_df[e_mask]

    xe = np.concatenate((user_history_df["Departure_x_coord"].values[0:1],
                         e_df["Destination_x_coord"].values))
    ye = np.concatenate((user_history_df["Departure_y_coord"].values[0:1],
                         e_df["Destination_y_coord"].values))
    if len(xe) > 1:
        xe_cm = np.mean(xe)
        ye_cm = np.mean(ye)
        return float(np.sqrt(np.mean((xe - xe_cm) ** 2 + (ye - ye_cm) ** 2)))
    return 0.0


def compute_rg_over_time(user_history_df):
    """
    Compute the radius of gyration at every time step

    The incremental update uses the identity:
        Var(x_{1..n}) = E[x²] - E[x]²
    maintained via running sums of x, y, x², y² 

    Parameters
    ----------
    user_history_df : pd.DataFrame
        User history with columns 'Clock', 'Destination_x_coord',
        'Destination_y_coord'.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame with columns ['clock', 'rg'].
    """
    clocks = user_history_df["Clock"].values
    x_vals = user_history_df["Destination_x_coord"].values
    y_vals = user_history_df["Destination_y_coord"].values

    n = len(x_vals)
    if n == 0:
        return pd.DataFrame({"clock": [], "rg": []})

    rg_values = np.empty(n)

    # Initialise with the starting point
    sum_x  = user_history_df["Departure_x_coord"].values[0]  
    sum_y  = user_history_df["Departure_y_coord"].values[0]  
    sum_x2 = user_history_df["Departure_x_coord"].values[0] ** 2
    sum_y2 = user_history_df["Departure_y_coord"].values[0] ** 2
    count  = 1          # the origin (0,0) is always part of the trajectory

    for i in range(n):
        sum_x  += x_vals[i]
        sum_y  += y_vals[i]
        sum_x2 += x_vals[i] ** 2
        sum_y2 += y_vals[i] ** 2
        count  += 1

        mean_x = sum_x / count
        mean_y = sum_y / count

        # Var = E[x²] - E[x]²  (includes origin via initial sums = 0)
        var_x = sum_x2 / count - mean_x ** 2
        var_y = sum_y2 / count - mean_y ** 2

        rg_values[i] = np.sqrt(var_x + var_y)
    return pd.DataFrame({"clock": clocks, "rg": rg_values})


def compute_s_over_time(user_history_df):
    """
    Compute S(t) — the cumulative number of distinct venues visited — at each
    exploration event.

    S starts at 2 because the user begins at Venue_1 and the first action is
    always an exploration to Venue_2.

    Parameters
    ----------
    user_history_df : pd.DataFrame
        User history with columns 'Clock' and 'State(E/R)'.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame with columns ['clock', 's'].
    """
    e_rows = user_history_df[user_history_df["State(E/R)"] == "E"]
    clocks   = e_rows["Clock"].values
    s_values = np.arange(2, 2 + len(clocks))
    return pd.DataFrame({"clock": clocks, "s": s_values})


def compute_gini_coefficient(visit_counts):
    """
    Compute the Gini coefficient for a distribution of visit counts:

        G = (1/|V|) * (|V| + 1 - 2 * (∑_{i=1}^{|V|} (|V|+1-i) * x_i) / ∑ x_i)

    where the x_i are sorted in ascending order and only venues with at
    least one visit are considered.

    A value of 0 means perfect equality (all venues visited equally);
    higher values indicate increasing concentration of visits on a smaller
    subset of venues.

    Parameters
    ----------
    visit_counts : array-like of int or float
        Raw visit counts for each venue.  Non-positive entries are ignored.

    Returns
    -------
    float
        Gini coefficient in [0, 1].  Returns 0.0 when fewer than two
        venues have been visited (degenerate case).
    """
    x = np.array(visit_counts, dtype=float)
    x = x[x > 0]           # keep only venues with ≥1 visit
    n = len(x)
    if n <= 1:
        return 0.0          # trivially equal with 0 or 1 venue
    x = np.sort(x)          # ascending order required by Eq. 1
    total = x.sum()
    if total == 0:
        return 0.0
    # ∑_{i=1}^{n} (n+1-i) * x_i   (i is 1-indexed)
    ranks = np.arange(1, n + 1)
    weighted_sum = np.dot((n + 1 - ranks), x)
    return (1.0 / n) * (n + 1 - 2.0 * weighted_sum / total)


def compute_individual_gini(venue_history_df_list):
    """
    Compute the individual Gini coefficient for a list of users.

    Applies :func:`compute_gini_coefficient` to the per-venue visit counts
    recorded in each user's venue history (Equation 1 in Pedreschi et al.,
    restricted to the visits made by each user).

    Parameters
    ----------
    venue_history_df_list : list of pd.DataFrame
        List of DataFrames containing venue history for each user, with at least a 'Number of visits' column.

    Returns
    -------
    float
        Individual Gini coefficient in [0, 1].
    """

    values = [compute_gini_coefficient(df['Number of visits'].values) for df in venue_history_df_list]
    mean = sum(values) / len(values)
    return mean


def compute_collective_gini(venue_history_df_list):
    """
    Compute the collective Gini coefficient over the entire dataset.

    Applies :func:`compute_gini_coefficient` to the aggregate visit counts
    per venue, where each venue's count is the total number of times it was
    visited across *all* users in the dataset (Equation 1 in Pedreschi et al.).

    Parameters
    ----------
    venue_history_df_list : list of pd.DataFrame
        List of DataFrames containing venue history for each user, with at least a 'Number of visits' column.

    Returns
    -------
    float
        Collective Gini coefficient in [0, 1].
    """

    #First compute the aggregate visit counts per venue across all users
    collective_venue_counts = (pd.concat(venue_history_df_list)
                    .groupby("Venue_name")["Number of visits"]
                    .sum()
                    .to_dict())

    visit_counts = list(collective_venue_counts.values())
    return compute_gini_coefficient(visit_counts)


def compute_random_entropy(venue_history_df):
    """
    Compute the random entropy S^rand = log2(N), where N is the number of
    distinct venues visited by the user.

    This is the maximum-entropy case: all N locations equally probable.

    Parameters
    ----------
    venue_history_df : pd.DataFrame
        Venue history with at least a 'Venue_name' column.

    Returns
    -------
    float
        S^rand in bits. Returns 0.0 for degenerate cases (N <= 1).
    """
    N = len(venue_history_df)
    if N <= 1:
        return 0.0
    return float(np.log2(N))

def compute_uncorrelated_entropy_frequentist(venue_history_df):
    """
    Compute the temporal-uncorrelated entropy
        S^unc = -sum_j  p_j * log2(p_j)
    where p_j is the empirical visit frequency of venue j.

    Captures spatial heterogeneity but ignores the temporal order of visits.

    Parameters
    ----------
    venue_history_df : pd.DataFrame
        Venue history with a 'Number of visits' column.

    Returns
    -------
    float
        S^unc in bits. Returns 0.0 for degenerate cases.
    """
    counts = venue_history_df["Number of visits"].values.astype(float)
    counts = counts[counts > 0]
    total = counts.sum()
    if total == 0 or len(counts) <= 1:
        return 0.0
    probs = counts / total
    return float(-np.sum(probs * np.log2(probs)))


class _SuffixAutomaton:
    """
    Online (incremental) suffix automaton over a sequence of arbitrary
    hashable symbols (here: small integer venue codes).

    Standard Blumer/Ukkonen-style online construction, generalized from
    single characters to arbitrary tokens via dict-based transition
    tables instead of fixed-size arrays. Used only by
    compute_real_entropy() below to replace its previous O(n^2)
    pure-Python double loop.

    Supports two operations, both O(1) amortized per symbol:
      - extend(c): append one symbol to the text represented so far.
      - match_length(sequence, start): the length of the longest prefix
        of sequence[start:] that occurs *anywhere* as a substring of the
        text inserted into the automaton so far (without modifying the
        automaton -- a pure lookahead).
    """
    __slots__ = ("length", "link", "next", "last")

    def __init__(self):
        # Parallel lists rather than a list of per-state objects: this
        # keeps the hot loop down to list/dict indexing, no attribute
        # lookups, which matters since extend()/match_length() run once
        # per trajectory step.
        self.length = [0]
        self.link   = [-1]
        self.next   = [{}]
        self.last   = 0

    def extend(self, c):
        length, link, nxt = self.length, self.link, self.next
        cur = len(length)
        length.append(length[self.last] + 1)
        link.append(-1)
        nxt.append({})

        p = self.last
        while p != -1 and c not in nxt[p]:
            nxt[p][c] = cur
            p = link[p]

        if p == -1:
            link[cur] = 0
        else:
            q = nxt[p][c]
            if length[p] + 1 == length[q]:
                link[cur] = q
            else:
                clone = len(length)
                length.append(length[p] + 1)
                link.append(link[q])
                nxt.append(dict(nxt[q]))
                while p != -1 and nxt[p].get(c) == q:
                    nxt[p][c] = clone
                    p = link[p]
                link[q] = clone
                link[cur] = clone
        self.last = cur

    def match_length(self, sequence, start):
        nxt = self.next
        state = 0
        length = 0
        n = len(sequence)
        j = start
        while j < n:
            nxt_state = nxt[state].get(sequence[j])
            if nxt_state is None:
                break
            state = nxt_state
            length += 1
            j += 1
        return length


def compute_real_entropy(user_history_df):
    """
    Estimate the real (actual) entropy S via the Lempel-Ziv compression estimator
    (Kontoyiannis et al., 1998):

        S^est = (1/n * sum_i Lambda_i)^{-1} * log2(n)

    where Lambda_i is the length of the shortest substring starting at position i
    that has not appeared in the preceding positions 1..i-1.

    This estimator converges to the true entropy rate of a stationary process and
    captures the full spatiotemporal structure (order and timing) of the trajectory.

    Parameters
    ----------
    user_history_df : pd.DataFrame
        User history with a 'Destination' column giving the sequence of visited
        venue names (one row per movement step).

    Returns
    -------
    float
        S^est in BITS.  Kontoyiannis et al. (1998) state the estimator with a
        natural logarithm (nats), which is the convention of the mathematical
        information-theory literature.  Song et al. (2010) -- the source of
        the maximum-predictability equation this feeds into -- restate it with
        log2 so that it is directly comparable with S^rand = log2(N) and
        S^unc.  Note that <Lambda> is a pure number (a count of symbols), so
        the entire unit of S^est is carried by the log(n) factor: switching
        base is exactly a multiplication by 1/ln(2) = 1.4427.

        Returns 0.0 for trajectories that are too short (< 2 steps).

    Notes
    -----
    Each Lambda_i is found by walking a suffix automaton built
    incrementally over sequence[0:i] -- this returns exactly the same
    Lambda_i as the textbook double loop (verified against it on random,
    all-distinct, all-identical, repeating and Zipf-distributed inputs),
    just without re-scanning the whole past for every candidate length.

    Complexity is O(n) to build the automaton plus O(sum_i Lambda_i) for
    the lookahead walks. For stationary ergodic sources -- the regime
    this estimator is statistically meant for -- Lambda_i ~ log(i), so
    this comes out to O(n log n) in practice. The only case that still
    degrades towards O(n^2) is a literally degenerate trajectory (e.g.
    a user sitting at a single venue forever, never exploring again);
    that case has S^est dominated by a single component anyway, and a
    handful of such users won't noticeably affect a population's
    wall-clock time the way every user being O(n^2) used to.
    """
    sequence = user_history_df["Destination"].values
    n = len(sequence)
    if n < 2:
        return 0.0

    # Map venue names to small ints once -- automaton transitions then
    # key off plain ints/hashables, which is faster than re-hashing
    # numpy string objects on every comparison.
    codes, _ = pd.factorize(sequence)

    sam = _SuffixAutomaton()
    sam.extend(codes[0])

    lambdas = np.empty(n - 1, dtype=np.int64)
    for idx, i in enumerate(range(1, n)):
        lambdas[idx] = sam.match_length(codes, i) + 1
        sam.extend(codes[i])

    mean_lambda = float(lambdas.mean())
    if mean_lambda == 0:
        return 0.0
    return float((1.0 / mean_lambda) * np.log2(n))  # bits, as in Song et al. (2010)


def _binary_entropy_bits(p):
    """H(p) = -p log2 p - (1-p) log2(1-p), in bits.  H(0) = H(1) = 0."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


def compute_max_predictability(s, n_locations, tol=1e-10, max_iter=60):
    """
    Solve the equation for Pi^max by bisection on [1/N, 1].

    Parameters
    ----------
    s : float
        Entropy of the location sequence, in BITS.  All three of
        compute_random_entropy, compute_uncorrelated_entropy_frequentist and
        compute_real_entropy return bits, so any of them can be fed in.
    n_locations : int
        |L_u|, the number of distinct locations visited by the user.  This is
        exactly the N whose log2 compute_random_entropy takes.
    tol : float
        Absolute tolerance, both on the bracket width and on |g(Pi) - s|.
    max_iter : int
        Iteration cap.  60 bisections shrink a unit bracket to ~1e-18, so the
        tolerance always fires first; this is only a safety net.

    Returns
    -------
    float
        Pi^max in [1/N, 1]: an upper bound on the accuracy of any predictor
        for a source with entropy `s` over `n_locations` symbols, NOT the
        accuracy of a particular one.  Conventions for the degenerate cases:
          N <= 1        -> 1.0   (a user who never moves is perfectly
                                  predictable; H is undefined there)
          s >= log2(N)  -> 1/N   (entropy at or above its random ceiling: the
                                  bound S <= S^rand holds for the true entropy
                                  rate, NOT for the finite-sample Lempel-Ziv
                                  estimator, which can overshoot it.  The
                                  correct limit as S -> S^rand is Pi -> 1/N,
                                  so we clamp rather than raise)
          s <= 0        -> 1.0
          s not finite  -> nan   (propagates cleanly through nanmean)

    Notes
    -----
    Cost: ~40 us per call (pure Python, 60 iterations).  For 100k users x 3
    entropies that is ~12 s single-core, i.e. well under 1% of the time
    compute_real_entropy already spends in the same worker process.  Solving
    once per user is therefore the right call, and it is also the only correct
    one: Pi(S) is strongly non-linear, so <Pi(S)> != Pi(<S>) by Jensen's
    inequality, and converting a *mean* entropy would not answer the question.
    """
    if not np.isfinite(s):
        return float("nan")

    n = int(n_locations)
    if n <= 1:
        return 1.0

    s_bits = float(s)

    s_max = np.log2(n)
    if s_bits >= s_max:
        return 1.0 / n
    if s_bits <= 0.0:
        return 1.0

    # log2(N - 1); for N == 2 this is log2(1) == 0 and the equation collapses to
    # S = H(Pi), whose root on [1/2, 1] is the usual inverse binary entropy.
    c = np.log2(n - 1) if n > 2 else 0.0

    def g_minus_s(p):
        return _binary_entropy_bits(p) + (1.0 - p) * c - s_bits

    lo, hi = 1.0 / n, 1.0     # f(lo) = log2(N) - s >= 0,  f(hi) = -s <= 0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = g_minus_s(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return float(mid)
        if f_mid > 0.0:       # still on the high-entropy side -> move right
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def load_parquet_folder(base_folder, filters, file_map, grid=False, return_run_folders=False, starting_position="core_periphery", user_number=None):
    """
    Generic loader for per-user Parquet datasets stored under a parameter folder.

    This function is the single shared loading implementation used by both
    load_raw_parquet() and load_analysis_parquet(). It scans base_folder for
    parameter sub-folders that match the supplied filters, then loads the files
    described by file_map for each user found.

    Folder layout expected
    ----------------------
    base_folder/
        ro_{ro}_gamma_{gamma}_a_{a}_b_{b}_time_{time}/
            {run_id}/
                user_0_{suffix_1}.parquet
                user_0_{suffix_2}.parquet
                ...

    For grid simulations, the parameter folder is extended with grid dimensions:
        ro_{ro}_gamma_{gamma}_a_{a}_b_{b}_time_{time}_clen_{clen}_cnum_{cnum}/

    Parameters
    ----------
    base_folder : str
        Root directory to scan (e.g. the full path to raw_data/ or analysis_data/).
    filters : dict
        Parameter filters with None values, "user_number" and "grid" already removed.
        Keys should match simulation parameter names
        ("ro", "gamma", "a_param", "b_param", "time").
        For grid simulations, "cell_lenght" and "cell_number" may also be present.
    file_map : dict
        Maps a result-dict key to the per-user filename suffix.
        Example: {"user_history": "_history.parquet", "venue_history": "_venue.parquet"}
        The first key in file_map is used to discover which user indices exist in a run
        folder (i.e. it must be present for every user).
    grid : bool, optional (default=False)
        Whether to match grid simulation folders (containing clen/cnum tokens) or
        non-grid folders. Folders whose grid status does not match this flag are
        silently skipped, preventing cross-contamination between grid and non-grid runs
        with otherwise identical parameters.
    return_run_folders : bool, optional (default=False)
        When False, returns plain user-data dicts (keys from file_map).
        When True, wraps each entry as {"data": <user_dict>, "run_folder": <path>}
        so callers can trace which run folder each user came from.
    user_number : int or None, optional (default=None)
        When positive, loading stops as soon as this many users have been
        accumulated (early-exit — no unnecessary disk reads).
        When negative or None, all matching users are loaded; the final
        slicing for negative values is done by the caller via
        _apply_user_number_slice().

    Returns
    -------
    list of dict
        When return_run_folders is False: one dict per loaded user with keys
        matching file_map; values are DataFrames (or None when a file is absent).
        When return_run_folders is True: one dict per user with keys "data"
        (the user dict) and "run_folder" (str path to the source run folder).

    Notes
    -----
    Run folders are always loaded; there is no longer any analysis lock.
    Run folders are iterated in sorted order so that positive user_number
    always selects the same deterministic subset of users.
    The "grid" key is handled separately via the grid parameter and must not
    appear in filters; callers are responsible for removing it beforehand.
    """
    if grid == True:
        key_map = {
            "ro":          "ro",
            "gamma":       "g",
            "a_param":     "a",
            "b_param":     "b",
            "time":        "t",
            "cell_lenght": "cl",
            "cell_number": "cn",
            "n_users":     "un",
            "top_k":       "tk",
            "spatial_min": "smin",
            "spatial_max": "smax",
            "temporal_min":"tmin",
            "temporal_max":"tmax",
            "algorithm":   "alg",
        }
    elif grid == False:
        key_map = {
            "ro":        "ro",
            "gamma":     "g",
            "a_param":   "a",
            "b_param":   "b",
            "time":      "t",
            "n_users":   "un",
            "top_k":     "tk",
            "spatial_min": "smin",
            "spatial_max": "smax",
            "temporal_min":"tmin",
            "temporal_max":"tmax",
            "algorithm": "alg",
        }

    primary_suffix = next(iter(file_map.values()))

    matching_runs = []

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        parts = folder_name.split("_")
        if len(parts) % 2 != 0:
            continue
        try:
            raw_params = {parts[i]: parts[i + 1] for i in range(0, len(parts), 2)}
        except Exception:
            continue

        converted = {}
        for k, v in raw_params.items():
            if k == "t":
                try:
                    converted[k] = int(v)
                except ValueError:
                    converted[k] = v
            else:
                try:
                    converted[k] = float(v)
                except ValueError:
                    converted[k] = v

        for tok, default in powerlaw_token_defaults().items():
            converted.setdefault(tok, float(default))

        # Scarta cartelle che non corrispondono alla modalità grid richiesta
        folder_has_grid = "cl" in converted and "cn" in converted
        if grid != folder_has_grid:
            continue

        # Scarta cartelle che non corrispondono alla modalità starting_position richiesta
        if grid:
            if converted.get("sp") != sp_abbr(starting_position):
                continue

        if any(converted.get(key_map.get(fk, fk)) != fv
               for fk, fv in filters.items()):
            continue

        for run_folder in sorted(os.listdir(folder_path)):  # sorted for determinism
            run_path = os.path.join(folder_path, run_folder)
            if not os.path.isdir(run_path):
                continue

            user_indices = []
            for fname in os.listdir(run_path):
                if fname.endswith(primary_suffix):
                    token = fname[len("user_") : -len(primary_suffix)]
                    try:
                        user_indices.append((int(token), token))
                    except ValueError:
                        user_indices.append((0, token))

            if user_indices:
                user_indices.sort(key=lambda x: x[0])
                matching_runs.append((run_path, [s for _, s in user_indices]))

    # Early-exit limit: for positive user_number stop as soon as N users are
    # accumulated, avoiding unnecessary disk reads.  For negative user_number
    # (last N) or None (all) we must load everything first.
    early_stop = user_number if (user_number is not None and user_number > 0) else None

    loaded = []

    def _wrap(entry, run_path):
        return {"data": entry, "run_folder": run_path} if return_run_folders else entry

    # ------------------------------------------------------------------
    # Fast path: when we must load every user (no positive early-stop) the
    # per-user reads inside a run are independent, so we dispatch them to a
    # thread pool.  Order within each run is preserved (executor.map keeps
    # input order), which keeps results deterministic.
    # ------------------------------------------------------------------
    if early_stop is None and LOAD_THREADS > 1:
        for run_path, user_indices in matching_runs:
            total_in_run = len(user_indices)
            with ThreadPoolExecutor(max_workers=LOAD_THREADS) as pool:
                entries = pool.map(
                    lambda idx, rp=run_path: _read_user_entry(rp, idx, file_map),
                    user_indices,
                )
                for entry in entries:
                    if entry is not None:
                        loaded.append(_wrap(entry, run_path))
            logger.debug(
                f"Loaded {total_in_run} users [{os.path.basename(run_path)}] "
                f"({LOAD_THREADS} threads)"
            )
        return loaded

    # ------------------------------------------------------------------
    # Serial path: required when a positive user_number early-stop is active,
    # since we must stop as soon as N users are accumulated.
    # ------------------------------------------------------------------
    done = False
    prog = Progress("loading", len(matching_runs), "runs")
    for k, (run_path, user_indices) in enumerate(matching_runs):
        prog.update(k)                  # runs loaded so far (heartbeat)
        if done:
            break
        total_in_run = len(user_indices)
        milestone    = max(1, total_in_run // 10)
        for pos, idx in enumerate(user_indices):
            if early_stop is not None and len(loaded) >= early_stop:
                done = True
                break
            entry = _read_user_entry(run_path, idx, file_map)
            if entry is not None:
                loaded.append(_wrap(entry, run_path))
            if (pos + 1) % milestone == 0 or (pos + 1) == total_in_run:
                pct = ((pos + 1) / total_in_run) * 100
                logger.debug(
                    f"Loading progress: {pct:.0f}% ({pos+1}/{total_in_run} users) "
                    f"[{os.path.basename(run_path)}]"
                )

    return loaded

def load_raw_parquet(filters=None, grid=False, return_run_folders=False):
    """
    Load raw simulation data (user_history, venue_history) from raw_data/.

    Parameters
    ----------
    filters : dict, optional
        Parameter filters. Accepted keys: "ro", "gamma", "a_param", "b_param",
        "time", "user_number", "algorithm", "cell_lenght", "cell_number".
        "user_number": positive = first N users, negative = last N, absent = all.
        "algorithm": one of "ce", "ge", "uk", "ik", "mf". When "ce", no grid
        folders are matched. For all others, grid folders are matched (requires
        "cell_lenght" and "cell_number").
    return_run_folders : bool, optional (default=False)
        When False (default), returns a plain list of dicts with keys
        "user_history" and "venue_history".
        When True, returns a list of dicts with keys "data" (the user dict)
        and "run_folder" (the raw run folder path).

    Returns
    -------
    list of dict
        See return_run_folders for the two possible shapes.
    """
    logger.debug("Loading raw simulation data (raw_data)...")

    if filters is None:
        filters = {}

    user_number = filters.get("user_number", None)

    # Derive grid flag from algorithm; fall back to the explicit grid argument
    alg  = filters.get("algorithm", None)
    grid = (alg != "ce") if alg is not None else grid

    filters_clean = {k: v for k, v in filters.items()
                 if v is not None and k not in ("user_number", "starting_position")}
    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "raw_data")

    if not os.path.exists(base_folder):
        logger.warning(f"raw_data folder not found: {base_folder}")
        return []

    file_map = {
        "user_history":  "_history.parquet",
        "venue_history": "_venue.parquet",
    }

    if grid:
        file_map["grid"] = "_grid.parquet"

    loaded = load_parquet_folder(base_folder, filters_clean, file_map, grid=grid,
                                 return_run_folders=return_run_folders,
                                 starting_position=filters.get("starting_position", "core_periphery"),
                                 user_number=user_number)
    loaded = _apply_user_number_slice(loaded, user_number)

    logger.debug(f"Loaded {len(loaded)} users from raw_data")
    return loaded


def load_analysis_parquet(filters=None, needed_files=None, return_run_folders=False):
    """
    Load analysis data from analysis_data/.

    Parameters
    ----------
    filters : dict, optional
        Parameter filters. Accepted keys: "ro", "gamma", "a_param", "b_param",
        "time", "user_number", "algorithm", "cell_lenght", "cell_number",
        "starting_position".
        "algorithm": one of "ce", "ge", "uk", "ik", "mf". Determines whether
        grid folders are matched.
    needed_files : set of str, optional
        Subset of file-map keys to load (e.g. {"stats", "venue_history"}).
        When None (default), all files are loaded.

    Returns
    -------
    list of dict
        One dict per user with keys from the effective file_map.
        Any value may be None if the corresponding file is absent.
    """
    logger.debug("Loading analysis data (analysis_data)...")

    if filters is None:
        filters = {}

    user_number = filters.get("user_number", None)
    alg         = filters.get("algorithm", "ce")
    grid        = alg != "ce"
    filters_clean = {k: v for k, v in filters.items()
                 if v is not None and k not in ("user_number", "starting_position")}

    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "analysis_data")

    if not os.path.exists(base_folder):
        logger.warning(f"analysis_data folder not found: {base_folder}")
        return []

    full_file_map = {
        "stats":         "_stats.parquet",
        "venue_history": "_venue.parquet",
        "user_history":  "_history.parquet",
    }

    if needed_files is not None:
        file_map = {k: v for k, v in full_file_map.items() if k in needed_files}
        if not file_map:
            # Nothing per-user to load — caller will skip this function anyway,
            # but return an empty list gracefully just in case.
            return []
    else:
        file_map = full_file_map

    if needed_files:
        logger.debug(f"Loading only: {sorted(needed_files)}")

    loaded = load_parquet_folder(base_folder, filters_clean, file_map, grid=grid,
                                 return_run_folders=return_run_folders,
                                 starting_position=filters.get("starting_position", "core_periphery"),
                                 user_number=user_number)
    loaded = _apply_user_number_slice(loaded, user_number)

    logger.debug(f"Analysis data loaded: {len(loaded)} users")
    return loaded


def _apply_user_number_slice(loaded, user_number):
    """
    Slice a list of loaded user entries according to the user_number filter.

    Positive user_number → keep first N.
    Negative user_number → keep last N.
    None or 0            → keep all.

    Note: for positive user_number the early-exit in load_parquet_folder already
    stops loading once N users are accumulated, so this function is typically a
    no-op in that case.  It is still called for safety and to handle the negative
    (last-N) case, which requires the full list.
    """
    if not user_number:   # covers None and 0 → keep all
        return loaded
    if user_number > 0:
        return loaded[:user_number]
    return loaded[user_number:]   # negative → last N 

def load_all_aggregate_grids(filters):
    """
    Load param-level and all batch-level aggregate grids from analysis_data/.

    Parameters
    ----------
    filters : dict
        Simulation parameter filters to identify the param folder.
        Standard keys: "ro", "gamma", "a_param", "b_param", "time".
        Grid keys (when applicable): "cell_lenght", "cell_number".

    Returns
    -------
    param_grid : np.ndarray or None
        Param-level aggregate matrix (sum over all batches).
        None if the file does not exist.
    param_user_count : int
        Total number of users aggregated across all batches.
    batch_grids : list of (str, np.ndarray, int)
        One entry per batch subfolder that contains aggregate_grid.parquet,
        as (batch_id, grid_matrix, user_count) triples, sorted by batch_id.
    """
    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "analysis_data")

    param_folder_name = build_param_folder_name(filters)

    param_folder = os.path.join(base_folder, param_folder_name)
    if not os.path.exists(param_folder):
        logger.warning(f"Param folder not found for aggregate grids: {param_folder}")
        return None, 0, []

    # Param-level aggregate
    param_agg_path = os.path.join(param_folder, "aggregate_grid.parquet")
    param_grid = (
        pd.read_parquet(param_agg_path).values.astype(int)
        if os.path.exists(param_agg_path) else None
    )

    # Batch-level aggregates — count user_*_grid.parquet files for each batch
    batch_grids = []
    param_user_count = 0
    for entry in sorted(os.listdir(param_folder)):
        entry_path = os.path.join(param_folder, entry)
        if not os.path.isdir(entry_path):
            continue
        agg_path = os.path.join(entry_path, "aggregate_grid.parquet")
        if os.path.exists(agg_path):
            mat = pd.read_parquet(agg_path).values.astype(int)
            batch_user_count = sum(
                1 for f in os.listdir(entry_path)
                if f.startswith("user_") and f.endswith("_grid.parquet")
            )
            batch_grids.append((entry, mat, batch_user_count))
            param_user_count += batch_user_count

    logger.debug(
        f"Loaded aggregate grids: 1 param-level ({param_user_count} users), "
        f"{len(batch_grids)} batch-level"
    )
    return param_grid, param_user_count, batch_grids


def load_all_batch_stats(filters):
    """
    Load batch_stats.parquet from every batch subfolder found inside the
    matching parameter folder in analysis_data/.

    Uses the same folder-naming convention as :func:`load_all_aggregate_grids`.

    Parameters
    ----------
    filters : dict
        Simulation parameter filters to identify the param folder.
        Standard keys: "ro", "gamma", "a_param", "b_param", "time".
        Grid keys (when applicable): "cell_lenght", "cell_number".
        Optional: "starting_position" (str), "algorithm" (str).

    Returns
    -------
    batch_stats_list : list of (str, pd.DataFrame)
        One entry per batch subfolder that contains batch_stats.parquet,
        as (batch_id, stats_df) pairs sorted by batch_id.
        Empty list when the param folder does not exist or no file is found.
    """
    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "analysis_data")

    param_folder_name = build_param_folder_name(filters)

    param_folder = os.path.join(base_folder, param_folder_name)
    logger.debug(f"load_all_batch_stats looking for: {param_folder}")
    if not os.path.exists(param_folder):
        # Log all available param folders to help diagnose mismatches
        if os.path.exists(base_folder):
            available = sorted(
                d for d in os.listdir(base_folder)
                if os.path.isdir(os.path.join(base_folder, d))
            )
            logger.warning(
                f"Param folder not found for batch stats: {param_folder_name}\n"
                f"  Available folders in analysis_data/:\n"
                + "\n".join(f"    {d}" for d in available)
            )
        else:
            logger.warning(f"analysis_data/ directory not found: {base_folder}")
        return []

    batch_stats_list = []
    for entry in sorted(os.listdir(param_folder)):
        entry_path = os.path.join(param_folder, entry)
        if not os.path.isdir(entry_path):
            continue
        stats_path = os.path.join(entry_path, "batch_stats.parquet")
        if os.path.exists(stats_path):
            stats_df = pd.read_parquet(stats_path)
            batch_stats_list.append((entry, stats_df))
            logger.debug(f"  Found batch stats: {entry}")
        else:
            logger.debug(f"  Skipping (no batch_stats.parquet): {entry}")

    logger.debug(f"Loaded batch stats for {len(batch_stats_list)} batch(es) from: {param_folder}")
    return batch_stats_list

def _build_combined_param_folder_name(filters, grid_params=None):
    """
    Build the parameter folder name used by _make_combined_run_folder().

    Expected keys in filters: "ro", "gamma", "a_param", "b_param",
    "t0" (time of base run), "t1" (time of resume run),
    "starting_position" (optional), "algorithm_0" (optional, default "ce"),
    "algorithm_1" (optional, default "ce").
    Run knobs are ALWAYS encoded in the name (defaults used when a key is
    absent, so the name matches the value the run actually used): "n_users"
    -> "_un_", "top_k" -> "_tk_" (CF variants only), "cf_updates" -> "_cfu_"
    (CF variants only), "num_latent_factors" -> "_nlf_" (mf only); the
    power-law bounds "_smin_/_smax_/_tmin_/_tmax_" only when they differ from
    POWERLAW_DEFAULTS.

    Mirrors the naming logic in Simulation._make_combined_run_folder().
    """
    ro      = filters["ro"]
    gamma   = filters["gamma"]
    a_param = filters["a_param"]
    b_param = filters["b_param"]
    t0      = filters["t0"]
    t1      = filters["t1"]
    sp      = filters.get("starting_position", "core_periphery")
    alg0    = filters.get("algorithm_0", "ce")
    alg1    = filters.get("algorithm_1", "ce")

    name = (
        f"ro_{ro}_g_{gamma}"
        f"_a_{a_param}_b_{b_param}"
        f"_t0_{t0}_t1_{t1}"
    )
    if grid_params is not None:
        name += f"_cl_{grid_params['cell_lenght']}_cn_{grid_params['cell_number']}"
    elif "cell_lenght" in filters and "cell_number" in filters:
        name += f"_cl_{filters['cell_lenght']}_cn_{filters['cell_number']}"
    name += (
        f"_sp_{sp_abbr(sp)}"
        f"{_user_number_suffix(filters)}{_top_k_suffix(filters, alg1)}"
        f"{_score_k_suffix(filters, alg1)}"
        f"{_cf_updates_suffix(filters, alg1)}{_num_latent_factors_suffix(filters, alg1)}"
        f"{_powerlaw_suffix(filters)}"
        f"_alg0_{alg0}_alg1_{alg1}"
    )
    _check_param_folder(name)
    return name


def load_combined_analysis_parquet(filters=None, needed_files=None, return_run_folders=False):
    """
    Load analysis data from combined_analysis_data/.

    Mirrors load_analysis_parquet() but targets the two-level folder structure
    produced by run_analysis(combined=True).

    Parameters
    ----------
    filters : dict, optional
        Must contain: "ro", "gamma", "a_param", "b_param", "t0", "t1".
        Optional: "starting_position", "algorithm_0", "algorithm_1",
        "user_number", "cell_lenght", "cell_number".
            Selects the subfolder inside each batch timestamp folder.
    needed_files : set of str, optional
        Subset of file-map keys to load.  When None, all files are loaded.

    Returns
    -------
    list of dict
        One dict per user with keys from the effective file_map.
    """
    logger.debug("Loading analysis data (combined_analysis_data)...")

    if filters is None:
        filters = {}

    user_number = filters.get("user_number", None)

    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "combined_analysis_data")

    if not os.path.exists(base_folder):
        logger.warning(f"combined_analysis_data folder not found: {base_folder}")
        return []

    param_folder_name = _build_combined_param_folder_name(filters)
    param_folder = os.path.join(base_folder, param_folder_name)

    if not os.path.exists(param_folder):
        available = sorted(
            d for d in os.listdir(base_folder)
            if os.path.isdir(os.path.join(base_folder, d))
        ) if os.path.exists(base_folder) else []
        logger.warning(
            f"Combined param folder not found: {param_folder_name}\n"
            f"  Available folders in combined_analysis_data/:\n"
            + "\n".join(f"    {d}" for d in available)
        )
        return []

    full_file_map = {
        "stats":         "_stats.parquet",
        "venue_history": "_venue.parquet",
        "user_history":  "_history.parquet",
    }

    if needed_files is not None:
        file_map = {k: v for k, v in full_file_map.items() if k in needed_files}
        if not file_map:
            return []
    else:
        file_map = full_file_map

    # Load from the batch subfolders
    primary_suffix = next(iter(file_map.values()))
    early_stop = user_number if (user_number is not None and user_number > 0) else None

    loaded = []
    done   = False
    batch_ids = sorted(b for b in os.listdir(param_folder)
                       if os.path.isdir(os.path.join(param_folder, b)))
    prog = Progress("loading", len(batch_ids), "runs")
    for k, batch_id in enumerate(batch_ids):
        prog.update(k)                  # runs loaded so far (heartbeat)
        if done:
            break
        batch_path = os.path.join(param_folder, batch_id)

        run_path = batch_path

        # Collect user indices from the primary file
        user_indices = []
        for fname in os.listdir(run_path):
            if fname.endswith(primary_suffix) and fname.startswith("user_"):
                token = fname[len("user_"):-len(primary_suffix)]
                try:
                    user_indices.append((int(token), token))
                except ValueError:
                    user_indices.append((0, token))
        user_indices.sort(key=lambda x: x[0])

        total_in_run = len(user_indices)

        def _wrap(entry):
            return {"data": entry, "run_folder": run_path} if return_run_folders else entry

        # Fast path: no positive early-stop -> read this batch's users in parallel.
        if early_stop is None and LOAD_THREADS > 1:
            idxs = [idx for _, idx in user_indices]
            with ThreadPoolExecutor(max_workers=LOAD_THREADS) as pool:
                for entry in pool.map(
                    lambda i: _read_user_entry(run_path, i, file_map), idxs
                ):
                    if entry is not None:
                        loaded.append(_wrap(entry))
            logger.debug(
                f"Loaded {total_in_run} users [{batch_id}] ({LOAD_THREADS} threads)"
            )
            continue

        # Serial path: needed when a positive user_number early-stop is active.
        milestone = max(1, total_in_run // 10)
        for pos, (_, idx) in enumerate(user_indices):
            if early_stop is not None and len(loaded) >= early_stop:
                done = True
                break
            entry = _read_user_entry(run_path, idx, file_map)
            if entry is not None:
                loaded.append(_wrap(entry))
            if (pos + 1) % milestone == 0 or (pos + 1) == total_in_run:
                pct = ((pos + 1) / total_in_run) * 100
                logger.debug(
                    f"Loading progress: {pct:.0f}% ({pos+1}/{total_in_run} users) [{batch_id}]"
                )

    loaded = _apply_user_number_slice(loaded, user_number)
    logger.debug(f"Combined analysis data loaded: {len(loaded)} users")
    return loaded


def load_combined_all_aggregate_grids(filters):
    """
    Load param-level and batch-level aggregate grids from combined_analysis_data/.

    Mirrors load_all_aggregate_grids() for the combined folder structure.

    Parameters
    ----------
    filters : dict
        Same keys as load_combined_analysis_parquet().

    Returns
    -------
    param_grid : np.ndarray or None
    param_user_count : int
    batch_grids : list of (str, np.ndarray, int)
    """
    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "combined_analysis_data")


    param_folder_name = _build_combined_param_folder_name(filters)
    param_folder = os.path.join(base_folder, param_folder_name)

    if not os.path.exists(param_folder):
        logger.warning(f"Param folder not found for combined aggregate grids: {param_folder}")
        return None, 0, []

    # Param-level aggregate (fall back gracefully if absent)
    param_agg_path = os.path.join(param_folder, "aggregate_grid.parquet")
    param_grid = (
        pd.read_parquet(param_agg_path).values.astype(int)
        if os.path.exists(param_agg_path) else None
    )

    batch_grids = []
    param_user_count = 0
    for entry in sorted(os.listdir(param_folder)):
        batch_path = os.path.join(param_folder, entry)
        if not os.path.isdir(batch_path):
            continue
        run_path = batch_path

        agg_path = os.path.join(run_path, "aggregate_grid.parquet")
        if os.path.exists(agg_path):
            mat = pd.read_parquet(agg_path).values.astype(int)
            batch_user_count = sum(
                1 for f in os.listdir(run_path)
                if f.startswith("user_") and f.endswith("_grid.parquet")
            )
            batch_grids.append((entry, mat, batch_user_count))
            param_user_count += batch_user_count

    logger.debug(
        f"Loaded combined aggregate grids: 1 param-level ({param_user_count} users), "
        f"{len(batch_grids)} batch-level"
    )
    return param_grid, param_user_count, batch_grids


# =============================================================================
# EXPLORATION AGGREGATE GRIDS (distinct users per cell)
# =============================================================================
# These mirror the visit-count aggregate-grid loaders above, but the per-cell
# value is the number of DISTINCT users that ever visited the cell (each user
# counted once) instead of the total visit count.  Feeds the
# batch_grid_exploration / aggregate_grid_exploration plots.

def _exploration_grid_for_run(run_path):
    """
    Return the per-cell distinct-user (exploration) aggregate for a single run
    folder.

    Prefers a cached ``aggregate_grid_exploration.parquet``.  When that file is
    absent (e.g. analysis produced before this plot existed) it is rebuilt on
    the fly from the per-user ``user_*_grid.parquet`` files — binarising each
    user grid to 0/1 before summing — and the result is cached so subsequent
    loads are fast.

    Returns
    -------
    np.ndarray or None
        The distinct-user count matrix, or None when neither the cache nor any
        per-user grid file exists in run_path.
    """
    cache = os.path.join(run_path, "aggregate_grid_exploration.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache).values.astype(int)

    expl_sum = None
    i = 0
    while True:
        fpath = os.path.join(run_path, f"user_{i}_grid.parquet")
        if not os.path.exists(fpath):
            break
        bmat = (pd.read_parquet(fpath).values.astype(int) > 0).astype(int)
        expl_sum = bmat if expl_sum is None else expl_sum + bmat
        i += 1

    if expl_sum is None:
        return None

    df = pd.DataFrame(expl_sum)
    df.columns = df.columns.astype(str)
    try:
        df.to_parquet(cache, index=False)
    except Exception as e:  # caching is best-effort; never fail the load over it
        logger.warning(f"Could not cache exploration aggregate in {run_path}: {e}")
    return expl_sum


def load_all_aggregate_grids_exploration(filters):
    """
    Exploration counterpart of :func:`load_all_aggregate_grids`.

    Returns
    -------
    param_grid : np.ndarray or None
        Param-level distinct-user matrix (sum over all batch exploration grids).
    param_user_count : int
        Total number of users aggregated across all batches.
    batch_grids : list of (str, np.ndarray, int)
        (batch_id, distinct_user_matrix, user_count) per batch, sorted by id.
    """
    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "analysis_data")

    param_folder_name = build_param_folder_name(filters)

    param_folder = os.path.join(base_folder, param_folder_name)
    if not os.path.exists(param_folder):
        logger.warning(f"Param folder not found for exploration grids: {param_folder}")
        return None, 0, []

    batch_grids = []
    param_user_count = 0
    param_sum = None
    for entry in sorted(os.listdir(param_folder)):
        entry_path = os.path.join(param_folder, entry)
        if not os.path.isdir(entry_path):
            continue
        mat = _exploration_grid_for_run(entry_path)
        if mat is None:
            continue
        batch_user_count = sum(
            1 for f in os.listdir(entry_path)
            if f.startswith("user_") and f.endswith("_grid.parquet")
        )
        # If only the cache exists (per-user grids removed) fall back to the max
        # cell value as a lower bound on the user count, else 0.
        if batch_user_count == 0:
            batch_user_count = int(mat.max()) if mat.size else 0
        batch_grids.append((entry, mat, batch_user_count))
        param_user_count += batch_user_count
        param_sum = mat.copy() if param_sum is None else param_sum + mat

    logger.debug(
        f"Loaded exploration grids: {len(batch_grids)} batch-level "
        f"({param_user_count} users total)"
    )
    return param_sum, param_user_count, batch_grids


def load_combined_all_aggregate_grids_exploration(filters):
    """
    Exploration counterpart of :func:`load_combined_all_aggregate_grids`.

    Same return contract as :func:`load_all_aggregate_grids_exploration`, for
    the combined_analysis_data/ folder structure.
    """
    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "combined_analysis_data")

    param_folder_name = _build_combined_param_folder_name(filters)
    param_folder = os.path.join(base_folder, param_folder_name)

    if not os.path.exists(param_folder):
        logger.warning(f"Param folder not found for combined exploration grids: {param_folder}")
        return None, 0, []

    batch_grids = []
    param_user_count = 0
    param_sum = None
    for entry in sorted(os.listdir(param_folder)):
        batch_path = os.path.join(param_folder, entry)
        if not os.path.isdir(batch_path):
            continue
        run_path = batch_path

        mat = _exploration_grid_for_run(run_path)
        if mat is None:
            continue
        batch_user_count = sum(
            1 for f in os.listdir(run_path)
            if f.startswith("user_") and f.endswith("_grid.parquet")
        )
        if batch_user_count == 0:
            batch_user_count = int(mat.max()) if mat.size else 0
        batch_grids.append((entry, mat, batch_user_count))
        param_user_count += batch_user_count
        param_sum = mat.copy() if param_sum is None else param_sum + mat

    logger.debug(
        f"Loaded combined exploration grids: "
        f"{len(batch_grids)} batch-level ({param_user_count} users total)"
    )
    return param_sum, param_user_count, batch_grids


def load_combined_all_batch_stats(filters):
    """
    Load batch_stats.parquet from combined_analysis_data/.

    Mirrors load_all_batch_stats() for the combined folder structure.

    Parameters
    ----------
    filters : dict
        Same keys as load_combined_analysis_parquet().

    Returns
    -------
    batch_stats_list : list of (str, pd.DataFrame)
    """
    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "combined_analysis_data")


    param_folder_name = _build_combined_param_folder_name(filters)
    param_folder = os.path.join(base_folder, param_folder_name)

    logger.debug(f"load_combined_all_batch_stats looking for: {param_folder}")
    if not os.path.exists(param_folder):
        if os.path.exists(base_folder):
            available = sorted(
                d for d in os.listdir(base_folder)
                if os.path.isdir(os.path.join(base_folder, d))
            )
            logger.warning(
                f"Param folder not found for combined batch stats: {param_folder_name}\n"
                f"  Available folders in combined_analysis_data/:\n"
                + "\n".join(f"    {d}" for d in available)
            )
        else:
            logger.warning(f"combined_analysis_data/ directory not found: {base_folder}")
        return []

    batch_stats_list = []
    for entry in sorted(os.listdir(param_folder)):
        batch_path = os.path.join(param_folder, entry)
        if not os.path.isdir(batch_path):
            continue
        run_path = batch_path

        stats_path = os.path.join(run_path, "batch_stats.parquet")
        if os.path.exists(stats_path):
            stats_df = pd.read_parquet(stats_path)
            batch_stats_list.append((entry, stats_df))
            logger.debug(f"  Found combined batch stats: {entry}")
        else:
            logger.debug(f"  Skipping (no batch_stats.parquet): {entry}")

    logger.debug(f"Loaded combined batch stats for {len(batch_stats_list)} batch(es) from: {param_folder}")
    return batch_stats_list


def load_combined_popularity_snapshots(filters):
    """Load popularity_snapshots.parquet (per-cell init/at_t0/final visit counts)
    from combined_analysis_data, ONE DataFrame PER BATCH (no summing across
    batches: persistence is a within-realization quantity).
    Returns a list of (batch_id, DataFrame[cell, init, at_t0, final]); [] if none."""
    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "combined_analysis_data")
    param_folder_name = _build_combined_param_folder_name(filters)
    param_folder = os.path.join(base_folder, param_folder_name)

    logger.debug(f"load_combined_popularity_snapshots looking for: {param_folder}")
    if not os.path.exists(param_folder):
        logger.warning(f"Param folder not found for popularity snapshots: {param_folder_name}")
        return []

    out = []
    for entry in sorted(os.listdir(param_folder)):
        batch_path = os.path.join(param_folder, entry)
        if not os.path.isdir(batch_path):
            continue
        run_path = batch_path
        snap_path = os.path.join(run_path, "popularity_snapshots.parquet")
        if os.path.exists(snap_path):
            out.append((entry, pd.read_parquet(snap_path)))
            logger.debug(f"  Found popularity snapshots: {entry}")

    if not out:
        logger.warning(f"No popularity_snapshots.parquet under: {param_folder}")
    else:
        logger.debug(f"Loaded popularity snapshots for {len(out)} batch(es).")
    return out


def load_combined_popularity_timeseries(filters):
    """Load popularity_timeseries.parquet (per-cell CUMULATIVE visit counts
    sampled every POP_SNAPSHOT_STEP clock units) from combined_analysis_data,
    ONE DataFrame PER BATCH (no summing across batches: these feed within-
    realization, time-resolved metrics that are aggregated per run and then
    averaged with an error band).

    Returns a list of (batch_id, DataFrame) where each DataFrame has a 'cell'
    column plus one 't_{clock}' column per snapshot time; [] if none found.
    Mirrors load_combined_popularity_snapshots()."""
    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "combined_analysis_data")
    param_folder_name = _build_combined_param_folder_name(filters)
    param_folder = os.path.join(base_folder, param_folder_name)

    logger.debug(f"load_combined_popularity_timeseries looking for: {param_folder}")
    if not os.path.exists(param_folder):
        logger.warning(f"Param folder not found for popularity time series: {param_folder_name}")
        return []

    out = []
    for entry in sorted(os.listdir(param_folder)):
        batch_path = os.path.join(param_folder, entry)
        if not os.path.isdir(batch_path):
            continue
        run_path = batch_path
        ts_path = os.path.join(run_path, "popularity_timeseries.parquet")
        if os.path.exists(ts_path):
            out.append((entry, pd.read_parquet(ts_path)))
            logger.debug(f"  Found popularity time series: {entry}")

    if not out:
        logger.warning(f"No popularity_timeseries.parquet under: {param_folder}")
    else:
        logger.debug(f"Loaded popularity time series for {len(out)} batch(es).")
    return out


def build_popularity_timeseries(histories, t_snaps=None, step=None):
    """Per-cell CUMULATIVE visit and exploration counts at fixed snapshot times.

    Single source of truth for the two time-series artifacts written next to
    each combined batch:

        popularity_timeseries.parquet   visits     (every arrival counts)
        exploration_timeseries.parquet  explorers  (first arrival of each user
                                                    only = distinct users)

    Counting convention (unchanged from the historical visits series):
      * each user's START cell (Departure of the earliest row) is counted once,
        in the first snapshot bucket, for both visits and explorers -- the grid
        pre-marks the initial cell as visited;
      * every Destination with Clock <= t_j contributes one visit at t_j;
      * a cell enters a user's explored set at its FIRST arrival (or at t_1 if
        it is the start cell), so exploration[c, t_j] = number of distinct
        users that had reached c by t_j.
    Jumps whose clock lies beyond the last snapshot are ignored.

    Parameters
    ----------
    histories : list of DataFrame
        Per-user histories with columns 'Clock', 'Destination', 'Departure'.
    t_snaps : array-like, optional
        Explicit snapshot times (ascending). When given, `step` is ignored --
        to reproduce the exact columns of an existing
        popularity_timeseries.parquet.
    step : float, optional
        Sampling step: snapshots are step, 2*step, ... up to the last multiple
        <= the max observed clock.

    Returns
    -------
    (visits_df, explorers_df) : DataFrames with a 'cell' column plus one
        't_{clock}' column per snapshot, or (None, None) when nothing usable.
    """
    hist = [h for h in histories if h is not None and len(h) > 0]
    if not hist:
        return None, None

    if t_snaps is None:
        if not step or step <= 0:
            return None, None
        max_clock = 0.0
        for uh in hist:
            c = np.asarray(uh["Clock"].to_numpy(), dtype=float)
            c = c[np.isfinite(c)]
            if c.size:
                max_clock = max(max_clock, float(c.max()))
        # ceil, not floor: resume() drops each user's last jump beyond t1, so
        # max_clock < t1 and floor() would lose the final snapshot at t1.
        # The last snapshot (>= max_clock) holds the whole trajectory.
        n_snaps = int(np.ceil(max_clock / float(step)))
        if n_snaps < 1:
            return None, None
        t_snaps = np.arange(1, n_snaps + 1, dtype=float) * float(step)
    t_snaps = np.asarray(t_snaps, dtype=float)
    n_snaps = t_snaps.size
    if n_snaps < 1:
        return None, None

    vis_cells, vis_bins = [], []
    exp_cells, exp_bins = [], []
    for uh in hist:
        clk = np.asarray(uh["Clock"].to_numpy(), dtype=float)
        dst = uh["Destination"].to_numpy()
        dep = uh["Departure"].to_numpy()
        start = dep[int(np.nanargmin(clk))]

        ok = np.isfinite(clk) & ~pd.isna(dst)
        b = np.searchsorted(t_snaps, clk[ok], side="left")
        keep = b < n_snaps
        d_ok = dst[ok][keep].astype(str)
        b_ok = b[keep]

        # visits: start cell (bucket 0) + every arrival
        vis_cells.append(np.concatenate(([str(start)], d_ok)))
        vis_bins.append(np.concatenate(([0], b_ok)))

        # explorers: first arrival per cell (start cell first, bucket 0).
        # Buckets are non-decreasing in row order, so the first occurrence
        # of a cell carries its earliest bucket.
        all_c = np.concatenate(([str(start)], d_ok))
        all_b = np.concatenate(([0], b_ok))
        _, first_idx = np.unique(all_c, return_index=True)
        exp_cells.append(all_c[first_idx])
        exp_bins.append(all_b[first_idx])

    vc = np.concatenate(vis_cells); vb = np.concatenate(vis_bins)
    ec = np.concatenate(exp_cells); eb = np.concatenate(exp_bins)
    cells = sorted(set(vc.tolist()) | set(ec.tolist()), key=str)
    code = {c: i for i, c in enumerate(cells)}

    def _cum(cell_arr, bin_arr):
        inc = np.zeros((len(cells), n_snaps), dtype=np.int64)
        rows = np.fromiter((code[c] for c in cell_arr), dtype=np.int64,
                           count=len(cell_arr))
        np.add.at(inc, (rows, bin_arr.astype(np.int64)), 1)
        return np.cumsum(inc, axis=1)

    def _frame(cum):
        data = {"cell": cells}
        for j, t in enumerate(t_snaps):
            data[f"t_{int(t)}"] = cum[:, j]
        return pd.DataFrame(data)

    return _frame(_cum(vc, vb)), _frame(_cum(ec, eb))


def _load_combined_batch_artifact(filters, filename, what):
    """Load one per-batch artifact from combined_analysis_data, ONE DataFrame
    PER BATCH.
    Returns a list of (batch_id, DataFrame); [] if none found."""
    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "combined_analysis_data")
    param_folder_name = _build_combined_param_folder_name(filters)
    param_folder = os.path.join(base_folder, param_folder_name)

    logger.debug(f"load {what} looking for: {param_folder}")
    if not os.path.exists(param_folder):
        logger.warning(f"Param folder not found for {what}: {param_folder_name}")
        return []

    out = []
    for entry in sorted(os.listdir(param_folder)):
        batch_path = os.path.join(param_folder, entry)
        if not os.path.isdir(batch_path):
            continue
        run_path = batch_path
        fpath = os.path.join(run_path, filename)
        if os.path.exists(fpath):
            out.append((entry, pd.read_parquet(fpath)))

    if not out:
        logger.warning(f"No {filename} under: {param_folder}")
    else:
        logger.debug(f"Loaded {what} for {len(out)} batch(es).")
    return out


def load_combined_exploration_timeseries(filters):
    """Like load_combined_popularity_timeseries, but for the per-cell
    CUMULATIVE number of distinct explorers (exploration_timeseries.parquet)."""
    return _load_combined_batch_artifact(
        filters, "exploration_timeseries.parquet", "exploration time series")


def compute_moran_i(batch_agg):
    """
    Compute Moran's I spatial autocorrelation statistic on a batch aggregate
    grid, using esda (PySAL) with a rook-contiguity spatial weights matrix.
 
    Parameters
    ----------
    batch_agg : pd.DataFrame
        Batch-level aggregate grid (aggregate_grid.parquet of a batch).
        Columns are expected to be strings (as saved to Parquet); they are
        converted internally to a numeric numpy array before computation.
        The grid must be square (n x n).
 
    Returns
    -------
    dict with keys:
        "I"       : float - Moran's I statistic.
                            > E_I  → spatial clustering of similar visit counts.
                            < E_I  → spatial dispersion (checkerboard pattern).
                            ≈ E_I  → spatial randomness.
        "E_I"     : float - Expected value under complete spatial randomness:
                            -1 / (N - 1), where N is the number of cells.
        "z_score" : float - Standardised z-score under normality assumption.
        "p_value" : float - Two-tailed p-value for the null hypothesis of
                            complete spatial randomness.
 
    Raises
    ------
    ValueError
        If the grid is not square, or if all cells share the same visit count
        (variance zero, Moran's I undefined).
 
    Notes
    -----
    Spatial weights: binary rook contiguity (4-neighbours: N, S, E, W),
    row-standardised so each row sums to 1.  Built via libpysal.weights.lat2W.
 
    The z-score and p-value are computed under the normality assumption
    (esda.moran.Moran attributes z_norm / p_norm).
    """
    grid_matrix = batch_agg.values.astype(float)
 
    if grid_matrix.ndim != 2 or grid_matrix.shape[0] != grid_matrix.shape[1]:
        raise ValueError(
            f"batch_agg must be a square 2-D grid, got shape {grid_matrix.shape}"
        )
 
    if np.var(grid_matrix) == 0.0:
        raise ValueError(
            "All cells have the same visit count: Moran's I is undefined."
        )
 
    n = grid_matrix.shape[0]
    W = lat2W(n, n, rook=True)
    W.transform = "r" #normalise rows to sum to 1
 
    mi = Moran(grid_matrix.flatten(), W)
 
    return {
        "I":       mi.I,
        "E_I":     mi.EI,
        "z_score": mi.z_norm,
        "p_value": mi.p_norm,
    }


# ==========================================================================
# BULK EXPORT OF BATCH STATISTICS TO CSV
# ==========================================================================
#
# Implements the "export all stats" feature: walk EVERY parameter folder under
# analysis_data/ and combined_analysis_data/, aggregate the per-batch
# batch_stats.parquet files belonging to the same parameter combination, and
# write two flat TAB-separated matrices:
#
#     rows    = one parameter combination (aggregated over its batches)
#     columns = the simulation parameters, then one triple
#               (_mean, _std, _sem) per metric
#
# Two separate matrices are produced because the two data sources carry
# different parameter columns: plain analysis runs are described by t/alg,
# while continuations are described by t0/t1/alg0/alg1.
#
# STATISTICAL CONVENTIONS
# -----------------------
# Aggregation happens ACROSS BATCHES (i.e. across independent simulations);
# each batch_stats.parquet already holds a value averaged over that batch's
# users.  For every metric we report:
#
#   _mean : arithmetic mean of the per-batch values.
#   _std  : SAMPLE standard deviation (ddof=1, Bessel-corrected).  Measures how
#           much the metric varies from simulation to simulation.  NaN when
#           fewer than two batches are available (undefined).
#   _sem  : standard error of the mean, std / sqrt(n).  Measures how precise
#           the estimate of the mean is; this is the quantity carrying the
#           sqrt(N) factor.  NaN when fewer than two batches are available.
#
# ddof=1 matches the convention already used by the plot error bands in
# utils_plot._compute_band() and by utils_plot.print_batch_stats().


# --------------------------------------------------------------------------
# Folder-name parsing
# --------------------------------------------------------------------------
#
# Parameter folder names are built by Simulation._make_run_folder() and
# Simulation._make_combined_run_folder() as a flat sequence of "key_value"
# pairs, e.g.
#
#   ro_0.6_g_0.21_a_0.55_b_0.8_t_345600_cl_1400_cn_51_sp_cp_un_6000_alg_ge
#
# so a pairwise scan recovers every token without any brittle regex.  Optional
# tokens (un / tk / ks / cfu / nlf) are absent where they do not apply and
# come back as None; absent power-law bounds come back as POWERLAW_DEFAULTS.

# Folder token -> output column name.  Order defines the CSV column order.
_ANALYSIS_PARAM_COLUMNS = [
    ("ro",   "ro"),
    ("g",    "gamma"),
    ("a",    "a_param"),
    ("b",    "b_param"),
    ("t",    "time"),
    ("cl",   "cell_lenght"),
    ("cn",   "cell_number"),
    ("sp",   "starting_position"),
    ("un",   "n_users"),
    ("tk",   "top_k"),
    ("smin", "spatial_min"),
    ("smax", "spatial_max"),
    ("tmin", "temporal_min"),
    ("tmax", "temporal_max"),
    ("alg",  "algorithm"),
]

_COMBINED_PARAM_COLUMNS = [
    ("ro",   "ro"),
    ("g",    "gamma"),
    ("a",    "a_param"),
    ("b",    "b_param"),
    ("t0",   "t0"),
    ("t1",   "t1"),
    ("cl",   "cell_lenght"),
    ("cn",   "cell_number"),
    ("sp",   "starting_position"),
    ("un",   "n_users"),
    ("tk",   "top_k"),
    ("ks",   "score_k"),
    ("cfu",  "cf_updates"),
    ("nlf",  "num_latent_factors"),
    ("smin", "spatial_min"),
    ("smax", "spatial_max"),
    ("tmin", "temporal_min"),
    ("tmax", "temporal_max"),
    ("alg0", "algorithm_0"),
    ("alg1", "algorithm_1"),
]

# Tokens whose value is a string, not a number.
_STRING_TOKENS = {"sp", "alg", "alg0", "alg1"}

# Reverse map for the starting-position abbreviation used in folder names.
_SP_FULL = SP_FULL

# Preferred metric ordering, mirroring _compute_and_save_batch_stats().
# Metrics not listed here (future additions) are appended alphabetically.
_METRIC_ORDER = [
    "n_users",
    "gini_collective",
    "gini_individual",
    "mean_rg",
    "mean_2rg",
    "mean_2rg_ratio",
    "mean_rg_explorative",
    "mean_travel_distance",
    "mean_num_venues",
    "mean_s_rand",
    "mean_s_unc",
    "mean_s_real",
    "mean_pi_rand",
    "mean_pi_unc",
    "mean_pi_real",
    "cm_displacement_abs",
    "cm_displacement_rel_x",
    "cm_displacement_rel_y",
    "collective_entropy_visits",
    "collective_entropy_exploration",
    "I",
    "E_I",
    "z_score",
    "p_value",
]


# --------------------------------------------------------------------------
# Significant-figures formatting
# --------------------------------------------------------------------------
#
# Metrology convention: the uncertainty fixes how many digits of the estimate
# are meaningful.  We round the FINEST uncertainty (the sem) to SIG_FIGS
# significant figures, read off the decimal place that implies, and then quote
# mean, std and sem all at that same decimal place ("in maniera concorde"), so
# the three numbers of a triple line up digit by digit.
#
#   mean = 234.4542345542, std = 0.3423, sem = 0.00345
#     -> sem to 2 s.f. = 0.0034  -> 4 decimals
#     -> "234.4542"  "0.3423"  "0.0034"
#
# Fallbacks: when sem is unusable (NaN with a single batch, or exactly zero)
# the std takes over; when both are unusable we fall back to _DEFAULT_DECIMALS.
# An uncertainty many orders of magnitude below the mean is floating-point
# noise (e.g. Moran's I identical across batches gives sem ~1e-17) and would
# otherwise demand 15 decimals, so it is treated as unusable too.

SIG_FIGS = 2            # significant figures kept on the uncertainty
_DEFAULT_DECIMALS = 4   # used when no usable uncertainty is available
_MAX_DECIMALS = 15      # beyond float64 resolution
_NOISE_REL_TOL = 1e-12  # errors below this fraction of the mean are noise


def _usable_error(err, reference=None):
    """True when *err* can meaningfully set a decimal place."""
    if err is None or not np.isfinite(err) or err <= 0:
        return False
    if reference is not None and np.isfinite(reference):
        if abs(err) < _NOISE_REL_TOL * max(abs(reference), 1.0):
            return False
    return True


def significant_decimals(sem, std=None, mean=None, sig=SIG_FIGS,
                         default=_DEFAULT_DECIMALS):
    """
    Decimal places implied by rounding the uncertainty to *sig* significant
    figures.  Prefers `sem`, falls back to `std`, then to *default*.
    """
    err = sem if _usable_error(sem, mean) else (std if _usable_error(std, mean) else None)
    if err is None:
        # Dispersion exactly zero across batches (e.g. n_users, identical in
        # every run): an integer estimate deserves no decimals at all.
        exact_zero = any(e is not None and np.isfinite(e) and e == 0 for e in (sem, std))
        if (exact_zero and mean is not None and np.isfinite(mean)
                and float(mean).is_integer()):
            return 0
        return default
    decimals = sig - 1 - math.floor(math.log10(abs(err)))
    return int(min(max(decimals, 0), _MAX_DECIMALS))


def format_to_decimals(value, decimals):
    """Fixed-point string for *value*; empty string for None/NaN/inf."""
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    return f"{v:.{decimals}f}"


# --------------------------------------------------------------------------
# Fixed-width table rendering
# --------------------------------------------------------------------------

def render_aligned_table(headers, rows, aligns=None, gap=2):
    """
    Render a table as fixed-width monospace lines.

    Tabs cannot align these tables: tab stops land every 8 columns, while metric
    names ("gini_collective") and batch ids ("20240601_120000_000000") overrun
    them, so every row shifts.  Padding each column to the width of its widest
    cell is the only layout that actually lines up.

    Parameters
    ----------
    headers : sequence of str
    rows : sequence of sequences
        Cells are str()-ed; rows shorter than *headers* are padded with "".
    aligns : sequence of str, optional
        Per-column format alignment ("<", ">", "^").  Defaults to left for the
        first column (labels) and right for the rest (numbers).
    gap : int
        Spaces between columns.

    Returns
    -------
    list of str
        Header line, rule line, then one line per row.  No trailing spaces.
    """
    headers = [str(h) for h in headers]
    ncol = len(headers)
    norm = [[str(r[c]) if c < len(r) else "" for c in range(ncol)] for r in rows]

    if aligns is None:
        aligns = ["<"] + [">"] * (ncol - 1)

    widths = [len(headers[c]) for c in range(ncol)]
    for row in norm:
        for c in range(ncol):
            widths[c] = max(widths[c], len(row[c]))

    pad = " " * gap

    def _line(cells):
        return pad.join(f"{cells[c]:{aligns[c]}{widths[c]}}" for c in range(ncol)).rstrip()

    out = [_line(headers), pad.join("-" * widths[c] for c in range(ncol))]
    out.extend(_line(row) for row in norm)
    return out


def _coerce_token(value):
    """Turn a folder token value into int / float / str, in that preference."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_param_folder_name(name, combined=False):
    """
    Parse a parameter-folder name into a dict of simulation parameters.

    The name is a flat sequence of "key_value" pairs, so it is scanned
    pairwise.  Unknown keys are ignored (forward-compatible); known keys absent
    from the name come back as None (backward-compatible with folders written
    before a token existed).

    Parameters
    ----------
    name : str
        Folder name, e.g. "ro_0.1_g_0.2_..._alg_ge".
    combined : bool
        Select the combined (t0/t1/alg0/alg1) token schema.

    Returns
    -------
    dict
        Mapping column-name -> value, with every expected column present.
    """
    schema = _COMBINED_PARAM_COLUMNS if combined else _ANALYSIS_PARAM_COLUMNS
    known = {tok for tok, _ in schema}

    parts = name.split("_")
    raw = {}
    i = 0
    while i + 1 < len(parts):
        key, val = parts[i], parts[i + 1]
        if key in known:
            raw[key] = val if key in _STRING_TOKENS else _coerce_token(val)
        i += 2
    for tok, default in powerlaw_token_defaults().items():
        raw.setdefault(tok, default)

    out = {}
    for tok, col in schema:
        v = raw.get(tok, None)
        if tok == "sp" and v is not None:
            v = _SP_FULL.get(v, v)
        out[col] = v
    return out


def _order_metrics(metric_names):
    """Sort metrics by the preferred order, appending unknown ones alphabetically."""
    known = [m for m in _METRIC_ORDER if m in metric_names]
    extra = sorted(m for m in metric_names if m not in _METRIC_ORDER)
    return known + extra


def aggregate_batch_stats(batch_stats_list):
    """
    Aggregate a list of per-batch stats DataFrames into mean / std / sem.

    Parameters
    ----------
    batch_stats_list : list of (str, pd.DataFrame)
        (batch_id, stats_df) pairs; each DataFrame has columns
        'metric' and 'value'.

    Returns
    -------
    dict
        {metric: (mean, std, sem)}.  std is the SAMPLE standard deviation
        (ddof=1); sem = std / sqrt(n).  Both are NaN when fewer than two
        non-NaN values are available, because dispersion is then undefined
        (reporting 0.0 would falsely imply zero variance).
    """
    recap = {}
    for _batch_id, stats_df in batch_stats_list:
        for metric, value in zip(stats_df["metric"], stats_df["value"]):
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            recap.setdefault(str(metric), []).append(v)

    out = {}
    for metric, values in recap.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        n = arr.size
        if n == 0:
            out[metric] = (np.nan, np.nan, np.nan)
        elif n == 1:
            # A single simulation carries no dispersion information.
            out[metric] = (float(arr[0]), np.nan, np.nan)
        else:
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1))
            sem = std / np.sqrt(n)
            out[metric] = (mean, std, float(sem))
    return out


def _read_param_folder_batches(param_path):
    """Collect (batch_id, stats_df) for every batch under a parameter folder."""
    found = []
    for entry in sorted(os.listdir(param_path)):
        stats_path = os.path.join(param_path, entry, "batch_stats.parquet")
        if os.path.exists(stats_path):
            found.append((entry, pd.read_parquet(stats_path)))
    return found


def _build_stats_rows(root, combined):
    """Build one row per parameter folder found under *root*."""
    rows = []
    if not os.path.isdir(root):
        logger.warning(f"export_all_stats: directory not found: {root}")
        return rows

    for pf in sorted(os.listdir(root)):
        param_path = os.path.join(root, pf)
        if not os.path.isdir(param_path):
            continue

        batches = _read_param_folder_batches(param_path)
        if not batches:
            continue

        row = parse_param_folder_name(pf, combined=combined)
        row["n_batches"] = len(batches)

        # Each triple is quoted at the decimal place set by its own sem, so
        # mean/std/sem of a metric are digit-aligned and no fake precision is
        # carried into the file.
        for metric, (mean, std, sem) in aggregate_batch_stats(batches).items():
            d = significant_decimals(sem, std, mean)
            row[f"{metric}_mean"] = format_to_decimals(mean, d)
            row[f"{metric}_std"] = format_to_decimals(std, d)
            row[f"{metric}_sem"] = format_to_decimals(sem, d)

        row["_param_folder"] = pf
        rows.append(row)
        logger.debug(f"export_all_stats: {pf} -> {len(batches)} batch(es)")
    return rows


def _stats_rows_to_frame(rows, param_columns):
    """
    Assemble rows into an ordered DataFrame.

    Columns: params + n_batches + (metric_mean, metric_std, metric_sem)*
    + _param_folder.  Metrics missing from a given parameter
    folder become NaN, which is exactly what a union of heterogeneous metric
    sets should produce (e.g. Moran's I is absent from non-grid "ce" runs).

    Rows are sorted by the parameter columns.
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Recover the metric names from the "_mean" columns.
    metrics = _order_metrics([c[:-5] for c in df.columns if c.endswith("_mean")])

    param_cols = [col for _tok, col in param_columns]
    lead = param_cols + ["n_batches"]
    metric_cols = [f"{m}_{suffix}" for m in metrics for suffix in ("mean", "std", "sem")]
    ordered = lead + metric_cols + ["_param_folder"]

    for col in ordered:
        if col not in df.columns:
            # Metric columns hold pre-formatted strings; a missing metric (e.g.
            # Moran's I on a non-grid "ce" run) becomes an empty cell.
            df[col] = "" if col.endswith(("_mean", "_std", "_sem")) else np.nan
    df = df[ordered]
    for col in metric_cols:
        df[col] = df[col].fillna("")

    # Optional tokens (un / tk / ks / cl / cn ...) are absent from some folders,
    # so their columns carry NaN and pandas silently upcasts them to float
    # ("cell_number" -> 15.0).  Nullable Int64 keeps them readable as integers.
    for col in param_cols + ["n_batches"]:
        s = df[col]
        if s.dtype.kind in "fiu":
            valid = s.dropna()
            if len(valid) and np.all(valid == valid.astype(np.int64)):
                df[col] = s.astype("Int64")

    sort_cols = [c for c in param_cols if df[c].notna().any()]
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def export_all_stats(output_dir=None, sep="	", also_txt=True):
    """
    Export aggregated batch statistics for every parameter combination on disk.

    Walks analysis_data/ and combined_analysis_data/ and writes:

    all_stats_baseline.tsv
        One row per parameter folder of the baselines.
    all_stats_continuation.tsv
        One row per parameter folder of the continuations.

    Unlike the plot loaders, this function takes NO filters: by construction it
    scans everything present on disk, like analyse_all / analyse_all_combined.

    Parameters
    ----------
    output_dir : str, optional
        Destination directory.  Defaults to <project_root>/stats_export/.
    also_txt : bool
        Additionally write a fixed-width, human-readable ".txt" twin of each
        table (same data, columns padded so they line up in any editor).
    sep : str
        Field separator.  Defaults to a TAB, which keeps columns aligned when
        the file is opened in an editor and avoids any clash with the decimal
        separator.  Pass "," for plain CSV.

    Returns
    -------
    dict
        {"baseline": path_or_None, "continuation": path_or_None}
    """
    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    if output_dir is None:
        output_dir = os.path.join(project_root, "stats_export")
    os.makedirs(output_dir, exist_ok=True)

    ext = "tsv" if sep == "	" else "csv"
    written = {"baseline": None, "continuation": None}

    def _dump(df, stem):
        """Write the machine-readable table and, optionally, its aligned twin."""
        path = os.path.join(output_dir, f"{stem}.{ext}")
        df.to_csv(path, index=False, sep=sep)
        if also_txt:
            cells = df.astype(object).where(df.notna(), "").astype(str)
            lines = render_aligned_table(list(df.columns), cells.values.tolist())
            txt_path = os.path.join(output_dir, f"{stem}.txt")
            with open(txt_path, "w") as fh:
                fh.write("\n".join(lines) + "\n")
            logger.debug(f"export_all_stats: wrote aligned twin -> {txt_path}")
        return path

    # ---------- analysis_data ----------
    rows = _build_stats_rows(os.path.join(project_root, "analysis_data"), combined=False)
    df = _stats_rows_to_frame(rows, _ANALYSIS_PARAM_COLUMNS)
    if df.empty:
        logger.warning("export_all_stats: no analysis_data stats found.")
    else:
        path = _dump(df, "all_stats_baseline")
        written["baseline"] = path
        logger.debug(f"export_all_stats: wrote {len(df)} row(s) -> {path}")

    # ---------- combined_analysis_data ----------
    croot = os.path.join(project_root, "combined_analysis_data")
    combined_rows = _build_stats_rows(croot, combined=True)

    cdf = _stats_rows_to_frame(combined_rows, _COMBINED_PARAM_COLUMNS)
    if cdf.empty:
        logger.warning("export_all_stats: no combined_analysis_data stats found.")
    else:
        path = _dump(cdf, "all_stats_continuation")
        written["continuation"] = path
        logger.debug(f"export_all_stats: wrote {len(cdf)} row(s) -> {path}")

    return written

# ##########################################################################
# BATCH-METRIC REGISTRY  (merged from the former batch_metrics.py)
#
# Uniform registry for every quantity written to batch_stats.parquet: the
# collective-entropy metrics, gini, Moran's I, and the mean_* aggregates.
# Because each batch metric is computed from on-disk artifacts, a selective
# re-analysis can (re)compute a chosen subset and MERGE it into an existing
# batch_stats.parquet without re-running the expensive per-user pass.
# ##########################################################################

# ==========================================================================
# Core: Shannon entropy and occupancy reconstruction
# ==========================================================================

def shannon_entropy_bits(counts):
    """
    Shannon entropy (base 2, in bits) of a non-negative count array.

    The array is flattened and normalised to a probability vector p_c = n_c/N;
    empty cells contribute 0 (0*log0 := 0).  Returns NaN for an all-zero or
    empty input (undefined distribution).
    """
    a = np.asarray(counts, dtype=float).ravel()
    total = a.sum()
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    p = a[a > 0] / total
    return float(-(p * np.log2(p)).sum()) + 0.0   # +0.0 collapses -0.0 to 0.0


_DEST_X = "Destination_x_coord"
_DEST_Y = "Destination_y_coord"


def aggregates_from_matrices(matrices):
    """
    Sum a list of per-user grid matrices into (visits_agg, exploration_agg).

    visits_agg      element-wise sum of the raw count matrices.
    exploration_agg sum of the binarised (>0) matrices == distinct users/cell.

    Returns (None, None) when no usable matrix is present.  Works in memory, so
    the aggregates are produced without persisting per-user grid parquets.
    """
    visits = None
    expl = None
    for m in matrices:
        if m is None:
            continue
        mat = np.asarray(m.values if isinstance(m, pd.DataFrame) else m).astype(np.int64)
        bmat = (mat > 0).astype(np.int64)
        visits = mat if visits is None else visits + mat
        expl = bmat if expl is None else expl + bmat
    return visits, expl


def rollup_param_aggregates(batch_folder):
    """
    Sum every sibling batch aggregate into a param-level aggregate.

    Walk the parent folder, sum each subfolder's aggregate_grid.parquet (and
    aggregate_grid_exploration.parquet) and overwrite the parent-level files.
    Safe to call after the batch-level aggregate has just been written.
    """
    param_folder = os.path.dirname(batch_folder)
    vis_sum = None
    exp_sum = None
    for entry in os.listdir(param_folder):
        ep = os.path.join(param_folder, entry)
        if not os.path.isdir(ep):
            continue
        vp = os.path.join(ep, "aggregate_grid.parquet")
        if os.path.exists(vp):
            m = pd.read_parquet(vp).values.astype(np.int64)
            vis_sum = m if vis_sum is None else vis_sum + m
        epath = os.path.join(ep, "aggregate_grid_exploration.parquet")
        if os.path.exists(epath):
            m = pd.read_parquet(epath).values.astype(np.int64)
            exp_sum = m if exp_sum is None else exp_sum + m
    if vis_sum is not None:
        df = pd.DataFrame(vis_sum); df.columns = df.columns.astype(str)
        df.to_parquet(os.path.join(param_folder, "aggregate_grid.parquet"), index=False)
    if exp_sum is not None:
        df = pd.DataFrame(exp_sum); df.columns = df.columns.astype(str)
        df.to_parquet(os.path.join(param_folder, "aggregate_grid_exploration.parquet"), index=False)


# ==========================================================================
# Batch context: lazy, cached access to a single analysed batch folder
# ==========================================================================

class BatchContext:
    """
    Read-side accessor for one analysed batch folder.

    A "batch folder" is a timestamped analysis folder (baseline or
    continuation).  Everything is read from
    disk lazily and cached, so building a metric costs only the files it needs.
    """

    def __init__(self, folder, cell_lenght=None, cell_number=None, n_users=None):
        self.folder = folder
        self.cell_lenght = cell_lenght
        self.cell_number = cell_number
        self._n_users = n_users
        self._cache = {}

    # ---- users ----
    def _count_users(self):
        i = 0
        while os.path.exists(os.path.join(self.folder, f"user_{i}_history.parquet")):
            i += 1
        return i

    @property
    def n_users(self):
        if self._n_users is None:
            self._n_users = self._count_users()
        return self._n_users

    def _read(self, name):
        p = os.path.join(self.folder, name)
        return pd.read_parquet(p) if os.path.exists(p) else None

    # ---- batch-level artifacts ----
    def aggregate_visits(self):
        if "av" not in self._cache:
            df = self._read("aggregate_grid.parquet")
            self._cache["av"] = None if df is None else df.values.astype(float)
        return self._cache["av"]

    def aggregate_exploration(self):
        if "ae" not in self._cache:
            df = self._read("aggregate_grid_exploration.parquet")
            self._cache["ae"] = None if df is None else df.values.astype(float)
        return self._cache["ae"]

    # ---- per-user artifacts ----
    def venue_list(self):
        if "vh" not in self._cache:
            self._cache["vh"] = [self._read(f"user_{j}_venue.parquet")
                                 for j in range(self.n_users)]
        return self._cache["vh"]

    def all_user_stats(self):
        """{metric_name: [value per user]} read once from user_{j}_stats.parquet."""
        if "us" in self._cache:
            return self._cache["us"]
        out = {}
        n = self.n_users
        for j in range(n):
            sf = self._read(f"user_{j}_stats.parquet")
            if sf is None:
                continue
            for _, row in sf.iterrows():
                out.setdefault(str(row["metric"]), [np.nan] * n)
                out[str(row["metric"])][j] = float(row["value"])
        self._cache["us"] = out
        return out

    def stat_values(self, metric):
        return self.all_user_stats().get(metric, [np.nan] * self.n_users)


# ==========================================================================
# Metric registry
# ==========================================================================
#
# Each entry: name -> dict(fn, needs_grid, group)
#   fn        : callable(ctx) -> float, OR (for a group) callable(ctx) -> dict
#   needs_grid: True if undefined without a grid (entropy, Moran)
#   group     : shared key for multi-output metrics computed by ONE call
#               (Moran yields I / E_I / z_score / p_value together)

def _m_n_users(ctx):
    return float(ctx.n_users)


def _m_gini_collective(ctx):
    return compute_collective_gini(ctx.venue_list())


def _m_gini_individual(ctx):
    return compute_individual_gini(ctx.venue_list())


def _mean_of(stat_name):
    def fn(ctx):
        # NaN without a warning when no individual has the statistic (e.g.
        # the quantities after t0 in a baseline).
        v = np.asarray(ctx.stat_values(stat_name), dtype=float)
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    return fn


def _m_mean_2rg_ratio(ctx):
    rg = np.asarray(ctx.stat_values("radius of gyration"), dtype=float)
    rg2 = np.asarray(ctx.stat_values("radius of gyration two"), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rg > 0, rg2 / rg, np.nan)
    return float(np.nanmean(ratio))


def _m_collective_entropy_visits(ctx):
    m = ctx.aggregate_visits()
    return shannon_entropy_bits(m) if m is not None else float("nan")


def _m_collective_entropy_exploration(ctx):
    m = ctx.aggregate_exploration()
    return shannon_entropy_bits(m) if m is not None else float("nan")


def _m_moran_group(ctx):
    nan4 = {"I": float("nan"), "E_I": float("nan"),
            "z_score": float("nan"), "p_value": float("nan")}
    m = ctx.aggregate_visits()
    if m is None:
        return nan4
    try:
        return compute_moran_i(pd.DataFrame(m))
    except ValueError:
        # Degenerate grid (all cells equal): Moran's I undefined.
        return nan4


REGISTRY = {
    "n_users":              {"fn": _m_n_users,              "needs_grid": False, "group": None},
    "gini_collective":      {"fn": _m_gini_collective,      "needs_grid": False, "group": None},
    "gini_individual":      {"fn": _m_gini_individual,      "needs_grid": False, "group": None},
    "mean_rg":              {"fn": _mean_of("radius of gyration"),     "needs_grid": False, "group": None},
    "mean_2rg":             {"fn": _mean_of("radius of gyration two"), "needs_grid": False, "group": None},
    "mean_2rg_ratio":       {"fn": _m_mean_2rg_ratio,       "needs_grid": False, "group": None},
    "mean_rg_explorative":  {"fn": _mean_of("rg_explorative"),         "needs_grid": False, "group": None},
    "mean_travel_distance": {"fn": _mean_of("mean_travel_distance"),   "needs_grid": False, "group": None},
    "mean_num_venues":      {"fn": _mean_of("num_venues"),  "needs_grid": False, "group": None},
    "mean_s_rand":          {"fn": _mean_of("s_rand"),      "needs_grid": False, "group": None},
    "mean_s_unc":           {"fn": _mean_of("s_unc"),       "needs_grid": False, "group": None},
    "mean_s_real":          {"fn": _mean_of("s_real"),      "needs_grid": False, "group": None},
    "mean_pi_rand":         {"fn": _mean_of("pi_rand"),     "needs_grid": False, "group": None},
    "mean_pi_unc":          {"fn": _mean_of("pi_unc"),      "needs_grid": False, "group": None},
    "mean_pi_real":         {"fn": _mean_of("pi_real"),     "needs_grid": False, "group": None},
    # --- centre-of-mass displacement between loaded and resumed phases (NEW) ---
    # abs = mean over users of the displacement magnitude (unsigned typical drift);
    # rel_x / rel_y = mean over users of the signed displacement components
    # (the net mean drift vector, so opposite drifts cancel).
    "cm_displacement_abs":   {"fn": _mean_of("cm_displacement_mag"), "needs_grid": False, "group": None},
    "cm_displacement_rel_x": {"fn": _mean_of("cm_displacement_dx"),  "needs_grid": False, "group": None},
    "cm_displacement_rel_y": {"fn": _mean_of("cm_displacement_dy"),  "needs_grid": False, "group": None},
    # --- collective spatial entropy (NEW) ---
    "collective_entropy_visits":      {"fn": _m_collective_entropy_visits,      "needs_grid": True, "group": None},
    "collective_entropy_exploration": {"fn": _m_collective_entropy_exploration, "needs_grid": True, "group": None},
    # --- Moran's I group (one call, four outputs) ---
    "I":       {"fn": _m_moran_group, "needs_grid": True, "group": "moran"},
    "E_I":     {"fn": _m_moran_group, "needs_grid": True, "group": "moran"},
    "z_score": {"fn": _m_moran_group, "needs_grid": True, "group": "moran"},
    "p_value": {"fn": _m_moran_group, "needs_grid": True, "group": "moran"},
}

# Names understood as batch metrics (used to decide if a selective re-analysis
# can skip the per-user pass entirely: it can iff every requested name is here).
BATCH_METRIC_NAMES = set(REGISTRY.keys())


def compute_batch_metrics(ctx, metrics=None, grid_active=True):
    """
    Compute the requested batch metrics from an on-disk batch folder.

    metrics : None  -> every registered metric
              list  -> only these names (unknown names are ignored with a warn)
    grid_active : when False, grid-only metrics (entropy, Moran) are skipped.

    Returns a list of {"metric": name, "value": float} rows.  Multi-output
    groups (Moran) are computed once even if several of their keys are asked.
    """
    if metrics is None:
        names = list(REGISTRY.keys())
    else:
        names = []
        for n in metrics:
            if n in REGISTRY:
                names.append(n)
            else:
                logger.warning(f"compute_batch_metrics: unknown metric '{n}' ignored.")

    rows = []
    done_groups = {}
    for name in names:
        entry = REGISTRY[name]
        if entry["needs_grid"] and not grid_active:
            continue
        group = entry["group"]
        if group is not None:
            if group not in done_groups:
                done_groups[group] = entry["fn"](ctx)   # dict of all group keys
            val = done_groups[group].get(name, float("nan"))
        else:
            val = entry["fn"](ctx)
        rows.append({"metric": name, "value": float(val)})
    return rows


# ==========================================================================
# Writing: full write vs selective merge
# ==========================================================================

def _order_rows(df):
    """Order rows by the export preference, unknown metrics appended after."""
    order = {m: i for i, m in enumerate(_METRIC_ORDER)}
    df = df.copy()
    df["_k"] = df["metric"].map(lambda m: order.get(m, len(order) + hash(m) % 1000))
    df = df.sort_values("_k").drop(columns="_k").reset_index(drop=True)
    return df


def write_batch_stats(folder, rows, merge=False):
    """
    Persist batch metric rows to <folder>/batch_stats.parquet.

    merge=False : write the given rows as the whole file (fresh analysis).
    merge=True  : load the existing file, replace ONLY the rows whose metric is
                  in `rows`, keep every other metric untouched (selective
                  re-analysis / backfill).
    """
    path = os.path.join(folder, "batch_stats.parquet")
    new_df = pd.DataFrame(rows)
    if merge and os.path.exists(path):
        old = pd.read_parquet(path)
        keep = old[~old["metric"].isin(new_df["metric"])]
        out = pd.concat([keep, new_df], ignore_index=True)
    else:
        out = new_df
    out = _order_rows(out)
    out.to_parquet(path, index=False)
    return path


# ==========================================================================
# Selective re-analysis / backfill over everything on disk
# ==========================================================================

_CL_CN_RE = re.compile(r"_cl_(\d+)_cn_(\d+)")


def _grid_from_param_folder(param_folder):
    """Parse (cell_lenght, cell_number) from a '..._cl_100_cn_51_...' folder name."""
    m = _CL_CN_RE.search(os.path.basename(param_folder))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _iter_batch_dirs(root, combined):
    """
    Yield (batch_folder, param_folder) for every analysed batch under `root`.
    """
    if not os.path.isdir(root):
        return
    for param in sorted(os.listdir(root)):
        pf = os.path.join(root, param)
        if not os.path.isdir(pf):
            continue
        for batch in sorted(os.listdir(pf)):
            bf = os.path.join(pf, batch)
            if not os.path.isdir(bf):
                continue
            yield bf, pf


def reanalyze_batch_metrics(project_root, metrics, do_analysis=True,
                            do_combined=True):
    """
    (Re)compute a SELECTIVE list of batch metrics across every analysed batch on
    disk and MERGE them into each batch_stats.parquet, without touching per-user
    artifacts and without re-running the expensive per-user analysis pass.

    This is the cheap engine behind "re-analyse everything with a metric list":
    it is valid precisely because every name in `metrics` is a batch metric
    (see BATCH_METRIC_NAMES) whose inputs already sit on disk.

    Returns the number of batch folders updated.
    """
    unknown = [m for m in metrics if m not in BATCH_METRIC_NAMES]
    if unknown:
        raise ValueError(
            f"reanalyze_batch_metrics only handles batch metrics; unknown/"
            f"non-batch names: {unknown}. Known: {sorted(BATCH_METRIC_NAMES)}")

    roots = []
    if do_analysis:
        roots.append((os.path.join(project_root, "analysis_data"), False))
    if do_combined:
        roots.append((os.path.join(project_root, "combined_analysis_data"), True))

    updated = 0
    for root, combined in roots:
        for bf, pf in _iter_batch_dirs(root, combined):
            cl, cn = _grid_from_param_folder(pf)
            grid_active = cn is not None
            ctx = BatchContext(bf, cell_lenght=cl, cell_number=cn)
            rows = compute_batch_metrics(ctx, metrics=metrics, grid_active=grid_active)
            if rows:
                write_batch_stats(bf, rows, merge=True)
                updated += 1
    logger.debug(f"reanalyze_batch_metrics: updated {updated} batch folder(s) "
                f"for metrics={metrics}")
    return updated