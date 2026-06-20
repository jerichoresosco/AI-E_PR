import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import inspect
import scipy.stats
from libpysal.weights import lat2W
from esda.moran import Moran
import time
import logging
logger = logging.getLogger(__name__)

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

    #logger.info(
    #    f"Generating {n_samples} samples from power-law "
    #    f"distribution (par={par}, min_val={min_val}, max_val={max_val})"
    #)


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
    #logger.info("Spatial sampling...")
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
    #logger.info("Temporal sampling...")
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
      - s_real  : real entropy (Lempel-Ziv)      [nats]

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
            {"metric": "s_rand",                 "value": float("nan")},
            {"metric": "s_unc",                  "value": float("nan")},
            {"metric": "s_real",                 "value": float("nan")},
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

    # --- entropy measures ---
    s_rand = compute_random_entropy(venue_history_df)
    s_unc  = compute_uncorrelated_entropy(venue_history_df)
    s_real = compute_real_entropy(user_history_df)

    return pd.DataFrame([
        {"metric": "x_center of mass",      "value": x_cm},
        {"metric": "y_center of mass",       "value": y_cm},
        {"metric": "x_center of mass two",   "value": x_cm_two},
        {"metric": "y_center of mass two",   "value": y_cm_two},
        {"metric": "radius of gyration",     "value": r_gyr},
        {"metric": "radius of gyration two", "value": r_gyr_two},
        {"metric": "s_rand",                 "value": s_rand},
        {"metric": "s_unc",                  "value": s_unc},
        {"metric": "s_real",                 "value": s_real},
    ])


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

def compute_uncorrelated_entropy(venue_history_df):
    """
    Compute the temporal-uncorrelated entropy
        S^unc = -sum_j  p_j * log2(p_j)
    where p_j is the fraction of total simulation time spent at venue j.

    Unlike the frequentist version (based on visit counts), this weights each
    venue by the actual time spent there, making the probability estimate
    more accurate for heterogeneous dwell times.

    Parameters
    ----------
    venue_history_df : pd.DataFrame
        Venue history with a 'Total time' column (cumulative dwell time per venue).

    Returns
    -------
    float
        S^unc in bits. Returns 0.0 for degenerate cases.
    """
    times = venue_history_df["Total time"].values.astype(float)
    times = times[times > 0]
    total = times.sum()
    if total == 0 or len(times) <= 1:
        return 0.0
    probs = times / total
    return float(-np.sum(probs * np.log2(probs)))

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

        S^est = (1/n * sum_i Lambda_i)^{-1} * ln(n)

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
        S^est in natural logarithm.
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
    return float((1.0 / mean_lambda) * np.log(n))  # natural log, as in the paper


def load_parquet_folder(base_folder, filters, file_map, grid=False, return_run_folders=False, starting_position="centered", user_number=None):
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
    Run folders that contain a ``_analyzed.lock`` file are silently skipped.
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
            "algorithm":   "alg",
        }
    elif grid == False:
        key_map = {
            "ro":        "ro",
            "gamma":     "g",
            "a_param":   "a",
            "b_param":   "b",
            "time":      "t",
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

        # Scarta cartelle che non corrispondono alla modalità grid richiesta
        folder_has_grid = "cl" in converted and "cn" in converted
        if grid != folder_has_grid:
            continue

        # Scarta cartelle che non corrispondono alla modalità starting_position richiesta
        if grid:
            _sp_abbr = {"centered": "c", "random": "r", "normal": "n"}.get(starting_position, "c")
            if converted.get("sp") != _sp_abbr:
                continue

        if any(converted.get(key_map.get(fk, fk)) != fv
               for fk, fv in filters.items()):
            continue

        for run_folder in sorted(os.listdir(folder_path)):  # sorted for determinism
            run_path = os.path.join(folder_path, run_folder)
            if not os.path.isdir(run_path):
                continue

            if os.path.exists(os.path.join(run_path, "_analyzed.lock")):
                logger.debug(f"Skipping already-analyzed run folder: {run_path}")
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
    done = False
    for run_path, user_indices in matching_runs:
        if done:
            break
        total_in_run = len(user_indices)
        milestone    = max(1, total_in_run // 10)
        for pos, idx in enumerate(user_indices):
            if early_stop is not None and len(loaded) >= early_stop:
                done = True
                break
            entry = {}
            for key, suffix in file_map.items():
                fpath = os.path.join(run_path, f"user_{idx}{suffix}")
                entry[key] = pd.read_parquet(fpath) if os.path.exists(fpath) else None
            if entry[next(iter(file_map))] is not None:
                if return_run_folders:
                    loaded.append({"data": entry, "run_folder": run_path})
                else:
                    loaded.append(entry)
            if (pos + 1) % milestone == 0 or (pos + 1) == total_in_run:
                pct = ((pos + 1) / total_in_run) * 100
                logger.info(
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
    logger.info("Loading raw simulation data (raw_data)...")

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
                                 starting_position=filters.get("starting_position", "centered"),
                                 user_number=user_number)
    loaded = _apply_user_number_slice(loaded, user_number)

    logger.info(f"Loaded {len(loaded)} users from raw_data")
    return loaded


def load_analysis_parquet(filters=None, needed_files=None):
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
    logger.info("Loading analysis data (analysis_data)...")

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
        logger.info(f"Loading only: {sorted(needed_files)}")

    loaded = load_parquet_folder(base_folder, filters_clean, file_map, grid=grid,
                                 starting_position=filters.get("starting_position", "centered"),
                                 user_number=user_number)
    loaded = _apply_user_number_slice(loaded, user_number)

    logger.info(f"Analysis data loaded: {len(loaded)} users")
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

def compute_and_save_aggregate_grid(analysis_folder):
    """
    Compute and save aggregate grid matrices at batch and parameter level.

    Batch level  : sums all user_i_grid.parquet inside analysis_folder and
                   saves the result as aggregate_grid.parquet in that same
                   folder.
    Param level  : re-scans every batch subfolder of the parent param folder,
                   sums their aggregate_grid.parquet files, and saves the
                   result as aggregate_grid.parquet directly in the param
                   folder (overwriting any previous version).

    Must be called after all per-user grid parquets have been written inside
    analysis_folder.

    Parameters
    ----------
    analysis_folder : str
        Path to the timestamped batch run folder produced by run_analysis()
        (e.g. analysis_data/ro_0.1_.../20260310_160442_960747/).

    Returns
    -------
    batch_agg : pd.DataFrame
        The batch-level aggregate grid (sum over all users in analysis_folder).
    param_agg : pd.DataFrame or None
        The param-level aggregate grid (sum over all batch aggregates in the
        parent param folder). None if no batch aggregates were found.
    """
    # --- batch-level aggregate ---
    grid_sum = None
    i = 0
    while True:
        fpath = os.path.join(analysis_folder, f"user_{i}_grid.parquet")
        if not os.path.exists(fpath):
            break
        mat = pd.read_parquet(fpath).values.astype(int)
        grid_sum = mat if grid_sum is None else grid_sum + mat
        i += 1

    if grid_sum is None:
        logger.warning(f"No user grid files found in {analysis_folder}; skipping aggregate.")
        return

    batch_agg = pd.DataFrame(grid_sum)
    batch_agg.columns = batch_agg.columns.astype(str)
    batch_agg.to_parquet(os.path.join(analysis_folder, "aggregate_grid.parquet"), index=False)
    logger.info(f"Batch-level aggregate grid saved to: {analysis_folder}")

    # --- param-level aggregate (sum over all existing batch aggregates) ---
    param_folder = os.path.dirname(analysis_folder)
    param_sum = None
    for entry in os.listdir(param_folder):
        entry_path = os.path.join(param_folder, entry)
        if not os.path.isdir(entry_path):
            continue
        agg_path = os.path.join(entry_path, "aggregate_grid.parquet")
        if not os.path.exists(agg_path):
            continue
        mat = pd.read_parquet(agg_path).values.astype(int)
        param_sum = mat if param_sum is None else param_sum + mat

    if param_sum is not None:
        param_agg = pd.DataFrame(param_sum)
        param_agg.columns = param_agg.columns.astype(str)
        param_agg.to_parquet(os.path.join(param_folder, "aggregate_grid.parquet"), index=False)
        logger.info(f"Param-level aggregate grid saved to: {param_folder}")

    return batch_agg, param_agg

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

    sp    = filters.get("starting_position", "centered")
    alg   = filters.get("algorithm", "ce")
    _sp_abbr = {"centered": "c", "random": "r", "normal": "n"}.get(sp, "c")
    param_folder_name = (
        f"ro_{filters['ro']}_g_{filters['gamma']}"
        f"_a_{filters['a_param']}_b_{filters['b_param']}_t_{filters['time']}"
    )
    if "cell_lenght" in filters and "cell_number" in filters:
        param_folder_name += f"_cl_{filters['cell_lenght']}_cn_{filters['cell_number']}"
    param_folder_name += f"_sp_{_sp_abbr}_alg_{alg}"

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

    logger.info(
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

    sp    = filters.get("starting_position", "centered")
    alg   = filters.get("algorithm", "ce")
    _sp_abbr = {"centered": "c", "random": "r", "normal": "n"}.get(sp, "c")
    param_folder_name = (
        f"ro_{filters['ro']}_g_{filters['gamma']}"
        f"_a_{filters['a_param']}_b_{filters['b_param']}_t_{filters['time']}"
    )
    if "cell_lenght" in filters and "cell_number" in filters:
        param_folder_name += f"_cl_{filters['cell_lenght']}_cn_{filters['cell_number']}"
    param_folder_name += f"_sp_{_sp_abbr}_alg_{alg}"

    param_folder = os.path.join(base_folder, param_folder_name)
    logger.info(f"load_all_batch_stats looking for: {param_folder}")
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
            logger.info(f"  Found batch stats: {entry}")
        else:
            logger.debug(f"  Skipping (no batch_stats.parquet): {entry}")

    logger.info(f"Loaded batch stats for {len(batch_stats_list)} batch(es) from: {param_folder}")
    return batch_stats_list

def _build_combined_param_folder_name(filters, grid_params=None):
    """
    Build the parameter folder name used by _make_combined_run_folder().

    Expected keys in filters: "ro", "gamma", "a_param", "b_param",
    "t0" (time of base run), "t1" (time of resume run),
    "starting_position" (optional), "algorithm_0" (optional, default "ce"),
    "algorithm_1" (optional, default "ce").

    Mirrors the naming logic in Simulation._make_combined_run_folder().
    """
    ro      = filters["ro"]
    gamma   = filters["gamma"]
    a_param = filters["a_param"]
    b_param = filters["b_param"]
    t0      = filters["t0"]
    t1      = filters["t1"]
    sp      = filters.get("starting_position", "centered")
    alg0    = filters.get("algorithm_0", "ce")
    alg1    = filters.get("algorithm_1", "ce")

    _sp_abbr = {"centered": "c", "random": "r", "normal": "n"}.get(sp, "c")

    name = (
        f"ro_{ro}_g_{gamma}"
        f"_a_{a_param}_b_{b_param}"
        f"_t0_{t0}_t1_{t1}"
    )
    if grid_params is not None:
        name += f"_cl_{grid_params['cell_lenght']}_cn_{grid_params['cell_number']}"
    elif "cell_lenght" in filters and "cell_number" in filters:
        name += f"_cl_{filters['cell_lenght']}_cn_{filters['cell_number']}"
    name += f"_sp_{_sp_abbr}_alg0_{alg0}_alg1_{alg1}"
    return name


def load_combined_analysis_parquet(filters=None, needed_files=None):
    """
    Load analysis data from combined_analysis_data/.

    Mirrors load_analysis_parquet() but targets the two-level folder structure
    produced by run_analysis(combined=True) / lsa mode.

    Parameters
    ----------
    filters : dict, optional
        Must contain: "ro", "gamma", "a_param", "b_param", "t0", "t1".
        Optional: "starting_position", "algorithm_0", "algorithm_1",
        "user_number", "cell_lenght", "cell_number".
        "variant" (str, optional): "pure" or "mixed" (default "mixed").
            Selects the subfolder inside each batch timestamp folder.
    needed_files : set of str, optional
        Subset of file-map keys to load.  When None, all files are loaded.

    Returns
    -------
    list of dict
        One dict per user with keys from the effective file_map.
    """
    logger.info("Loading analysis data (combined_analysis_data)...")

    if filters is None:
        filters = {}

    user_number = filters.get("user_number", None)
    variant     = filters.get("variant", "mixed")   # "pure" or "mixed"

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

    # Load from batch subfolders, navigating into the variant subfolder
    primary_suffix = next(iter(file_map.values()))
    early_stop = user_number if (user_number is not None and user_number > 0) else None

    loaded = []
    done   = False
    for batch_id in sorted(os.listdir(param_folder)):
        if done:
            break
        batch_path = os.path.join(param_folder, batch_id)
        if not os.path.isdir(batch_path):
            continue

        # Navigate into variant subfolder (pure/ or mixed/).
        # Fall back to the batch folder itself for old-format data that has no subfolders.
        variant_path = os.path.join(batch_path, variant)
        run_path = variant_path if os.path.isdir(variant_path) else batch_path

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
        milestone    = max(1, total_in_run // 10)
        for pos, (_, idx) in enumerate(user_indices):
            if early_stop is not None and len(loaded) >= early_stop:
                done = True
                break
            entry = {}
            for key, suffix in file_map.items():
                fpath = os.path.join(run_path, f"user_{idx}{suffix}")
                entry[key] = pd.read_parquet(fpath) if os.path.exists(fpath) else None
            if entry[next(iter(file_map))] is not None:
                loaded.append(entry)
            if (pos + 1) % milestone == 0 or (pos + 1) == total_in_run:
                pct = ((pos + 1) / total_in_run) * 100
                logger.info(
                    f"Loading progress: {pct:.0f}% ({pos+1}/{total_in_run} users) [{batch_id}/{variant}]"
                )

    loaded = _apply_user_number_slice(loaded, user_number)
    logger.info(f"Combined analysis data loaded: {len(loaded)} users (variant={variant})")
    return loaded


def load_combined_all_aggregate_grids(filters):
    """
    Load param-level and batch-level aggregate grids from combined_analysis_data/.

    Mirrors load_all_aggregate_grids() for the combined folder structure.

    Parameters
    ----------
    filters : dict
        Same keys as load_combined_analysis_parquet().
        "variant" (str, optional): "pure" or "mixed" (default "mixed").

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

    variant = filters.get("variant", "mixed")

    param_folder_name = _build_combined_param_folder_name(filters)
    param_folder = os.path.join(base_folder, param_folder_name)

    if not os.path.exists(param_folder):
        logger.warning(f"Param folder not found for combined aggregate grids: {param_folder}")
        return None, 0, []

    # Param-level aggregate: look inside the variant subfolder of param_folder
    # (a param-level aggregate_grid.parquet placed directly in param_folder is
    #  not written by the new code; fall back gracefully if absent)
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
        # Navigate into variant subfolder; fall back to batch folder for old-format data
        variant_path = os.path.join(batch_path, variant)
        run_path = variant_path if os.path.isdir(variant_path) else batch_path

        agg_path = os.path.join(run_path, "aggregate_grid.parquet")
        if os.path.exists(agg_path):
            mat = pd.read_parquet(agg_path).values.astype(int)
            batch_user_count = sum(
                1 for f in os.listdir(run_path)
                if f.startswith("user_") and f.endswith("_grid.parquet")
            )
            batch_grids.append((entry, mat, batch_user_count))
            param_user_count += batch_user_count

    logger.info(
        f"Loaded combined aggregate grids (variant={variant}): 1 param-level ({param_user_count} users), "
        f"{len(batch_grids)} batch-level"
    )
    return param_grid, param_user_count, batch_grids


def load_combined_all_batch_stats(filters):
    """
    Load batch_stats.parquet from combined_analysis_data/.

    Mirrors load_all_batch_stats() for the combined folder structure.

    Parameters
    ----------
    filters : dict
        Same keys as load_combined_analysis_parquet().
        "variant" (str, optional): "pure" or "mixed" (default "mixed").

    Returns
    -------
    batch_stats_list : list of (str, pd.DataFrame)
    """
    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "combined_analysis_data")

    variant = filters.get("variant", "mixed")

    param_folder_name = _build_combined_param_folder_name(filters)
    param_folder = os.path.join(base_folder, param_folder_name)

    logger.info(f"load_combined_all_batch_stats looking for: {param_folder} (variant={variant})")
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
        # Navigate into variant subfolder; fall back to batch folder for old-format data
        variant_path = os.path.join(batch_path, variant)
        run_path = variant_path if os.path.isdir(variant_path) else batch_path

        stats_path = os.path.join(run_path, "batch_stats.parquet")
        if os.path.exists(stats_path):
            stats_df = pd.read_parquet(stats_path)
            batch_stats_list.append((entry, stats_df))
            logger.info(f"  Found combined batch stats: {entry}/{variant}")
        else:
            logger.debug(f"  Skipping (no batch_stats.parquet): {entry}/{variant}")

    logger.info(f"Loaded combined batch stats for {len(batch_stats_list)} batch(es) from: {param_folder} (variant={variant})")
    return batch_stats_list


def compute_moran_i(batch_agg):
    """
    Compute Moran's I spatial autocorrelation statistic on a batch aggregate
    grid, using esda (PySAL) with a rook-contiguity spatial weights matrix.
 
    Parameters
    ----------
    batch_agg : pd.DataFrame
        Batch-level aggregate grid as returned by compute_and_save_aggregate_grid().
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