import numpy as np
import os
from datetime import datetime
import logging
import model.utils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic-dump switch.
#
# Historically each CF interval dumped the FULL similarity / user-venue matrix
# (sim_matrix_*.txt), the per-user neighbour lists (neighbors_*.txt) and the
# MF singular values into a separate `check/` tree. The
# sim_matrix files are O(n_venues^2) / O(n_users^2) and dominate disk usage.
#
# When False (default) those heavy artifacts are NOT written; instead a compact
# one-line summary per interval is logged (and therefore captured into the
# per-run run.log next to the data). Set to True to restore the legacy dumps.
# ---------------------------------------------------------------------------
SAVE_CHECK_ARTIFACTS = False


# ---------------------------------------------------------------------------
# The two neighbourhood methods have two parameters (thesis, Section 3.2):
#   top_k   (k_top) length of the recommendation list;
#   score_k (k)     number of terms in the sum that defines the score:
#                   the peer group of each venue for User-kNN ("uk"), the
#                   visited venues used to score a candidate for Item-kNN ("ik").
# User-KNN follows Aggarwal (2016), Section 2.3.1: the score of a candidate
# venue j for user u is computed from the peer group P_u(j) = the score_k users
# most similar to u AMONG THOSE WHO VISITED j (cosine similarity on the visit
# counts), without normalisation:
#       score(u, j) = sum_{v in P_u(j)} sim(u, v) * r_vj
# ---------------------------------------------------------------------------
SCORE_K_DEFAULT = 25


# ---------------------------------------------------------------------------
# Recommender-health diagnostic
#
# A cheap, per-interval check of whether a recommender is actually
# discriminating between venues or handing them out almost at random. It is
# computed straight from the per-user recommendation lists (cf_neighbors), so
# it is independent of any downstream outcome metric, and it works the same for
# uk, ik and mf (each cf_neighbors entry is a
# (venue_row, score) pair, with the recommendation score of the venue).
#
# Metrics (aggregated over users, reported per interval):
#   * coverage             — % of users that received a non-empty list, i.e.
#                            the recommender had something to say for them.
#   * peakiness            — median over users of (1 - normalised entropy) of
#                            each user's score vector. It measures how
#                            concentrated the scores are:
#                              0  = perfectly flat scores  -> the choice is
#                                   ~uniform among candidates == NEAR-RANDOM;
#                              1  = a single dominant candidate == a decisive,
#                                   strongly discriminating recommendation.
#                            This is the key "recommending vs guessing" number
#                            and, being scale-free, is comparable across
#                            recommender families.
#   * weakest_score_median — median over users of the smallest score they use,
#                            i.e. the weakest recommended venue score.
#   * top_score_median     — median over users of their largest score (the
#                            strength of the single best recommendation).
# ---------------------------------------------------------------------------


def compute_recommender_health(cf_neighbors):
    """Return the recommender-health dict for one CF build (see module comment
    above for the meaning of each field), or None when the lists carry no usable
    scores (e.g. the "ra" dict payload)."""
    peak, weak, top = [], [], []
    n_nonempty = 0
    n = len(cf_neighbors)
    for lst in cf_neighbors:
        if not isinstance(lst, (list, tuple)) or len(lst) == 0:
            continue
        try:
            s = np.asarray([float(e[1]) for e in lst], dtype=float)
        except (TypeError, IndexError, KeyError, ValueError):
            return None                       # not a scored list -> skip family
        s = s[np.isfinite(s)]
        if s.size == 0 or s.sum() <= 0:
            continue
        n_nonempty += 1
        top.append(float(s.max()))
        weak.append(float(s.min()))
        if s.size == 1:
            peak.append(1.0)
        else:
            p = s / s.sum()
            H = -np.sum(p * np.log(p + 1e-12))
            peak.append(1.0 - H / np.log(s.size))
    if n_nonempty == 0:
        return None
    peak = np.asarray(peak)
    health = {
        "coverage":             100.0 * n_nonempty / max(n, 1),
        "peakiness_median":     float(np.median(peak)),
        "peakiness_p25":        float(np.percentile(peak, 25)),
        "peakiness_p75":        float(np.percentile(peak, 75)),
        "weakest_score_median": float(np.median(weak)),
        "top_score_median":     float(np.median(top)),
    }
    return health


def _log_recommender_health(cf_tag, interval_label, cf_neighbors):
    """Compute + log the recommender-health line ."""
    h = compute_recommender_health(cf_neighbors)
    if h is None:
        return
    logger.debug(
        f"[{cf_tag}] interval {interval_label}: recommender health — "
        f"recs for {h['coverage']:.0f}% of users | signal peakiness median "
        f"{h['peakiness_median']:.3f} (p25 {h['peakiness_p25']:.3f}, "
        f"p75 {h['peakiness_p75']:.3f}) | weakest used score median "
        f"{h['weakest_score_median']:.3f} | top score median {h['top_score_median']:.3f}"
    )


# =============================================================================
# CF per-interval counters
#
# Instead of emitting one log line per user per fallback (thousands of lines
# per interval), each exploration outcome is tallied on the live User object
# in `user._cf_counts`.  The worker resets this dict at the start of every
# interval and resume() aggregates it across all users into a single concise
# summary line per interval.
#
# Reasons:
#   _CF_SUCCESS    : CF exploration succeeded (a venue was recommended & taken)
#   _CF_NO_RECS    : the recommender produced no usable data for this user
#                    (no neighbors / empty recommendation list / no popularity
#                    data / empty catalogue / zero similarity weights)
#   _CF_EXHAUSTED  : data existed, but every candidate venue had already been
#                    visited by this user (e.g. the top_k list was consumed
#                    within the interval) -> EPR fallback
# =============================================================================
_CF_SUCCESS   = "success"
_CF_NO_RECS   = "fb_no_recs"
_CF_EXHAUSTED = "fb_exhausted"

def _bump(user, key):
    """Increment user._cf_counts[key], initialising the dict if needed."""
    counts = getattr(user, "_cf_counts", None)
    if counts is None:
        counts = {}
        user._cf_counts = counts
    counts[key] = counts.get(key, 0) + 1


def make_decision_cf(user, neighbors, cf_type):
    """
    Drive a single mobility step for a CF-augmented user.

    Parameters
    ----------
    user                 : User
    neighbors            : list  - pre-built CF neighbor structure for this user
    cf_type              : "uk" | "ik" | "mf" | "ra".
        "uk" is User-KNN with a per-venue peer group (Aggarwal) and a
        truncated recommendation list; it explores like "ik" and "mf".
    """
    P_new = user.simulation.ro * user.num_distinct_venues ** (-user.simulation.gamma)

    _explore = np.random.random_sample() < P_new

    if not _explore:
        user.returning()               # classic EPR return
    else:
        if cf_type in ("uk", "ik", "mf"):
            _ok = _venue_list_exploring(user, neighbors)
        elif cf_type == "ra":
            _ok = _ra_exploring(user, neighbors)
        else:
            raise ValueError(f"Unknown cf_type '{cf_type}'. Expected 'uk', 'ik', 'mf', 'ra'.")

        if not _ok:
            # The recommender could not place a venue; fall back to plain EPR.
            user.exploring()

    _advance_clock(user)


def build_cf_data(users_list, top_k, cf_type,
                  simulate_params, grid_exists, grid_params,
                  check_run_dir, interval_idx=0, batch_ts=None,
                  num_latent_factors=20, score_k=SCORE_K_DEFAULT):
    """
    Build neighbor / similarity structures for every user, depending on cf_type.

    Parameters
    ----------
    users_list      : list of User
    top_k           : int  - max neighbors to retain per user
    cf_type         : "uk" | "ik" | "mf" | ...
    simulate_params : dict - used only for check/ folder naming
    grid_exists     : bool
    grid_params     : dict | None
    check_run_dir   : str | None - pass None on first call; returned path reused afterwards
    interval_idx    : int  - 0-based interval index for file naming
    batch_ts        : str | None - timestamp shared with the raw/analysis batch so that
                      the check folder receives the same ID.  Generated fresh when None.
    num_latent_factors : int - truncated-SVD rank for "mf" (A2). Ignored by the
                      other cf_types. Defaults to 20 (the historical hardcoded value).
    score_k         : int - number of terms in the score of "uk" / "ik" (k).

    Returns
    -------
    cf_neighbors   : list  - one entry per user; semantics depend on cf_type
    check_run_dir  : str   - folder path (created on first call, reused afterwards)
    """
    if cf_type == "uk":
        return _build_uk_data(users_list, top_k, simulate_params,
                              grid_exists, grid_params, check_run_dir, interval_idx,
                              batch_ts=batch_ts, cf_tag=cf_type,
                              score_k=score_k)
    elif cf_type == "ik":
        return _build_ik_data(users_list, top_k, simulate_params,
                              grid_exists, grid_params, check_run_dir, interval_idx,
                              batch_ts=batch_ts, cf_tag=cf_type,
                              score_k=score_k)
    elif cf_type == "mf":
        return _build_mf_data(users_list, top_k, simulate_params,
                              grid_exists, grid_params, check_run_dir, interval_idx,
                              num_latent_factors=num_latent_factors,
                              batch_ts=batch_ts, cf_tag=cf_type)
    elif cf_type == "ra":
        return _build_ra_data(users_list, simulate_params,
                              grid_exists, grid_params, check_run_dir, interval_idx,
                              batch_ts=batch_ts)
    else:
        raise ValueError(f"Unknown cf_type '{cf_type}'. Expected 'uk', 'ik', 'mf', 'ra'.")



def _resolve_check_run_dir(check_run_dir, cf_type, simulate_params,
                           grid_exists, grid_params, batch_ts=None):
    """
    Resolve (or create) the per-run check subfolder.

    Folder hierarchy:
        check/
            <cf_type>_ro_X_g_Y_a_Z_b_W_t_T[_cl_CL_cn_CN]_rs_R_ef_E/
                YYYYMMDD_HHMMSS_ffffff/

    Parameters
    ----------
    batch_ts : str, optional
        Timestamp string in the format "YYYYMMDD_HHMMSS_ffffff".
        When provided, this exact string is used as the batch folder name
        so that the check folder shares the same ID as the raw/analysis
        batch that generated the data.  When None, a fresh timestamp is
        generated (legacy behaviour).
    """
    if check_run_dir is not None:
        return check_run_dir

    root_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    check_base = os.path.join(root_dir, "check")

    p      = simulate_params
    efrac  = float(p.get("epr_fraction", 1.0))
    rstart = p.get("random_start", False)

    base = (
        f"{cf_type}_ro_{p['ro']}_g_{p['gamma']}"
        f"_a_{p['a_param']}_b_{p['b_param']}_t_{p['time']}"
    )
    if grid_exists and grid_params is not None:
        base += f"_cl_{grid_params['cell_lenght']}_cn_{grid_params['cell_number']}"
    base += f"_rs_{'y' if rstart else 'n'}_ef_{efrac}"

    if batch_ts is None:
        batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    check_run_dir = os.path.join(check_base, base, batch_ts)
    os.makedirs(check_run_dir, exist_ok=True)
    logger.debug(f"Check folder for this run: {check_run_dir}")
    return check_run_dir


def _build_uv_matrix(users_list):
    """
    Build the user-venue visit-count matrix and associated index structures.

    Iterates venue_history once per user to collect the global venue list,
    coordinates metadata, and fill the count matrix.

    Returns
    -------
    uv_matrix    : np.ndarray, shape (n_users, n_venues)
    all_venues   : list of venue names (column order)
    venue_to_idx : dict {venue_name: col_index}
    venue_meta   : dict {venue_name: (x, y)}
    """
    all_venues, venue_set = [], set()
    venue_meta = {}
    for user in users_list:
        for row in user.venue_history[1:]:
            if row[0] not in venue_set:
                venue_set.add(row[0])
                all_venues.append(row[0])
                venue_meta[row[0]] = (row[2], row[3])

    venue_to_idx = {v: i for i, v in enumerate(all_venues)}
    n_venues     = len(all_venues)
    n_users      = len(users_list)

    uv_matrix = np.zeros((n_users, n_venues), dtype=float)
    for ui, user in enumerate(users_list):
        for row in user.venue_history[1:]:
            uv_matrix[ui, venue_to_idx[row[0]]] = row[1]

    return uv_matrix, all_venues, venue_to_idx, venue_meta


def _set_user_exploration_state(user, chosen):
    """
    Apply an exploration outcome to the user object.

    `chosen` must be a sequence with at minimum:
        chosen[0]  : venue name
        chosen[2]  : x coordinate
        chosen[3]  : y coordinate
    """
    user.state = "E"
    user.num_distinct_venues += 1
    user.next_position = (chosen[2], chosen[3])
    user.destination   = chosen[0]
    user._venue_index[user.destination] = len(user.venue_history)
    dx = chosen[2] - user.position[0]
    dy = chosen[3] - user.position[1]
    user.delta_r = float(np.sqrt(dx * dx + dy * dy))
    user.angle   = None


def _build_mega_user_venue_list(neighbors, my_venues):
    """
    Build the mega-user for UK, IK and MF exploration.

    UK, IK and MF pre-compute a flat list of (venue_row, score) tuples where
    each score already encodes the user-user, item-item or latent-factor
    recommendation strength. Sampling is always weighted by these scores — no flat option.
    Venues visited since the last rebuild are filtered out here.

    Parameters
    ----------
    neighbors  : list of ([vname, 1, x, y], score)
    my_venues  : set of venue names already visited by the current user

    Returns
    -------
    venue_rows   : list of venue rows
    mega_weights : np.ndarray of scores
    """
    valid = [(row, score) for row, score in neighbors if row[0] not in my_venues]
    if not valid:
        return [], np.array([], dtype=float)

    venue_rows   = [r for r, _ in valid]
    mega_weights = np.array([s for _, s in valid], dtype=float)
    return venue_rows, mega_weights


# =============================================================================
# EXPLORING FUNCTIONS
# =============================================================================


def _venue_list_exploring(user, neighbors):
    """
    Shared exploration step for UK, IK and MF.

    Sampling is always weighted by the pre-computed recommendation scores.
    Returns False to trigger EPR fallback when no valid venues remain.

    On success sets: state="E", num_distinct_venues, next_position,
    destination, _venue_index entry, delta_r=None, angle=None.
    """
    if not neighbors:
        _bump(user, _CF_NO_RECS)
        return False

    my_venues = set(user._venue_index.keys())
    venue_rows, mega_weights = _build_mega_user_venue_list(neighbors, my_venues)
    #Mega user needed to exlude venues visited since last rebuild

    if len(venue_rows) == 0:
        _bump(user, _CF_EXHAUSTED)
        return False

    probs      = mega_weights / mega_weights.sum()
    chosen_idx = np.random.choice(len(venue_rows), p=probs)
    chosen     = venue_rows[chosen_idx]

    _set_user_exploration_state(user, chosen)
    _bump(user, _CF_SUCCESS)
    return True


# =============================================================================
# USER-KNN  (cf_type = "uk")
# =============================================================================

# Rows (target users) processed together in _build_uk_data: bounds the memory
# of the per-venue similarity block (UK_ROW_BLOCK x visitors of the venue).
UK_ROW_BLOCK = 500


def _uk_scores(sim, uv_matrix, k_peer, row_block=None):
    """
    Score matrix of the "uk" recommender (Aggarwal, Eq. 2.4, without
    normalisation and without mean-centring):

        score[u, j] = sum_{v in P_u(j)} sim[u, v] * uv_matrix[v, j]

    where P_u(j) are the k_peer users with the largest sim[u, v] among the
    users v who visited j (uv_matrix[v, j] > 0).  Users with zero similarity
    contribute nothing, so P_u(j) effectively contains only positively similar
    users; u itself is excluded because sim[u, u] = 0.

    Computed venue by venue: for venue j only the columns of its visitors are
    needed, and the top-k_peer of each row is found with argpartition.
    """
    if row_block is None:
        row_block = UK_ROW_BLOCK
    # float32 halves the memory traffic of the argpartition (about 1.5x faster);
    # the scores only rank and weight the candidates, so the precision is ample.
    sim       = np.asarray(sim, dtype=np.float32)
    uv_matrix = np.asarray(uv_matrix, dtype=np.float32)
    n_users, n_venues = uv_matrix.shape
    scores = np.zeros((n_users, n_venues), dtype=np.float64)
    for j in range(n_venues):
        vis = np.flatnonzero(uv_matrix[:, j])
        n_j = vis.size
        if n_j == 0:
            continue
        r_vis = uv_matrix[vis, j]
        if n_j <= k_peer:
            # every visitor is in the peer group
            scores[:, j] = sim[:, vis] @ r_vis
            continue
        for a in range(0, n_users, row_block):
            b = min(a + row_block, n_users)
            S = sim[a:b][:, vis]                                   # (rows, n_j)
            idx = np.argpartition(S, n_j - k_peer, axis=1)[:, n_j - k_peer:]
            top = np.take_along_axis(S, idx, axis=1)               # (rows, k_peer)
            scores[a:b, j] = np.einsum("ij,ij->i", top, r_vis[idx])
    return scores


def _build_uk_data(users_list, top_k, simulate_params,
                   grid_exists, grid_params, check_run_dir, interval_idx,
                   batch_ts=None, cf_tag="uk",
                   score_k=SCORE_K_DEFAULT):
    """
    Build the recommendation lists of the "uk" recommender: User-KNN with a
    per-venue peer group, as in Aggarwal (2016), Section 2.3.1.

    For each user u and each venue j not yet visited by u:
        P_u(j)      = the k most similar users to u among those who visited j
                      (k = score_k);
        score(u, j) = sum_{v in P_u(j)} sim(u, v) * r_vj      (no normalisation)
    with sim the cosine similarity between the visit-count vectors and r_vj
    the number of visits of v to j.  The list of each user keeps the top_k
    positive scores, in the same (venue_row, score) format as "ik" and "mf", and the exploration
    samples a venue with probability proportional to its score.
    """
    n_users = len(users_list)
    uv_matrix, all_venues, venue_to_idx, venue_meta = _build_uv_matrix(users_list)
    n_venues = len(all_venues)

    # ------------------------------------------------------------------
    # Cosine similarity (user x user)
    # ------------------------------------------------------------------
    norms = np.linalg.norm(uv_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    uv_norm    = uv_matrix / norms
    sim_matrix = uv_norm @ uv_norm.T
    np.fill_diagonal(sim_matrix, 0.0)
    sim_matrix[sim_matrix < 0] = 0.0          # (cannot happen with counts; safety)

    interval_label = interval_idx + 1
    if SAVE_CHECK_ARTIFACTS:
        check_run_dir = _resolve_check_run_dir(
            check_run_dir, cf_tag, simulate_params, grid_exists, grid_params,
            batch_ts=batch_ts
        )
    else:
        check_run_dir = None
    _off = sim_matrix[sim_matrix > 0]
    k_peer = int(score_k)
    logger.debug(
        f"[{cf_tag}] interval {interval_label}: user-user sim matrix "
        f"{n_users}x{n_users}, mean positive similarity "
        f"{float(_off.mean()) if _off.size else 0.0:.4f}; peer group per venue "
        f"k = {k_peer}, list length top_k = {top_k}"
    )

    # ------------------------------------------------------------------
    # Scores and lists
    # ------------------------------------------------------------------
    scores_all = _uk_scores(sim_matrix, uv_matrix, k_peer)

    cf_neighbors = []
    n_no_visited = 0
    n_no_pos     = 0
    for ui in range(n_users):
        visited_mask = uv_matrix[ui] > 0
        if not visited_mask.any():
            n_no_visited += 1
            cf_neighbors.append([])
            continue
        scores = scores_all[ui].copy()
        scores[visited_mask] = 0.0
        positive_mask = scores > 0
        if not positive_mask.any():
            n_no_pos += 1
            cf_neighbors.append([])
            continue
        pos_indices = np.where(positive_mask)[0]
        order = pos_indices[np.argsort(scores[pos_indices])[::-1]]
        top_indices = order[:min(top_k, len(order))]

        user_neighbors = []
        for vi in top_indices:
            vname = all_venues[vi]
            x, y  = venue_meta[vname]
            user_neighbors.append(([vname, 1, x, y], float(scores[vi])))
        cf_neighbors.append(user_neighbors)

    if SAVE_CHECK_ARTIFACTS and check_run_dir is not None:
        neighbors_path = os.path.join(check_run_dir, f"neighbors_interval_{interval_label:02d}.txt")
        with open(neighbors_path, "w") as f:
            for row in cf_neighbors:
                f.write(",".join(str(entry) for entry in row) + "\n")

    _log_recommender_health(cf_tag, interval_label, cf_neighbors)
    n_with_recs = n_users - n_no_visited - n_no_pos
    logger.debug(
        f"{cf_tag.upper()} CF data built [interval {interval_label}]: {n_users} users, "
        f"{n_venues} unique venues, k={k_peer}, top_k={top_k} -> "
        f"{n_with_recs} users with recommendations, "
        f"{n_no_visited} with no visited venues, "
        f"{n_no_pos} with no positive-score recommendations (empty list)."
    )
    return cf_neighbors, check_run_dir


# =============================================================================
# ITEM-KNN  (cf_type = "ik")
# =============================================================================


def _build_ik_data(users_list, top_k, simulate_params,
                   grid_exists, grid_params, check_run_dir, interval_idx,
                   batch_ts=None, cf_tag="ik",
                   score_k=SCORE_K_DEFAULT):
    n_users = len(users_list)

    # ------------------------------------------------------------------
    # Build user-venue matrix + metadata
    # ------------------------------------------------------------------
    uv_matrix, all_venues, venue_to_idx, venue_meta = _build_uv_matrix(users_list)
    n_venues = len(all_venues)

    # ------------------------------------------------------------------
    # Item-item cosine similarity 
    # ------------------------------------------------------------------
    norms = np.linalg.norm(uv_matrix, axis=0, keepdims=True)  
    norms[norms == 0] = 1.0
    M_norm = uv_matrix / norms     
    sim_items = M_norm.T @ M_norm   
    np.fill_diagonal(sim_items, 0.0)

    # ------------------------------------------------------------------
    # Resolve check folder (only when heavy diagnostics are enabled)
    # ------------------------------------------------------------------
    interval_label = interval_idx + 1
    if SAVE_CHECK_ARTIFACTS:
        check_run_dir = _resolve_check_run_dir(
            check_run_dir, cf_tag, simulate_params, grid_exists, grid_params,
            batch_ts=batch_ts
        )
        sim_matrix_path = os.path.join(check_run_dir, f"sim_matrix_interval_{interval_label:02d}.txt")
        np.savetxt(sim_matrix_path, sim_items, fmt="%.6f")
    else:
        check_run_dir = None
        _off = sim_items[sim_items > 0]
        logger.debug(
            f"[{cf_tag}] interval {interval_label}: item-item sim matrix "
            f"{sim_items.shape[0]}x{sim_items.shape[1]}, "
            f"mean positive similarity {float(_off.mean()) if _off.size else 0.0:.4f}"
        )

    # ------------------------------------------------------------------
    # For each user we first compute the score of every unvisited venue as the
    # weighted sum over the k most similar visited venues (Aggarwal, Eq. 2.15
    # without the normalisation). Then we keep only the top_k venues with
    # positive scores as neighbors
    # ------------------------------------------------------------------
    k_score = int(score_k)
    logger.debug(f"[{cf_tag}] interval {interval_label}: scoring neighbourhood "
                f"k = {k_score}, list length top_k = {top_k}")
    cf_neighbors = []
    n_no_visited = 0      # users with no visited venues at all
    n_no_pos     = 0      # users with visits but no positive-score recommendations
    for ui, user in enumerate(users_list):
        user_vec     = uv_matrix[ui]
        visited_mask = user_vec > 0
        visited_idx  = np.where(visited_mask)[0]

        if not visited_mask.any():
            n_no_visited += 1
            cf_neighbors.append([])
            continue

        unvisited_idx = np.where(~visited_mask)[0]
        scores = np.zeros(n_venues)

        for vi in unvisited_idx:
            sims_to_visited = sim_items[vi, visited_idx]
            k_neighbors = min(k_score, len(visited_idx))
            top_w_local = np.argsort(sims_to_visited)[::-1][:k_neighbors]
            #Per ora lo terrei così, come è nel libro, da capire se per noi ha senso
            #Anche perché questo k non ha lo stesso ruolo del k successivo.
            #Il primo k serve per calcolare gli scores, il secondo serve per limitare le raccomandazioni

            sim_vals  = sims_to_visited[top_w_local]
            rate_vals = user_vec[visited_idx[top_w_local]]

            scores[vi] = np.dot(sim_vals, rate_vals)

        positive_mask = scores > 0
        if not positive_mask.any():
            n_no_pos += 1
            cf_neighbors.append([])
            continue

        pos_indices = np.where(positive_mask)[0]
        k           = min(top_k, len(pos_indices))
        top_indices = pos_indices[np.argsort(scores[pos_indices])[::-1]][:k]

        user_neighbors = []
        for vi in top_indices:
            vname = all_venues[vi]
            x, y  = venue_meta[vname]
            user_neighbors.append(([vname, 1, x, y], float(scores[vi])))
        cf_neighbors.append(user_neighbors)

    # ------------------------------------------------------------------
    # Persist neighbor list (only heavy-diagnostics mode)
    # ------------------------------------------------------------------
    if SAVE_CHECK_ARTIFACTS and check_run_dir is not None:
        neighbors_path = os.path.join(check_run_dir, f"neighbors_interval_{interval_label:02d}.txt")
        with open(neighbors_path, "w") as f:
            for row in cf_neighbors:
                f.write(",".join(str(entry) for entry in row) + "\n")

    _log_recommender_health(cf_tag, interval_label, cf_neighbors)
    n_with_recs = n_users - n_no_visited - n_no_pos
    logger.debug(
        f"{cf_tag.upper()} CF data built [interval {interval_label}]: {n_users} users, "
        f"{n_venues} unique venues, top_k={top_k} -> "
        f"{n_with_recs} users with recommendations, "
        f"{n_no_visited} with no visited venues, "
        f"{n_no_pos} with no positive-score recommendations (empty list)."
    )
    return cf_neighbors, check_run_dir


# =============================================================================
# MATRIX FACTORIZATION  (cf_type = "mf")
# =============================================================================


def _build_mf_data(users_list, top_k, simulate_params,
                   grid_exists, grid_params, check_run_dir, interval_idx,
                   num_latent_factors=20, batch_ts=None, cf_tag="mf"):
    """
    Build MF-based recommendation lists for every user via truncated SVD.

    The user-venue visit-count matrix is decomposed as:
        M ≈ U_k · Σ_k · V_k^T

    User factors  = U_k · Σ_k   (shape: n_users  x k)
    Item factors  = V_k^T        (shape: n_venues x k, after transpose)

    Predicted score for user u on venue v:
        score(u, v) = user_factors[u] · item_factors[v]

    For each user, unvisited venues are ranked by predicted score and the
    top_k positive-score venues are stored as cf_neighbors, using the same
    format as IK: list of (venue_row, score) tuples.

    Parameters
    ----------
    num_latent_factors : int
        Number of singular vectors to keep (k).  Automatically capped at
        min(n_users, n_venues, num_latent_factors).
    """
    n_users = len(users_list)

    # ------------------------------------------------------------------
    # Build user-venue matrix + metadata
    # ------------------------------------------------------------------
    uv_matrix, all_venues, venue_to_idx, venue_meta = _build_uv_matrix(users_list)
    n_venues = len(all_venues)

    # ------------------------------------------------------------------
    # Truncated SVD
    # np.linalg.svd returns the full decomposition; we take only the
    # leading k components to capture the dominant latent structure.
    # ------------------------------------------------------------------
    k = min(num_latent_factors, n_users, n_venues)
    logger.debug(f"Performing truncated SVD with k={k} latent factors...")

    U, sigma, VT = np.linalg.svd(uv_matrix, full_matrices=False)
    sigma_diag   = np.diag(sigma[:k])
    user_factors = U[:, :k] @ sigma_diag    # (n_users, k)
    item_factors = VT[:k, :].T              # (n_venues, k)

    # ------------------------------------------------------------------
    # Predicted rating matrix  (n_users x n_venues)
    # ------------------------------------------------------------------
    pred_matrix = user_factors @ item_factors.T

    # ------------------------------------------------------------------
    # Resolve check folder and persist singular values (only heavy mode).
    # The top singular values are always logged (they are tiny and useful).
    # ------------------------------------------------------------------
    interval_label = interval_idx + 1
    _sv = np.array2string(sigma[:k], precision=3, separator=", ", max_line_width=200)
    logger.debug(f"[{cf_tag}] interval {interval_label}: top-{k} singular values {_sv}")
    if SAVE_CHECK_ARTIFACTS:
        check_run_dir = _resolve_check_run_dir(
            check_run_dir, cf_tag, simulate_params, grid_exists, grid_params,
            batch_ts=batch_ts
        )
        sigma_path = os.path.join(check_run_dir, f"singular_values_interval_{interval_label:02d}.txt")
        np.savetxt(sigma_path, sigma[:k].reshape(1, -1), fmt="%.6f",
                   header=f"Top-{k} singular values")
    else:
        check_run_dir = None

    # ------------------------------------------------------------------
    # For each user, rank unvisited venues by predicted score, keep top_k
    # ------------------------------------------------------------------
    cf_neighbors = []
    n_no_visited = 0      # users with no visited venues at all
    n_no_pos     = 0      # users with visits but no positive-score recommendations
    for ui, user in enumerate(users_list):
        visited_mask = uv_matrix[ui] > 0

        if not visited_mask.any():
            n_no_visited += 1
            cf_neighbors.append([])
            continue

        scores = pred_matrix[ui].copy()
        scores[visited_mask] = 0.0

        positive_mask = scores > 0
        if not positive_mask.any():
            n_no_pos += 1
            cf_neighbors.append([])
            continue

        pos_indices = np.where(positive_mask)[0]
        k_sel       = min(top_k, len(pos_indices))
        top_indices = pos_indices[np.argsort(scores[pos_indices])[::-1]][:k_sel]

        user_neighbors = []
        for vi in top_indices:
            vname = all_venues[vi]
            x, y  = venue_meta[vname]
            user_neighbors.append(([vname, 1, x, y], float(scores[vi])))
        cf_neighbors.append(user_neighbors)

    # ------------------------------------------------------------------
    # Persist neighbor list for inspection (only heavy-diagnostics mode)
    # ------------------------------------------------------------------
    if SAVE_CHECK_ARTIFACTS and check_run_dir is not None:
        neighbors_path = os.path.join(check_run_dir, f"neighbors_interval_{interval_label:02d}.txt")
        with open(neighbors_path, "w") as f:
            for row in cf_neighbors:
                f.write(",".join(str(entry) for entry in row) + "\n")

    _log_recommender_health(cf_tag, interval_label, cf_neighbors)
    n_with_recs = n_users - n_no_visited - n_no_pos
    logger.debug(
        f"{cf_tag.upper()} CF data built [interval {interval_label}]: {n_users} users, "
        f"{n_venues} unique venues, k={k}, top_k={top_k} -> "
        f"{n_with_recs} users with recommendations, "
        f"{n_no_visited} with no visited venues, "
        f"{n_no_pos} with no positive-score recommendations (empty list)."
    )
    return cf_neighbors, check_run_dir


# =============================================================================
# RANDOM  (cf_type = "ra")
# =============================================================================


def _build_ra_data(users_list, simulate_params,
                   grid_exists, grid_params, check_run_dir, interval_idx,
                   batch_ts=None):
    """
    Build the global venue catalogue for uniform-random exploration.

    Collects every venue ever visited by any user (name + coordinates).
    No counts or weights are needed: _ra_exploring samples uniformly
    among all unvisited venues.

    Returns
    -------
    cf_neighbors  : list of dict  - one entry per user, all pointing to
                    the same ra_data dict {venue_name: (x, y)}.
    check_run_dir : str
    """
    ra_data    = {}   # venue_name -> (x, y)

    for user in users_list:
        for row in user.venue_history[1:]:
            vname = row[0]
            if vname not in ra_data:
                ra_data[vname] = (row[2], row[3])

    if not ra_data:
        return [{} for _ in users_list], check_run_dir

    # ------------------------------------------------------------------
    # Persist catalogue (only heavy-diagnostics mode)
    # ------------------------------------------------------------------
    interval_label = interval_idx + 1
    if SAVE_CHECK_ARTIFACTS:
        check_run_dir  = _resolve_check_run_dir(
            check_run_dir, "ra", simulate_params, grid_exists, grid_params,
            batch_ts=batch_ts
        )
        catalogue_path = os.path.join(
            check_run_dir, f"ra_catalogue_interval_{interval_label:02d}.txt"
        )
        with open(catalogue_path, "w") as f:
            f.write("venue_name,x,y\n")
            for v, (x, y) in ra_data.items():
                f.write(f"{v},{x},{y}\n")
    else:
        check_run_dir = None

    n_venues = len(ra_data)
    n_users  = len(users_list)
    logger.debug(
        f"RA data built: {n_users} users, {n_venues} unique venues, "
        f"interval={interval_label}"
    )
    cf_neighbors = [ra_data] * n_users
    return cf_neighbors, check_run_dir


def _ra_exploring(user, ra_data):
    """
    Random exploration step.

    Uniformly samples one venue from the global catalogue that the user
    has not yet visited.

    Returns False to trigger EPR fallback when all known venues have
    already been visited by this user.
    """
    if not ra_data:
        _bump(user, _CF_NO_RECS)
        return False

    my_venues = set(user._venue_index.keys())

    candidates = [(v, x, y) for v, (x, y) in ra_data.items()
                  if v not in my_venues]

    if not candidates:
        _bump(user, _CF_EXHAUSTED)
        return False

    chosen_idx = np.random.randint(len(candidates))
    v, x, y    = candidates[chosen_idx]
    chosen     = [v, 1, x, y]

    _set_user_exploration_state(user, chosen)
    _bump(user, _CF_SUCCESS)
    return True




def _advance_clock(user):
    """Sample delta_t, advance position and clock, update histories and grid."""
    _sim   = user.simulation
    _tmin  = getattr(_sim, "temporal_min", 1)
    _tmax  = getattr(_sim, "temporal_max", 61200)
    user.delta_t           = model.utils.temporal_power_law_sampling(
        par=user.b_param, min_val=_tmin, max_val=_tmax).item()
    user.previous_position = user.position
    user.position          = user.next_position
    user.departure         = user.user_history[-1][user.user_history[0].index("Destination")]
    user.clock             = user.clock + user.delta_t
    user.update_histories()
    if user.has_grid:
        user.grid.update_cell_value_given_position(user.position)