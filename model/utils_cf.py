import numpy as np
import os
from datetime import datetime
import logging
import model.utils

logger = logging.getLogger(__name__)


def make_decision_cf(user, neighbors, cf_type, weight_by_similarity=True):
    """
    Drive a single mobility step for a CF-augmented user.

    Parameters
    ----------
    user                 : User
    neighbors            : list  - pre-built CF neighbor structure for this user
    cf_type              : "uk" | "ik" | "mf"
    weight_by_similarity : bool (default True)
        Passed through to the exploration functions.
        True  → venue sampling is weighted by similarity / recommendation score
        False → flat merge; all candidate venues are sampled uniformly
    """
    P_new = user.simulation.ro * user.num_distinct_venues ** (-user.simulation.gamma)

    if np.random.random_sample() >= P_new:
        user.returning()               # classic EPR return
    else:
        if cf_type == "uk":
            if not _uk_exploring(user, neighbors,
                                 weight_by_similarity=weight_by_similarity):
                user.exploring()           # EPR fallback
        elif cf_type == "ik":
            if not _venue_list_exploring(user, neighbors, cf_label="IK"):
                user.exploring()           # EPR fallback
        elif cf_type == "mf":
            if not _venue_list_exploring(user, neighbors, cf_label="MF"):
                user.exploring()           # EPR fallback
        elif cf_type == "ba":
            if not _ba_exploring(user, neighbors):
                user.exploring()           # EPR fallback
        else:
            raise ValueError(f"Unknown cf_type '{cf_type}'. Expected 'uk', 'ik', 'mf', or 'ba'.")

    _advance_clock(user)


def build_cf_data(users_list, top_k, cf_type,
                  simulate_params, grid_exists, grid_params,
                  check_run_dir, interval_idx=0):
    """
    Build neighbor / similarity structures for every user, depending on cf_type.

    Parameters
    ----------
    users_list      : list of User
    top_k           : int  - max neighbors to retain per user
    cf_type         : "uk" | "ik" | "mf"
    simulate_params : dict - used only for check/ folder naming
    grid_exists     : bool
    grid_params     : dict | None
    check_run_dir   : str | None - pass None on first call; returned path reused afterwards
    interval_idx    : int  - 0-based interval index for file naming

    Returns
    -------
    cf_neighbors   : list  - one entry per user; semantics depend on cf_type
    check_run_dir  : str   - folder path (created on first call, reused afterwards)
    """
    if cf_type == "uk":
        return _build_uk_data(users_list, top_k, simulate_params,
                              grid_exists, grid_params, check_run_dir, interval_idx)
    elif cf_type == "ik":
        return _build_ik_data(users_list, top_k, simulate_params,
                              grid_exists, grid_params, check_run_dir, interval_idx)
    elif cf_type == "mf":
        return _build_mf_data(users_list, top_k, simulate_params,
                              grid_exists, grid_params, check_run_dir, interval_idx)
    elif cf_type == "ba":
        return _build_ba_data(users_list, simulate_params,
                              grid_exists, grid_params, check_run_dir, interval_idx)
    else:
        raise ValueError(f"Unknown cf_type '{cf_type}'. Expected 'uk', 'ik', 'mf', or 'ba'.")


# =============================================================================
# SHARED HELPERS
# =============================================================================


def _resolve_check_run_dir(check_run_dir, cf_type, simulate_params,
                           grid_exists, grid_params):
    """
    Resolve (or create) the per-run check subfolder.

    Folder hierarchy:
        check/
            <cf_type>_ro_X_g_Y_a_Z_b_W_t_T[_cl_CL_cn_CN]_rs_R_ef_E/
                YYYYMMDD_HHMMSS_ffffff/
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

    batch_ts      = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    check_run_dir = os.path.join(check_base, base, batch_ts)
    os.makedirs(check_run_dir, exist_ok=True)
    logger.info(f"Check folder for this run: {check_run_dir}")
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


def _build_mega_user_uk(neighbors, my_venues, weight_by_similarity=True):
    """
    Merge the venue histories of the k similar User neighbors into a single
    virtual user (the "mega-user") for UK exploration.

    Venues already visited by the current user are excluded.

    Parameters
    ----------
    neighbors             : list of (User, similarity_score)
    my_venues             : set of venue names already visited by the current user
    weight_by_similarity  : bool (default True)
        True  → mega_weight[venue] += similarity * visit_count
        False → mega_weight[venue] += visit_count  (flat merge, similarity ignored)

    Returns
    -------
    venue_rows   : list of venue rows [name, count, x, y]
    mega_weights : np.ndarray of aggregated counts (same order)
    """
    aggregated = {}   # venue_name -> [row, accumulated_weight]

    for neighbor_user, sim in neighbors:
        contribution_factor = sim if weight_by_similarity else 1.0
        for row in neighbor_user.venue_history[1:]:
            vname = row[0]
            if vname in my_venues:
                continue
            w = contribution_factor * row[1]
            if vname in aggregated:
                aggregated[vname][1] += w
            else:
                aggregated[vname] = [row, w]

    if not aggregated:
        return [], np.array([], dtype=float)

    venue_rows   = [v[0] for v in aggregated.values()]
    mega_weights = np.array([v[1] for v in aggregated.values()], dtype=float)
    return venue_rows, mega_weights


def _build_mega_user_venue_list(neighbors, my_venues):
    """
    Build the mega-user for IK and MF exploration.

    IK and MF pre-compute a flat list of (venue_row, score) tuples where
    each score already encodes the item-item or latent-factor recommendation
    strength. Sampling is always weighted by these scores — no flat option.
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


def _uk_exploring(user, neighbors, weight_by_similarity=True):
    """
    UK exploration step using the mega-user strategy.

    The k similar neighbors are merged into a single virtual user and a
    venue is sampled proportionally to the aggregated weights.

    Parameters
    ----------
    weight_by_similarity : bool (default True)
        True  → weight[venue] = sum( similarity * visit_count )
        False → weight[venue] = sum( visit_count )  (flat merge)

    Returns False to trigger EPR fallback when no valid venues remain.

    On success sets: state="E", num_distinct_venues, next_position,
    destination, _venue_index entry, delta_r=None, angle=None.
    """
    if not neighbors:
        logger.warning(f"User {user.uid}: UK exploration skipped - no neighbors.")
        return False

    weights = np.array([w for _, w in neighbors], dtype=float)
    if weights.sum() == 0:
        logger.warning(f"User {user.uid}: UK exploration skipped - zero similarity weights.")
        return False

    my_venues = set(user._venue_index.keys())
    venue_rows, mega_weights = _build_mega_user_uk(neighbors, my_venues,
                                                   weight_by_similarity=weight_by_similarity)

    if len(venue_rows) == 0:
        logger.warning(f"User {user.uid}: UK exploration - EPR fallback (no valid venues in mega-user).")
        return False

    probs      = mega_weights / mega_weights.sum()
    chosen_idx = np.random.choice(len(venue_rows), p=probs)
    chosen     = venue_rows[chosen_idx]

    _set_user_exploration_state(user, chosen)
    return True


def _venue_list_exploring(user, neighbors, cf_label):
    """
    Shared exploration step for IK and MF.

    Sampling is always weighted by the pre-computed recommendation scores.
    Returns False to trigger EPR fallback when no valid venues remain.

    On success sets: state="E", num_distinct_venues, next_position,
    destination, _venue_index entry, delta_r=None, angle=None.
    """
    if not neighbors:
        logger.warning(f"User {user.uid}: {cf_label} exploration skipped - no recommended venues.")
        return False

    my_venues = set(user._venue_index.keys())
    venue_rows, mega_weights = _build_mega_user_venue_list(neighbors, my_venues)
    #Mega user needed to exlude venues visited since last rebuild

    if len(venue_rows) == 0:
        logger.warning(f"User {user.uid}: {cf_label} exploration - all recommended venues already visited.")
        return False

    probs      = mega_weights / mega_weights.sum()
    chosen_idx = np.random.choice(len(venue_rows), p=probs)
    chosen     = venue_rows[chosen_idx]

    _set_user_exploration_state(user, chosen)
    return True


# =============================================================================
# USER-KNN  (cf_type = "uk")
# =============================================================================


def _build_uk_data(users_list, top_k, simulate_params,
                   grid_exists, grid_params, check_run_dir, interval_idx):

    n_users = len(users_list)

    # ------------------------------------------------------------------
    # Build user-venue matrix
    # ------------------------------------------------------------------
    uv_matrix, all_venues, venue_to_idx, _ = _build_uv_matrix(users_list)
    n_venues = len(all_venues)

    # ------------------------------------------------------------------
    # Cosine similarity (user x user)
    # ------------------------------------------------------------------
    norms = np.linalg.norm(uv_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    uv_norm    = uv_matrix / norms
    sim_matrix = uv_norm @ uv_norm.T
    np.fill_diagonal(sim_matrix, 0.0)

    # ------------------------------------------------------------------
    # Resolve check folder
    # ------------------------------------------------------------------
    check_run_dir = _resolve_check_run_dir(
        check_run_dir, "uk", simulate_params, grid_exists, grid_params
    )
    interval_label = interval_idx + 1

    sim_matrix_path = os.path.join(check_run_dir, f"sim_matrix_interval_{interval_label:02d}.txt")
    np.savetxt(sim_matrix_path, sim_matrix, fmt="%.6f")

    # ------------------------------------------------------------------
    # Build neighbor lists  (top_k similar users per user)
    # ------------------------------------------------------------------
    top_k_actual = min(top_k, n_users - 1)
    cf_neighbors = []
    for ui in range(n_users):
        sims     = sim_matrix[ui]
        pos_mask = sims > 0
        if not pos_mask.any():
            cf_neighbors.append([])
            continue
        pos_indices = np.where(pos_mask)[0]
        if len(pos_indices) > top_k_actual:
            pos_indices = pos_indices[np.argsort(sims[pos_indices])[::-1]][:top_k_actual]
        else:
            pos_indices = pos_indices[np.argsort(sims[pos_indices])[::-1]]
        cf_neighbors.append([(users_list[j], float(sims[j])) for j in pos_indices])

    # ------------------------------------------------------------------
    # Persist neighbor list
    # ------------------------------------------------------------------
    neighbors_path = os.path.join(check_run_dir, f"neighbors_interval_{interval_label:02d}.txt")
    with open(neighbors_path, "w") as f:
        for row in cf_neighbors:
            f.write(",".join(map(str, row)) + "\n")

    # ------------------------------------------------------------------
    # Plot similarity distribution
    # ------------------------------------------------------------------
    try:
        import model.utils_plot
        model.utils_plot.plot_cf_neighbor_similarity_distribution(
            cf_neighbors,
            check_run_dir=check_run_dir,
            interval_idx=interval_idx,
            top_n_ranks=3,
        )
    except Exception as e:
        logger.warning(f"UK similarity plot failed: {e}")

    logger.info(
        f"UK CF data built: {n_users} users, {n_venues} unique venues, "
        f"top_k={top_k_actual}, interval={interval_label}"
    )
    return cf_neighbors, check_run_dir


# =============================================================================
# ITEM-KNN  (cf_type = "ik")
# =============================================================================


def _build_ik_data(users_list, top_k, simulate_params,
                   grid_exists, grid_params, check_run_dir, interval_idx):

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
    # Resolve check folder
    # ------------------------------------------------------------------
    check_run_dir = _resolve_check_run_dir(
        check_run_dir, "ik", simulate_params, grid_exists, grid_params
    )
    interval_label = interval_idx + 1

    sim_matrix_path = os.path.join(check_run_dir, f"sim_matrix_interval_{interval_label:02d}.txt")
    np.savetxt(sim_matrix_path, sim_items, fmt="%.6f")

    # ------------------------------------------------------------------
    # For each user we first compute the scores for the k-most similar venues, normalized by the sum
    #of the k-similarities (as 2.15 in Aggarwal's book). Then we keep only the top_k venues with positive scores as neighbors
    # ------------------------------------------------------------------
    cf_neighbors = []
    for ui, user in enumerate(users_list):
        user_vec     = uv_matrix[ui]
        visited_mask = user_vec > 0
        visited_idx  = np.where(visited_mask)[0]

        if not visited_mask.any():
            logger.warning(f"User {ui} has no visited venues, skipping recommendations.")
            cf_neighbors.append([])
            continue

        unvisited_idx = np.where(~visited_mask)[0]
        scores = np.zeros(n_venues)

        for vi in unvisited_idx:
            sims_to_visited = sim_items[vi, visited_idx]
            k_neighbors = min(top_k, len(visited_idx))
            top_w_local = np.argsort(sims_to_visited)[::-1][:k_neighbors]
            #Per ora lo terrei così, come è nel libro, da capire se per noi ha senso
            #Anche perché questo k non ha lo stesso ruolo del k successivo.
            #Il primo k serve per calcolare gli scores, il secondo serve per limitare le raccomandazioni

            sim_vals  = sims_to_visited[top_w_local]
            rate_vals = user_vec[visited_idx[top_w_local]]

            denom = np.sum(np.abs(sim_vals))
            if denom > 0:
                scores[vi] = np.dot(sim_vals, rate_vals) / denom
                #logger.info(f"User {ui}, venue {vi}: score={scores[vi]:.4f} (denom={denom:.4f})")
            #else:
                #logger.warning(f"User {ui}, venue {vi}: zero denominator, score set to 0.")

        positive_mask = scores > 0
        if not positive_mask.any():
            logger.warning(f"User {ui} has visited venues but no positive-score recommendations.")
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
    # Persist neighbor list
    # ------------------------------------------------------------------
    neighbors_path = os.path.join(check_run_dir, f"neighbors_interval_{interval_label:02d}.txt")
    with open(neighbors_path, "w") as f:
        for row in cf_neighbors:
            f.write(",".join(str(entry) for entry in row) + "\n")

    logger.info(
        f"IK CF data built: {n_users} users, {n_venues} unique venues, "
        f"top_k={top_k}, interval={interval_label}"
    )
    return cf_neighbors, check_run_dir


# =============================================================================
# MATRIX FACTORIZATION  (cf_type = "mf")
# =============================================================================


def _build_mf_data(users_list, top_k, simulate_params,
                   grid_exists, grid_params, check_run_dir, interval_idx,
                   num_latent_factors=20):
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
    logger.info(f"Performing truncated SVD with k={k} latent factors...")

    U, sigma, VT = np.linalg.svd(uv_matrix, full_matrices=False)
    sigma_diag   = np.diag(sigma[:k])
    user_factors = U[:, :k] @ sigma_diag    # (n_users, k)
    item_factors = VT[:k, :].T              # (n_venues, k)

    # ------------------------------------------------------------------
    # Predicted rating matrix  (n_users x n_venues)
    # ------------------------------------------------------------------
    pred_matrix = user_factors @ item_factors.T

    # ------------------------------------------------------------------
    # Resolve check folder and persist singular values for diagnostics
    # ------------------------------------------------------------------
    check_run_dir = _resolve_check_run_dir(
        check_run_dir, "mf", simulate_params, grid_exists, grid_params
    )
    interval_label = interval_idx + 1

    sigma_path = os.path.join(check_run_dir, f"singular_values_interval_{interval_label:02d}.txt")
    np.savetxt(sigma_path, sigma[:k].reshape(1, -1), fmt="%.6f",
               header=f"Top-{k} singular values")

    # ------------------------------------------------------------------
    # For each user, rank unvisited venues by predicted score, keep top_k
    # ------------------------------------------------------------------
    cf_neighbors = []
    for ui, user in enumerate(users_list):
        visited_mask = uv_matrix[ui] > 0

        if not visited_mask.any():
            logger.warning(f"User {ui} (MF): no visited venues - skipping recommendations.")
            cf_neighbors.append([])
            continue

        scores = pred_matrix[ui].copy()
        scores[visited_mask] = 0.0

        positive_mask = scores > 0
        if not positive_mask.any():
            logger.warning(f"User {ui} (MF): no positive-score unvisited venues.")
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
    # Persist neighbor list for inspection
    # ------------------------------------------------------------------
    neighbors_path = os.path.join(check_run_dir, f"neighbors_interval_{interval_label:02d}.txt")
    with open(neighbors_path, "w") as f:
        for row in cf_neighbors:
            f.write(",".join(str(entry) for entry in row) + "\n")

    logger.info(
        f"MF CF data built: {n_users} users, {n_venues} unique venues, "
        f"k={k}, top_k={top_k}, interval={interval_label}"
    )
    return cf_neighbors, check_run_dir


# =============================================================================
# BARABASI-ALBERT  (cf_type = "ba")
# =============================================================================


def _build_ba_data(users_list, simulate_params,
                   grid_exists, grid_params, check_run_dir, interval_idx):
    """
    Build Barabási-Albert popularity weights for exploration.

    For each venue (cell) known across all users, count how many *distinct*
    users have visited it at least once.  Normalise to obtain a probability
    vector.  This vector is the same for all users; individual filtering of
    already-visited venues happens at exploration time inside _ba_exploring().

    Parameters
    ----------
    users_list      : list of User
    simulate_params : dict  - used only for folder naming
    grid_exists     : bool
    grid_params     : dict | None
    check_run_dir   : str | None
    interval_idx    : int

    Returns
    -------
    cf_neighbors  : list of dict  - one entry per user, all pointing to the same
                    ba_data dict {venue_name: (x, y, raw_count, probability)}.
                    Wrapped in a list to match the interface expected by resume().
    check_run_dir : str
    """
    # ------------------------------------------------------------------
    # Count distinct users per venue
    # ------------------------------------------------------------------
    venue_user_count = {}   # venue_name -> set of user indices that visited it
    venue_meta       = {}   # venue_name -> (x, y)

    for ui, user in enumerate(users_list):
        for row in user.venue_history[1:]:
            vname = row[0]
            if vname not in venue_user_count:
                venue_user_count[vname] = set()
                venue_meta[vname]       = (row[2], row[3])
            venue_user_count[vname].add(ui)

    if not venue_user_count:
        # No data yet — return empty structure; _ba_exploring will fall back to EPR
        return [{} for _ in users_list], check_run_dir

    # ------------------------------------------------------------------
    # Build raw counts and normalise
    # ------------------------------------------------------------------
    venues      = list(venue_user_count.keys())
    raw_counts  = np.array([len(venue_user_count[v]) for v in venues], dtype=float)
    total       = raw_counts.sum()
    probs       = raw_counts / total

    ba_data = {
        v: (venue_meta[v][0], venue_meta[v][1], int(raw_counts[i]), float(probs[i]))
        for i, v in enumerate(venues)
    }

    # ------------------------------------------------------------------
    # Resolve check folder and persist for diagnostics
    # ------------------------------------------------------------------
    check_run_dir  = _resolve_check_run_dir(
        check_run_dir, "ba", simulate_params, grid_exists, grid_params
    )
    interval_label = interval_idx + 1

    popularity_path = os.path.join(
        check_run_dir, f"ba_popularity_interval_{interval_label:02d}.txt"
    )
    with open(popularity_path, "w") as f:
        f.write("venue_name,x,y,distinct_users,probability\n")
        for v, (x, y, cnt, p) in ba_data.items():
            f.write(f"{v},{x},{y},{cnt},{p:.8f}\n")

    n_venues = len(venues)
    n_users  = len(users_list)
    logger.info(
        f"BA data built: {n_users} users, {n_venues} unique venues, "
        f"interval={interval_label}"
    )
    # ba_data is the same for every user; wrap it in a list so that
    # resume() can index it as cf_neighbors[i], consistent with uk/ik/mf.
    cf_neighbors = [ba_data] * n_users
    return cf_neighbors, check_run_dir


def _ba_exploring(user, ba_data):
    """
    Barabási-Albert exploration step.

    Sample an unvisited venue proportionally to its global popularity
    (number of distinct users who have visited it).  Venues already
    visited by *this* user are excluded and the remaining probabilities
    are re-normalised before sampling.

    Returns False to trigger EPR fallback when no unvisited venues remain
    in ba_data (e.g. on the very first interval when data is sparse).

    On success sets: state="E", num_distinct_venues, next_position,
    destination, _venue_index entry, delta_r, angle=None.
    """
    if not ba_data:
        logger.warning(f"User {user.uid}: BA exploration skipped - no popularity data.")
        return False

    my_venues = set(user._venue_index.keys())

    # Filter to unvisited venues only
    candidates = [(v, x, y, p) for v, (x, y, cnt, p) in ba_data.items()
                  if v not in my_venues]

    if not candidates:
        logger.warning(
            f"User {user.uid}: BA exploration - EPR fallback "
            "(all known venues already visited)."
        )
        return False

    names  = [c[0] for c in candidates]
    xs     = [c[1] for c in candidates]
    ys     = [c[2] for c in candidates]
    raw_p  = np.array([c[3] for c in candidates], dtype=float)
    probs  = raw_p / raw_p.sum()   # re-normalise after filtering

    chosen_idx = np.random.choice(len(names), p=probs)
    chosen = [names[chosen_idx], 1, xs[chosen_idx], ys[chosen_idx]]

    _set_user_exploration_state(user, chosen)
    return True




def _advance_clock(user):
    """Sample delta_t, advance position and clock, update histories and grid."""
    user.delta_t           = model.utils.temporal_power_law_sampling(par=user.b_param).item()
    user.previous_position = user.position
    user.position          = user.next_position
    user.departure         = user.user_history[-1][user.user_history[0].index("Destination")]
    user.clock             = user.clock + user.delta_t
    user.update_histories()
    if user.has_grid:
        user.grid.update_cell_value_given_position(user.position)