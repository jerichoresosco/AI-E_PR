import numpy as np
import os
from datetime import datetime
import logging
import model.utils

logger = logging.getLogger(__name__)

def make_decision_cf(user, neighbors,cf_type):

    P_new = user.simulation.ro * user.num_distinct_venues ** (-user.simulation.gamma)

    if np.random.random_sample() >= P_new:
        user.returning()               # classic EPR return
    else:
        if cf_type == "uk":
            if not _uk_exploring(user, neighbors):
                user.exploring()           # EPR fallback
        elif cf_type == "ik":
            if not _ik_exploring(user, neighbors):
                user.exploring()           # EPR fallback
        else:
            raise ValueError(f"Unknown cf_type '{cf_type}'. Expected 'uk' or 'ik'.")  

    _advance_clock(user)


def build_cf_data(users_list, top_k, cf_type,
                  simulate_params, grid_exists, grid_params,
                  check_run_dir, interval_idx=0):
    """
    Build neighbor / similarity structures for every user, depending on cf_type.

    Parameters
    ----------
    users_list      : list of User
    top_k           : int  – max neighbors to retain per user
    cf_type         : "uk" | "ik"
    simulate_params : dict – used only for check/ folder naming
    grid_exists     : bool
    grid_params     : dict | None
    check_run_dir   : str | None – pass None on first call; returned path reused afterwards
    interval_idx    : int  – 0-based interval index for file naming

    Returns
    -------
    cf_neighbors   : list  – one entry per user; semantics depend on cf_type
    check_run_dir  : str   – folder path (created on first call, reused afterwards)
    """
    if cf_type == "uk":
        return _build_uk_data(users_list, top_k, simulate_params,
                              grid_exists, grid_params, check_run_dir, interval_idx)
    elif cf_type == "ik":
        return _build_ik_data(users_list, top_k, simulate_params,
                              grid_exists, grid_params, check_run_dir, interval_idx)
    else:
        raise ValueError(f"Unknown cf_type '{cf_type}'. Expected 'uk' or 'ik'.")


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


# =============================================================================
# USER-KNN  (cf_type = "uk")
# =============================================================================



def _build_uk_data(users_list, top_k, simulate_params,
                   grid_exists, grid_params, check_run_dir, interval_idx):
  
    n_users = len(users_list)

    # ------------------------------------------------------------------
    # User–venue visit-count matrix
    # ------------------------------------------------------------------
    all_venues, venue_set = [], set()
    for user in users_list:
        for row in user.venue_history[1:]:
            if row[0] not in venue_set:
                venue_set.add(row[0])
                all_venues.append(row[0])
    venue_to_idx = {v: i for i, v in enumerate(all_venues)}
    n_venues = len(all_venues)

    uv_matrix = np.zeros((n_users, n_venues), dtype=float)
    for ui, user in enumerate(users_list):
        for row in user.venue_history[1:]:
            uv_matrix[ui, venue_to_idx[row[0]]] = row[1]

    # ------------------------------------------------------------------
    # Cosine similarity (user × user)
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
    # Build neighbor lists
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


def _uk_exploring(user, neighbors):
    """
    UK exploration step.

    Searches for a venue visited by a similar user that this user has not
    yet visited. Neighbors tried in a single weighted-random permutation.
    Returns False to trigger EPR fallback when all neighbors are exhausted.

    On success sets: state="E", num_distinct_venues, next_position,
    destination, _venue_index entry, delta_r=None, angle=None.
    """
    if not neighbors:
        logger.warning(f"User {user.uid}: UK exploration skipped – no neighbors.")
        return False

    neighbor_users = [u for u, _ in neighbors]
    weights = np.array([w for _, w in neighbors], dtype=float)
    if weights.sum() == 0:
        logger.warning(f"User {user.uid}: UK exploration skipped – zero similarity weights.")
        return False
    probs = weights / weights.sum()

    my_venues = set(user._venue_index.keys())
    order = np.random.choice(len(neighbor_users), size=len(neighbor_users),
                             replace=False, p=probs)

    for idx in order:
        neighbor = neighbor_users[idx]
        candidate_rows = [row for row in neighbor.venue_history[1:]
                          if row[0] not in my_venues]
        if not candidate_rows:
            continue

        visit_counts = np.array([row[1] for row in candidate_rows], dtype=float)
        p = visit_counts / visit_counts.sum()
        chosen = candidate_rows[np.random.choice(len(candidate_rows), p=p)]

        user.state = "E"
        user.num_distinct_venues += 1
        user.next_position = (chosen[2], chosen[3])
        user.destination   = chosen[0]
        user._venue_index[user.destination] = len(user.venue_history)
        user.delta_r = None
        user.angle   = None
        return True

    logger.warning(f"User {user.uid}: UK exploration – EPR fallback (no valid venues).")
    return False


# =============================================================================
# ITEM-KNN  (cf_type = "ik")
# =============================================================================




def _build_ik_data(users_list, top_k, simulate_params,
                   grid_exists, grid_params, check_run_dir, interval_idx):
  
    n_users = len(users_list)

    # ------------------------------------------------------------------
    # Collect all venues globally, build index
    # ------------------------------------------------------------------
    all_venues, venue_set = [], set()
    for user in users_list:
        for row in user.venue_history[1:]:
            if row[0] not in venue_set:
                venue_set.add(row[0])
                all_venues.append(row[0])
    venue_to_idx = {v: i for i, v in enumerate(all_venues)}
    n_venues = len(all_venues)

    # venue metadata: [name, x, y]
    venue_meta = {}
    for user in users_list:
        for row in user.venue_history[1:]:
            if row[0] not in venue_meta:
                venue_meta[row[0]] = (row[2], row[3])

    # ------------------------------------------------------------------
    # User–venue matrix  (needed both for co-visit and per-user scores)
    # ------------------------------------------------------------------
    uv_matrix = np.zeros((n_users, n_venues), dtype=float)
    for ui, user in enumerate(users_list):
        for row in user.venue_history[1:]:
            uv_matrix[ui, venue_to_idx[row[0]]] = row[1]

    # ------------------------------------------------------------------
    # Item–item cosine similarity via co-visit matrix
    # V = uv_matrix^T @ uv_matrix  (n_venues × n_venues)
    # Then normalise rows to get cosine similarities.
    # ------------------------------------------------------------------
    V = uv_matrix.T @ uv_matrix          # co-visit counts
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V_norm    = V / norms
    sim_items = V_norm @ V_norm.T        # item-item cosine similarity
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
    # For each user, score unvisited venues and keep top_k
    # ------------------------------------------------------------------
    cf_neighbors = []
    for ui, user in enumerate(users_list):
        user_vec = uv_matrix[ui]                         # (n_venues,) visit counts
        visited_mask = user_vec > 0

        if not visited_mask.any():
            logger.warning(f"User {ui} has no visited venues, so no recommendations.")
            cf_neighbors.append([])
            continue

        # score(v) = sum_w  sim_items[w, v] * visits_u(w)   for w in visited(u)
        # = (sim_items.T @ user_vec)[v]   — but we want sim_items[w, v], so:
        # score = sim_items.T @ user_vec  (shape: n_venues)
        scores = sim_items.T @ user_vec        # shape: (n_venues,)

        # Zero out already-visited venues
        scores[visited_mask] = 0.0

        # Keep top_k by score
        positive_mask = scores > 0
        if not positive_mask.any():
            logger.warning(f"User {ui} has visited venues but no positive-score recommendations.")
            cf_neighbors.append([])
            continue

        pos_indices = np.where(positive_mask)[0]
        k = min(top_k, len(pos_indices))
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


def _ik_exploring(user, neighbors):
    """
    IK exploration step.

    `neighbors` : list of (venue_row, score)  — pre-computed unvisited venues
        sorted by descending recommendation score.

    Selects a venue proportionally to its recommendation score.
    Returns False to trigger EPR fallback when neighbors is empty.

    On success sets: state="E", num_distinct_venues, next_position,
    destination, _venue_index entry, delta_r=None, angle=None.
    """
    if not neighbors:
        logger.warning(f"User {user.uid}: IK exploration skipped – no recommended venues.")
        return False

    my_venues = set(user._venue_index.keys())

    # Filter out any venue the user has since visited (clock may have advanced)
    valid = [(row, score) for row, score in neighbors if row[0] not in my_venues]
    if not valid:
        logger.warning(f"User {user.uid}: IK exploration – all recommended venues already visited.")
        return False

    rows   = [r for r, _ in valid]
    scores = np.array([s for _, s in valid], dtype=float)
    probs  = scores / scores.sum()

    chosen_idx = np.random.choice(len(rows), p=probs)
    chosen     = rows[chosen_idx]

    user.state = "E"
    user.num_distinct_venues += 1
    user.next_position = (chosen[2], chosen[3])
    user.destination   = chosen[0]
    user._venue_index[user.destination] = len(user.venue_history)
    user.delta_r = None
    user.angle   = None
    return True


# =============================================================================
# SHARED CLOCK ADVANCE (identical for all CF types)
# =============================================================================

def _advance_clock(user):
    """Sample delta_t, advance position and clock, update histories and grid."""
    user.delta_t          = model.utils.temporal_power_law_sampling(par=user.b_param).item()
    user.previous_position = user.position
    user.position          = user.next_position
    user.departure         = user.user_history[-1][user.user_history[0].index("Destination")]
    user.clock             = user.clock + user.delta_t
    user.update_histories()
    if user.has_grid:
        user.grid.update_cell_value_given_position(user.position)