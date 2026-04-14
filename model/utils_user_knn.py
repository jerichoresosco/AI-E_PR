import numpy as np
import model.user
import model.utils
import os
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

def make_decision_cf(user, neighbors):
        """
        CF variant of make_decision. Uses the same EPR probability to decide
        between exploration and return, then delegates to the CF movement methods.
        Falls back to standard EPR movement when no CF-compatible venues are found.

        Parameters
        ----------
        neighbors : list of (User, float)
            Pre-computed list of (neighbor_user, similarity_weight) pairs,
            sorted by descending similarity, produced by Simulation._build_cf_data().

        Notes
        -----
        - E/R decision uses the same P_new as EPR.
        - CF exploration: venue from similar user not yet visited by this user.
          Fallback to EPR exploration if all neighbors are exhausted.
        - CF return: Same as EPR
        - delta_r and angle are set to None for CF moves (no spatial sampling).
        - delta_t is always sampled from the temporal power-law, same as EPR.
        """
        P_new = user.simulation.ro * user.num_distinct_venues ** (-user.simulation.gamma)

        if np.random.random_sample(size=None) >= P_new:
            #if not user.cf_returning(neighbors):
            user.returning() #Classic EPR returning
        else:
            if not cf_exploring(user, neighbors):
                user.exploring()

        user.delta_t = model.utils.temporal_power_law_sampling(par=user.b_param).item()
        user.previous_position = user.position
        user.position = user.next_position
        user.departure = user.user_history[-1][user.user_history[0].index("Destination")]
        user.clock = user.clock + user.delta_t
        user.update_histories()
        if user.has_grid:
            user.grid.update_cell_value_given_position(user.position)


def build_cf_data(users_list, top_k, simulate_params, grid_exists,
                  grid_params, check_run_dir, interval_idx=0):
    """
    Build cosine-similarity-based top-k neighbor lists for every user.
 
    Constructs a user-venue visit-count matrix from the current state of
    users_list, computes pairwise cosine similarities, and returns for
    each user a ranked list of the most similar neighbors.
 
    Files are written to a per-run subfolder inside check/ whose name mirrors
    the analysis_data folder convention (params + batch timestamp), so every
    run is kept separate and can be traced back to its simulation parameters.
 
    Parameters
    ----------
    users_list : list of User
        The live user objects whose venue_history is used to build the matrix.
    top_k : int
        Maximum number of neighbors to retain per user.
        If fewer users have positive similarity, all of them are kept.
    simulate_params : dict
        The simulation parameter dict (must contain "ro", "gamma", "a_param",
        "b_param", "time"; optionally "epr_fraction", "random_start").
        Used only for building the check/ folder name.
    grid_exists : bool
        Whether the simulation was run with an active grid.
        Used together with grid_params for folder naming.
    grid_params : dict or None
        Grid parameters dict with keys "cell_lenght" and "cell_number".
        Required when grid_exists is True, otherwise ignored.
    check_run_dir : str or None
        Path to the per-run check subfolder created on the first CF interval.
        Pass None on the first call: the folder is created and the path is
        returned so the caller can pass it back on subsequent intervals,
        ensuring all intervals of the same run share the same folder.
    interval_idx : int, optional (default=0)
        0-based index of the current CF interval, forwarded to the
        similarity-distribution plot so each interval gets its own file.
 
    Returns
    -------
    cf_neighbors : list of list of (User, float)
        cf_neighbors[i] = [(neighbor_User, similarity_weight), ...]
        sorted by descending similarity for user i.
        An empty list is returned for users with no positively similar neighbor.
    check_run_dir : str
        The check subfolder path used (or just created) for this run.
        The caller must pass this back on the next call so all intervals
        of the same run land in the same folder.
 
    Notes
    -----
    - Similarity is cosine similarity on raw visit counts.
    - Self-similarity is zeroed out (a user is never its own neighbor).
    - Only neighbors with strictly positive similarity are included.
    """
    n_users = len(users_list)
 
    # ------------------------------------------------------------------
    # Build the user-venue visit-count matrix
    # ------------------------------------------------------------------
    all_venues = []
    venue_set  = set()
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
    # Cosine similarity matrix
    # ------------------------------------------------------------------
    norms = np.linalg.norm(uv_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    uv_norm    = uv_matrix / norms
    sim_matrix = uv_norm @ uv_norm.T
    np.fill_diagonal(sim_matrix, 0.0)
 
    # ------------------------------------------------------------------
    # Resolve (or create) the per-run check subfolder
    #
    # check/
    #   ro_{ro}_g_{gamma}_a_{a}_b_{b}_t_{time}[_cl_X_cn_Y]_rs_{y|n}_ef_{efrac}/
    #       {YYYYMMDD_HHMMSS_ffffff}/
    #           sim_matrix_interval_{N}.txt
    #           neighbors_interval_{N}.txt
    #           neighbor_sim_dist_interval_{N}.pdf
    # ------------------------------------------------------------------
    if check_run_dir is None:
        root_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        check_base = os.path.join(root_dir, "check")
 
        p      = simulate_params
        efrac  = float(p.get("epr_fraction", 1.0))
        rstart = p.get("random_start", False)
 
        if grid_exists and grid_params is not None:
            gp = grid_params
            param_name = (
                f"ro_{p['ro']}_g_{p['gamma']}"
                f"_a_{p['a_param']}_b_{p['b_param']}_t_{p['time']}"
                f"_cl_{gp['cell_lenght']}_cn_{gp['cell_number']}"
                f"_rs_{'y' if rstart else 'n'}_ef_{efrac}"
            )
        else:
            param_name = (
                f"ro_{p['ro']}_g_{p['gamma']}"
                f"_a_{p['a_param']}_b_{p['b_param']}_t_{p['time']}"
                f"_rs_{'y' if rstart else 'n'}_ef_{efrac}"
            )
 
        batch_ts      = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        check_run_dir = os.path.join(check_base, param_name, batch_ts)
        os.makedirs(check_run_dir, exist_ok=True)
        logger.info(f"Check folder for this run: {check_run_dir}")
 
    interval_label = interval_idx + 1   # 1-based for file names
 
    # ------------------------------------------------------------------
    # Persist the similarity matrix
    # ------------------------------------------------------------------
    sim_matrix_path = os.path.join(
        check_run_dir, f"sim_matrix_interval_{interval_label:02d}.txt"
    )
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
        cf_neighbors.append(
            [(users_list[j], float(sims[j])) for j in pos_indices]
        )
 
    # ------------------------------------------------------------------
    # Persist the neighbor list
    # ------------------------------------------------------------------
    neighbors_path = os.path.join(
        check_run_dir, f"neighbors_interval_{interval_label:02d}.txt"
    )
    with open(neighbors_path, "w") as f:
        for row in cf_neighbors:
            f.write(",".join(map(str, row)) + "\n")
 
    # ------------------------------------------------------------------
    # Plot similarity distribution for this interval
    # ------------------------------------------------------------------
    model.utils_plot.plot_cf_neighbor_similarity_distribution(
        cf_neighbors,
        check_run_dir=check_run_dir,
        interval_idx=interval_idx,
        top_n_ranks=3,
    )
 
    logger.info(
        f"CF neighbor data built: {n_users} users, {n_venues} unique venues, "
        f"top_k={top_k_actual}, interval={interval_label}"
    )
    return cf_neighbors, check_run_dir


def cf_exploring(user, neighbors):
    """
    CF exploration step. Searches for a venue visited by a similar user that
    this user has not yet visited, selected proportionally to the similar user's
    visit frequency for that venue.

    Neighbors are tried in a single weighted-random permutation (without
    replacement), so each neighbor is attempted at most once per call.
    Returns False when all neighbors have been tried without finding a valid
    venue, triggering an EPR exploration fallback in make_decision_cf().

    Parameters
    ----------
    neighbors : list of (User, float)
        (neighbor_user, similarity_weight) pairs.

    Returns
    -------
    bool
        True if a CF venue was found and state was updated; False otherwise.

    Notes
    -----
    On success sets: state="E", num_distinct_venues, next_position,
    destination, _venue_index entry, delta_r=None, angle=None.
    """
    if not neighbors:
        logger.warning(f"User {user.uid}: CF exploration skipped due to no neighbors.")
        return False

    my_venues = set(user._venue_index.keys())

    neighbor_users = [u for u, _ in neighbors]
    weights = np.array([w for _, w in neighbors], dtype=float)
    if weights.sum() == 0:
        logger.warning(f"User {user.uid}: CF exploration skipped due to zero similarity weights.")
        return False
    probs = weights / weights.sum()

    # Weighted random permutation: try each neighbor at most once
    order = np.random.choice(len(neighbor_users), size=len(neighbor_users),
                                replace=False, p=probs)

    for idx in order:
        neighbor = neighbor_users[idx]
        # Venues visited by this neighbor but not yet by current user
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
        user.destination = chosen[0]
        user._venue_index[user.destination] = len(user.venue_history)#?
        user.delta_r = None
        user.angle = None
        return True
    
    logger.warning(f"User {user.uid}: CF exploration fallback to EPR due to no valid venues from any neighbor.")
    return False

# def cf_returning(user, neighbors):
#     """
#     CF return step. Selects a venue visited by both a similar user and this user
#     (excluding the current position), chosen proportionally to this user's own
#     visit frequency for those venues.

#     Neighbors are tried in a single weighted-random permutation (without
#     replacement), so each neighbor is attempted at most once per call.
#     Returns False only when all neighbors have been tried without finding a
#     common venue, triggering an EPR return fallback in make_decision_cf().

#     Parameters
#     ----------
#     neighbors : list of (User, float)oggle
#         (neighbor_user, similarity_weight) pairs.

#     Returns
#     -------
#     bool
#         True if a CF venue was found and state was updated; False otherwise.

#     Notes
#     -----
#     On success sets: state="R", next_position, next_venue_return,
#     _row_index, delta_r=None, angle=None.
#     """
#     if not neighbors:
#         logger.info(f"User {user.uid}: CF return skipped due to no neighbors.")
#         return False

#     neighbor_users = [u for u, _ in neighbors]
#     weights = np.array([w for _, w in neighbors], dtype=float)
#     if weights.sum() == 0:
#         logger.info(f"User {user.uid}: CF return skipped due to zero similarity weights.")
#         return False
#     probs = weights / weights.sum()

#     # Weighted random permutation: try each neighbor at most once
#     order = np.random.choice(len(neighbor_users), size=len(neighbor_users),
#                             replace=False, p=probs)

#     for idx in order:
#         neighbor = neighbor_users[idx]
#         neighbor_venue_names = {row[0] for row in neighbor.venue_history[1:]}

#         # Venues visited by both this user and the chosen neighbor, excluding current position
#         common_rows = [row for row in user.venue_history[1:]
#                     if row[0] in neighbor_venue_names
#                     and (row[2], row[3]) != user.position]

#         if not common_rows:
#             continue

#         visit_counts = np.array([row[1] for row in common_rows], dtype=float)
#         p = visit_counts / visit_counts.sum()
#         chosen = common_rows[np.random.choice(len(common_rows), p=p)]

#         user.state = "R"
#         user.next_position = (chosen[2], chosen[3])
#         user.next_venue_return = chosen[0]
#         user._row_index = user._venue_index[chosen[0]]
#         user.delta_r = None
#         user.angle = None
#         return True

#     logger.info(f"User {user.uid}: CF return fallback to EPR due to no common venues with any neighbor.")
#     return False
