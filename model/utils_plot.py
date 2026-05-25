import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import inspect
import scipy.stats

import logging
logger = logging.getLogger(__name__)

def build_plot_title(base_title, filters):
    """
    Build a plot title by combining a base title with simulation parameters.

    This function constructs a descriptive title for plots by appending loading
    parameters in a formatted string, using abbreviated parameter names.

    Parameters
    ----------
    base_title : str
        The base title for the plot.
    filters : dict
        Dictionary of filters to include in the title.

    Returns
    -------
    str
        The constructed title string.
    """
    _PARAM_ABBREV = {
        "ro":           "ρ",
        "gamma":        "γ",
        "a_param":      "a",
        "b_param":      "b",
        "time":         "t",
        "user_number":  "u_n",
        "cell_lenght":  "c_len",
        "cell_number":  "c_num",
        "grid":         "grid",
        "random_start": "r_s",
    }
    if not filters:
        return base_title

    # If a variant is present ("pure" or "mixed"), prepend it to the base title
    # so it is always visible regardless of where the title is truncated.
    variant = filters.get("variant")
    if variant in ("pure", "mixed"):
        prefixed_title = f"[{variant.upper()}] {base_title}"
    else:
        prefixed_title = base_title

    # Exclude internal/display-only keys from the parameter string
    _EXCLUDE = {"variant", "combined"}
    param_str = ", ".join(
        f"{_PARAM_ABBREV.get(k, k)}={v}"
        for k, v in filters.items()
        if k not in _EXCLUDE
    )
    if param_str:
        return f"{prefixed_title} ({param_str})"
    return prefixed_title
        

def filepath_save_plot_or_show(folder = "test_plots", name: str = None, filters: dict = None, show = False):
    """
    Save function for plots (pdf format), which can be saved in different folders of the project.
    The name of the plot, if not passed, is automatically assigned as the name of the calling 
    function through the package 'inspect'

    Parameters:
    ----------
    folder : string, optional (default = "test_plots")
        name of the target folder. The folder must be in AI-E_PR folder.
    name : string, optional (default = None)
        name of the plot figure. If not passed the name of the calling function
        (the function which is calling 'filepath_save_plot', will be the name of the plot)
    filters : dict, optional (default = None)
        Dictionary of filters to be included in the plot name.

    Returns:
    --------
    None
        This function only produces plots for visual inspection and does not
        return any value.
    """
    if show:
        plt.show()
    else:

        # if name is not given, the calling function name is used instead
        if name is None:
            name = inspect.stack()[1].function

        # base folder 
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # folder where plots will be saved
        plots_dir = os.path.join(base_dir, f"{folder}")
        os.makedirs(plots_dir, exist_ok=True)

        filters = ", ".join(f"{k}={v}" for k, v in filters.items())
        filename = f"{name}_{filters}.pdf"
        filepath = os.path.join(plots_dir, filename)

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

def dotted_histogram(numpy_array, bins_per_decade=4):
    """
    Compute a logarithmic histogram with density normalization.

    This function creates logarithmically spaced bins for the input array,
    computes frequencies, and normalizes to density. It filters out zero values
    and returns only bins with positive density.

    Parameters
    ----------
    numpy_array : array-like
        Input array of values to histogram (must be positive).
    bins_per_decade : int, optional (default=4)
        Number of bins per decade for logarithmic spacing.

    Returns
    -------
    centers_filtered : np.ndarray
        Geometric centers of the bins with positive density.
    density_filtered : np.ndarray
        Normalized density values for the bins.
    """
    array = numpy_array[numpy_array > 0]   # avoiding zeros

    min_array = array.min()
    max_array = array.max()

    log_min = np.log10(min_array)
    log_max = np.log10(max_array)

    #Avoiding boundary problems using a margin
    margin = 0.05
    log_min -= margin
    log_max += margin

    num_decades = log_max - log_min
    num_bins = int(np.ceil(num_decades * bins_per_decade))

    # Creation of logarithmic bins
    bins = np.logspace(log_min, log_max, num_bins + 1)

    # frequencies
    counts, edges = np.histogram(array, bins=bins)

    # bin logarithmic center
    centers = np.sqrt(edges[:-1] * edges[1:])

    # normailization
    widths = np.diff(edges)
    density = counts / widths / len(array)

    mask = density > 0 #Avoiding zero density bins 
    centers_filtered = centers[mask]
    density_filtered = density[mask]

    return centers_filtered, density_filtered

def plot_mobility_visualization(user_history, venue_history, stats = None, filters = None, title = None, show = False):
    logger.info("Plotting mobility visualization")
    """
    Visualizes the user mobility trajectory and visited venues.

    Parameters
    ----------
    user_history : pandas.DataFrame
        DataFrame containing the user's trajectory with columns including 'Destination_x_coord',
        'Destination_y_coord', and 'Clock'.
    venue_history : pandas.DataFrame
        DataFrame containing visited venues with columns 'Venue_x_coord', 'Venue_y_coord',
        and 'Number of visits'.
    filters : dict, optional
        Dictionary of filters to be included in the plot title.
        If provided, the center of mass and radius of gyration are plotted.
    title : str, optional
        Title for the plot.
    
    Returns
    -------
    None
    """


    x = venue_history["Venue_x_coord"].to_numpy()
    y = venue_history["Venue_y_coord"].to_numpy()
    num_visits = venue_history["Number of visits"].to_numpy()
    x_traj = user_history["Destination_x_coord"].to_numpy()
    y_traj = user_history["Destination_y_coord"].to_numpy()
    # Adding starting position
    x_traj, y_traj = np.insert(x_traj, 0, 0), np.insert(y_traj, 0, 0)

    _, ax = plt.subplots()

    # Normalize number of visits for colormap
    norm = plt.Normalize(num_visits.min(), num_visits.max())
    cmap = plt.cm.viridis  # Most used color scale
    sc = ax.scatter(x, y, s=10, c=num_visits, cmap=cmap, norm=norm, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Number of visits')
    ax.plot(x_traj, y_traj, color='red', linewidth=1, alpha=0.7)

    if stats is not None:
        r_gyr = stats.query('metric == "radius of gyration"')['value'].iloc[0]
        cm_x = stats.query('metric == "x_center of mass"')['value'].iloc[0]
        cm_y = stats.query('metric == "y_center of mass"')['value'].iloc[0]

        ax.scatter(cm_x,cm_y, marker='x', c="purple", alpha=0.6, s=50, zorder=3)
        circle = plt.Circle(
            (cm_x, cm_y),
            r_gyr,
            facecolor=(0.6, 0, 0.8, 0.2),
            edgecolor=(0.4, 0, 0.6, 0.8), 
            linewidth=2
        )
        ax.add_patch(circle) 

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title,fontsize=7)

    filepath_save_plot_or_show("plots", filters=filters, show=show)


_MP_COLORS = [
    "#E73948", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#FF5722", "#8BC34A", "#673AB7", "#009688",
]

def _single_as_series(data, label=""):
    """Wrap a single-series call into the list-of-(label, data) format."""
    return [(label, data)]


def plot_rg_distribution(rg_list, filters=None, title=None, show=False, series_list=None):
    """
    Plot the distribution of radius of gyration values on a log-log scale.

    In single-series mode (series_list=None) behaves exactly as before.
    In multi-series mode pass series_list as a list of (label, rg_list) tuples;
    rg_list is then ignored.

    Parameters
    ----------
    rg_list : list or array-like
        Per-user rg values (used only when series_list is None).
    filters : dict, optional
    title : str, optional
    show : bool, optional
    series_list : list of (str, list), optional
        Multi-series override: [(label, rg_list), ...].
    """
    logger.info("Plotting rg distribution")

    if series_list is None:
        series_list = _single_as_series(rg_list)
    multi = len(series_list) > 1

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (label, data) in enumerate(series_list):
        arr = np.array(data)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if len(arr) == 0:
            logger.warning(f"plot_rg_distribution: no valid data for series '{label}'")
            continue
        centers, density = dotted_histogram(arr, bins_per_decade=16)
        kw = {"color": _MP_COLORS[idx % len(_MP_COLORS)]} if multi else {}
        ax.loglog(centers, density, "o", markersize=5, label=label if multi else None, **kw)

    ax.set_xlabel("Radius of gyration")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show("plots", filters=filters, show=show)


def plot_dist_distribution(mean_dist_list, filters=None, title=None, show=False, series_list=None):
    """
    Plot the distribution of individual travel distances (delta_r) on a log-log scale.

    Each data point represents a single exploration jump, not a per-user mean.
    All delta_r values from all users in a batch are pooled together and binned
    into a log-log histogram, giving the empirical distribution of jump sizes.

    In single-series mode (series_list=None) ``mean_dist_list`` must be a flat
    array-like of raw delta_r values (one entry per exploration step across all users).
    In multi-series mode pass series_list as [(label, delta_r_flat_list), ...];
    ``mean_dist_list`` is then ignored.

    Parameters
    ----------
    mean_dist_list : list or array-like
        Flat list of individual delta_r values for all exploration steps (used only
        when series_list is None).
    filters : dict, optional
    title : str, optional
    show : bool, optional
    series_list : list of (str, list), optional
    """
    logger.info("Plotting travel distance distribution (all delta_r values)")

    if series_list is None:
        series_list = _single_as_series(mean_dist_list)
    multi = len(series_list) > 1

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (label, data) in enumerate(series_list):
        arr = np.array(data, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if len(arr) == 0:
            logger.warning(f"plot_dist_distribution: no valid data for series '{label}'")
            continue
        centers, density = dotted_histogram(arr, bins_per_decade=8)
        kw = {"color": _MP_COLORS[idx % len(_MP_COLORS)]} if multi else {}
        ax.loglog(centers, density, "o", markersize=5, label=label if multi else None, **kw)

    ax.set_xlabel(r"Travel distance $\Delta r$")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show("plots", filters=filters, show=show)


def plot_frequency_distribution(venue_history_list, gamma, filters=None, title=None, show=False):

    logger.info("Plotting frequency distribution (averaged across users)")
    """
    Plot the rank-frequency distribution of venue visits averaged across all users.

    For each user the venues are ranked in descending order of visit count.
    The visit counts at each rank are then averaged across all users that have
    at least that many venues, producing a single mean curve.  A power-law fit
    line is overlaid for comparison.

    Parameters
    ----------
    venue_history_list : list of pandas.DataFrame or pandas.DataFrame
        One DataFrame per user, each containing a 'Number of visits' column.
        A bare DataFrame is accepted for backward compatibility (treated as a
        single-user list).
    gamma : float
        Exponent for the power-law fit line (f(x) = x^(-1-gamma)).
    filters : dict, optional
    title : str, optional
    show : bool, optional
    """
    if isinstance(venue_history_list, pd.DataFrame):
        venue_history_list = [venue_history_list]

    rank_visits = {}
    for vh in venue_history_list:
        sorted_visits = vh['Number of visits'].sort_values(ascending=False).values
        for rank, visits in enumerate(sorted_visits, start=1):
            rank_visits.setdefault(rank, []).append(visits)

    ranks      = np.array(sorted(rank_visits.keys()))
    avg_visits = np.array([np.mean(rank_visits[r]) for r in ranks])

    plt.figure(figsize=(8, 6))
    plt.scatter(ranks, avg_visits, color='blue', label='avg data', s=50)

    x_fit = np.linspace(ranks.min(), ranks.max(), 100)
    y_fit = x_fit ** (-1 - gamma)
    y_fit = y_fit * avg_visits.max() / y_fit.max()
    plt.plot(x_fit, y_fit, color='red', linestyle='--', label=r'$f(x) = x^{-1-\gamma}$')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Rank")
    plt.ylabel("Average frequency")
    if title:
        plt.title(title, fontsize=7)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    filepath_save_plot_or_show("plots", filters=filters, show=show)


def plot_two_radius_vs_radius(rg_list, two_rg_list, filters=None, title=None, show=False,
                              series_list=None):
    """
    Plot 2-rg vs rg.

    In single-series mode (series_list=None) behaves exactly as before.
    In multi-series mode pass series_list as [(label, rg_list, two_rg_list), ...].

    Parameters
    ----------
    rg_list : list or array-like
        Used only when series_list is None.
    two_rg_list : list or array-like
        Used only when series_list is None.
    filters : dict, optional
    title : str, optional
    show : bool, optional
    series_list : list of (str, list, list), optional
    """
    logger.info("Plotting 2-radius vs radius of gyration")

    if series_list is None:
        series_list = [("", rg_list, two_rg_list)]
    multi = len(series_list) > 1

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (label, rg, two_rg) in enumerate(series_list):
        kw = {"color": _MP_COLORS[idx % len(_MP_COLORS)], "alpha": 0.7} if multi else {}
        ax.scatter(rg, two_rg, s=5, label=label if multi else None, **kw)

    ax.set_xlabel("Radius of Gyration")
    ax.set_ylabel("2-Radius of Gyration")
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show("plots", filters=filters, show=show)


def plot_S_over_time_from_df(s_over_time_list, filters=None, title=None, bins_per_decade=4,
                             show=False, series_list=None):
    """
    Plot S(t) over time.

    In single-series mode (series_list=None) behaves exactly as before.
    In multi-series mode pass series_list as [(label, s_over_time_list), ...];
    s_over_time_list is then ignored.

    Parameters
    ----------
    s_over_time_list : list of pd.DataFrame
        Each DataFrame has columns ['clock', 's'].  Used only when series_list is None.
    filters : dict, optional
    title : str, optional
    bins_per_decade : int, optional
    show : bool, optional
    series_list : list of (str, list of pd.DataFrame), optional
    """
    logger.info("Plotting S over time (from analysis data)")

    if series_list is None:
        series_list = _single_as_series(s_over_time_list)
    multi = len(series_list) > 1

    if not any(data for _, data in series_list):
        logger.warning("No S-over-time data available to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (label, data_list) in enumerate(series_list):
        if not data_list:
            logger.warning(f"plot_S_over_time_from_df: no data for series '{label}'")
            continue

        all_clocks = np.concatenate([df["clock"].values for df in data_list])
        min_c, max_c = all_clocks.min(), all_clocks.max()
        num_samples  = max(2, int(round((np.log10(max_c) - np.log10(min_c)) * bins_per_decade)))
        sample_times = np.logspace(np.log10(min_c), np.log10(max_c), num_samples)
        sample_times[-1] = max_c

        pct_by_time = {t: [] for t in sample_times}
        for df in data_list:
            clocks = df["clock"].values
            s      = df["s"].values
            idx_s1 = np.where(s >= 1.0)[0]
            if len(idx_s1) == 0:
                continue
            t_reach_1 = clocks[idx_s1[0]]
            for t in sample_times:
                if t < t_reach_1:
                    continue
                mask   = clocks <= t
                i_last = int(np.where(mask)[0][-1])
                pct_by_time[t].append(s[i_last])

        sample_t_plot = [t for t in sample_times if pct_by_time[t]]
        avg_pct_plot  = [np.mean(pct_by_time[t]) for t in sample_t_plot]

        kw = {"color": _MP_COLORS[idx % len(_MP_COLORS)], "s": 25} if multi else {"s": None}
        ax.scatter(sample_t_plot, avg_pct_plot, marker='o',
                   label=label if multi else None,
                   **({k: v for k, v in kw.items() if v is not None}))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Clock")
    ax.set_ylabel("S(t)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(folder="plots", filters=filters, show=show)


def plot_r_gyr_over_time_from_df(rg_over_time_list, filters=None, title=None, bins_per_decade=4,
                                 show=False, series_list=None):
    """
    Plot normalised rg(t)/rg_final [%] over time.

    In single-series mode (series_list=None) behaves exactly as before.
    In multi-series mode pass series_list as [(label, rg_over_time_list), ...];
    rg_over_time_list is then ignored.

    Parameters
    ----------
    rg_over_time_list : list of pd.DataFrame
        Each DataFrame has columns ['clock', 'rg'].  Used only when series_list is None.
    filters : dict, optional
    title : str, optional
    bins_per_decade : int, optional
    show : bool, optional
    series_list : list of (str, list of pd.DataFrame), optional
    """
    logger.info("Plotting r_gyr over time (normalized percentage, from analysis data)")

    if series_list is None:
        series_list = _single_as_series(rg_over_time_list)
    multi = len(series_list) > 1

    if not any(data for _, data in series_list):
        logger.warning("No rg-over-time data available to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (label, data_list) in enumerate(series_list):
        if not data_list:
            logger.warning(f"plot_r_gyr_over_time_from_df: no data for series '{label}'")
            continue

        all_clocks = np.concatenate([df["clock"].values for df in data_list])
        min_c, max_c = all_clocks.min(), all_clocks.max()
        num_samples  = max(2, int(round((np.log10(max_c) - np.log10(min_c)) * bins_per_decade)))
        sample_times = np.logspace(np.log10(min_c), np.log10(max_c), num_samples)
        sample_times[-1] = max_c

        pct_by_time = {t: [] for t in sample_times}
        for df in data_list:
            clocks = df["clock"].values
            rg     = df["rg"].values
            idx_start = np.where(rg > 0)[0]
            if len(idx_start) == 0:
                continue
            t_start  = clocks[idx_start[0]]
            final_rg = rg[-1]
            if final_rg == 0:
                logger.warning("Final rg is 0 despite rg > 0 earlier, skipping user")
                continue
            for t in sample_times:
                if t < t_start:
                    continue
                mask   = clocks <= t
                i_last = int(np.where(mask)[0][-1])
                pct_by_time[t].append(rg[i_last] / final_rg * 100.0)

        sample_t_plot = [t for t in sample_times if pct_by_time[t]]
        avg_pct_plot  = [np.mean(pct_by_time[t]) for t in sample_t_plot]

        kw = {"color": _MP_COLORS[idx % len(_MP_COLORS)], "s": 25} if multi else {}
        ax.scatter(sample_t_plot, avg_pct_plot, marker='o',
                   label=label if multi else None, **kw)

    ax.set_xlabel("Clock")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$r_g(t)\,/\,r_g^{last}$ [%]")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show(folder="plots", filters=filters, show=show)

def print_batch_stats(batch_stats_list, filters=None, title=None):
    """
    Print a formatted summary table of batch-level statistics to stdout.

    Fully generic: iterates over every metric/value row present in each
    batch_stats.parquet without any hardcoded metric names.  This means
    that any new metric added to batch_stats.parquet during analysis is
    automatically displayed here without touching this function.

    One section is printed per batch, mirroring the per-batch loop used
    by the batch_grid plotter.  When more than one batch is available, a
    recap table follows with mean and std for every numeric metric.

    Parameters
    ----------
    batch_stats_list : list of (str, pd.DataFrame)
        As returned by :func:`load_all_batch_stats`:
        (batch_id, stats_df) pairs sorted by batch_id.
        Each DataFrame must have columns 'metric' and 'value'.
    filters : dict, optional
        Simulation parameters — printed as a header line for context.
    title : str, optional
        Optional title printed above the output block.
    """
    if not batch_stats_list:
        logger.warning("print_batch_stats: no batch stats to display.")
        return

    SEP_WIDE   = "=" * 72
    SEP_NARROW = "-" * 72

    # --- header ---
    print(f"\n{SEP_WIDE}")
    if title:
        print(f"  {title}")
    else:
        print("  BATCH STATS SUMMARY")
    if filters:
        param_str = "  params: " + ", ".join(f"{k}={v}" for k, v in filters.items())
        print(param_str)
    print(f"  {len(batch_stats_list)} batch(es) found")
    print(SEP_WIDE)

    # recap_data[metric_name] = [value_batch_0, value_batch_1, ...]
    recap_data = {}

    for batch_id, stats_df in batch_stats_list:
        print(f"\n  Batch : {batch_id}")
        print(SEP_NARROW)

        for _, row in stats_df.iterrows():
            metric = str(row["metric"])
            value  = row["value"]

            # Format integers without decimal point when the value is whole
            if isinstance(value, float) and value.is_integer():
                val_str = f"{int(value):>10d}"
            else:
                val_str = f"{value:>10.4f}"

            print(f"    {metric:<30s} {val_str}")
            recap_data.setdefault(metric, []).append(value)

        print(SEP_NARROW)

    # --- recap table (only when > 1 batch) ---
    if len(batch_stats_list) > 1:
        print(f"\n  {'RECAP':^70s}")
        print(SEP_WIDE)

        # Header: Batch | metric_0 | metric_1 | ...
        metric_names = list(recap_data.keys())
        col_w = 12
        hdr = f"  {'Batch':<26s}" + "".join(f"{m[:col_w]:>{col_w}s}" for m in metric_names)
        print(hdr)
        print(SEP_NARROW)

        # One row per batch
        for idx, (batch_id, stats_df) in enumerate(batch_stats_list):
            metrics_dict = dict(zip(stats_df["metric"], stats_df["value"]))
            row_str = f"  {batch_id:<26s}"
            for m in metric_names:
                v = metrics_dict.get(m, float("nan"))
                if isinstance(v, float) and v.is_integer():
                    row_str += f"{int(v):>{col_w}d}"
                else:
                    row_str += f"{v:>{col_w}.4f}"
            print(row_str)

        # Mean / std rows
        print(SEP_NARROW)
        for label, func in [("mean", np.mean), ("std", np.std)]:
            row_str = f"  {label:<26s}"
            for m in metric_names:
                vals = [v for v in recap_data[m] if not np.isnan(v)]
                row_str += f"{func(vals):>{col_w}.4f}" if vals else f"  {'n/a':>{col_w-2}s}"
            print(row_str)

        print(SEP_WIDE)

    print()

def plot_aggregate_grid(grid_matrix, filters=None, title=None, batch_id=None, mask_center=True, show=False):
    """ Plot a heatmap of an aggregate grid matrix.

        Parameters
        ----------
        grid_matrix : np.ndarray
            2D integer array with cumulative visit counts per cell.
        filters : dict, optional
            Used for the saved plot file name.
        title : str, optional
            Plot title.
        batch_id : str, optional
            When provided, included in the saved file name to distinguish individual batch plots from the param-level plot.
        mask_center : bool, optional
            Whether to mask the center cell in the plot.
            
    """

    logger.info(f"Plotting aggregate grid heatmap{' for batch ' + batch_id if batch_id else ''}")

    # Copy
    m = grid_matrix.copy()
    cx = m.shape[0] // 2
    cy = m.shape[1] // 2
    m_safe = np.where(m == 0, 1, m)
    if mask_center:
        m_masked = np.ma.array(m_safe, mask=False)
        m_masked.mask[cx, cy] = True
    else:
        m_masked = np.ma.array(m_safe, mask=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(
        m_masked.T,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        #norm=LogNorm()
    )

    plt.colorbar(im, ax=ax, label="Total visits")
    ax.set_xlabel("Cell X")
    ax.set_ylabel("Cell Y")

    if title:
        ax.set_title(title, fontsize=6)

    name = f"plot_aggregate_grid{'_' + batch_id if batch_id else ''}"

    filepath_save_plot_or_show("plots", name=name, filters=filters, show=show)


def plot_entropy_distribution(entropy_list, entropy_type="srand", filters=None, title=None,
                              show=False, series_list=None):
    """
    Plot the distribution of one entropy measure across users.

    Works in both single-series mode (plot / "p") and multi-series mode
    (multipleplot / "mp"), following exactly the same pattern as
    plot_rg_distribution().

    Parameters
    ----------
    entropy_list : list or array-like
        Per-user entropy values.  Ignored when series_list is provided.
    entropy_type : str
        One of "srand", "sunc", "sreal".  Controls the axis label.
    filters : dict, optional
    title : str, optional
    show : bool, optional
    series_list : list of (str, list), optional
        Multi-series override: [(label, entropy_list), ...].
    """
    _LABELS = {
        "srand": (r"$S^{rand}$", "Probability Density"),
        "sunc":  (r"$S^{unc}$",  "Probability Density"),
        "sreal": (r"$S^{real}$", "Probability Density"),
    }
    xlabel, ylabel = _LABELS.get(entropy_type, ("Entropy", "Probability Density"))

    logger.info(f"Plotting entropy distribution ({entropy_type})")

    if series_list is None:
        series_list = _single_as_series(entropy_list)
    multi = len(series_list) > 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (label, data) in enumerate(series_list):
        arr = np.array(data, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            logger.warning(f"plot_entropy_distribution({entropy_type}): no valid data for '{label}'")
            continue

        color = _MP_COLORS[idx % len(_MP_COLORS)] if multi else "#E73948"

        counts, edges = np.histogram(arr, bins=30, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, counts, "o", markersize=5, color=color,
                label=label if multi else None)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    if multi:
        ax.legend()
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    filepath_save_plot_or_show("plots", filters=filters, show=show)


def plot_entropy_triple(s_rand_list, s_unc_list, s_real_list, filters=None, title=None, show=False):
    """
    Plot the distributions of S^rand, S^unc, and S^real on a single graph,
    all on linear scale.
    """
    logger.info("Plotting entropy triple (s_rand, s_unc, s_real)")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    specs = [
        (s_rand_list, r"$S^{rand}$", "r"),
        (s_unc_list,  r"$S^{unc}$",  "k"),
        (s_real_list, r"$S^{real}$", "b"),
    ]

    for data, label, color in specs:
        arr = np.array(data, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue

        counts, edges = np.histogram(arr, bins=30, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, counts, "o", markersize=5, color=color, label=label)

    ax.set_xlabel("Entropy")
    ax.set_ylabel("Probability Density")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    if title:
        fig.suptitle(title, fontsize=7, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    filepath_save_plot_or_show("plots", filters=filters, show=show)


def plot_cf_neighbor_similarity_distribution(cf_neighbors, check_run_dir, interval_idx, top_n_ranks=3):
    """
    Plot the similarity-score distribution for the top-N ranks across all users,
    for a single CF interval (i.e. one call to _build_cf_data).

    For each rank position k in {1, …, top_n_ranks}, the function collects the
    similarity value of the k-th most-similar neighbor for every user that has at
    least k neighbors, then draws all rank distributions as overlapping histograms
    on the same axes (one colour per rank).

    The plot is saved as a PDF inside check_run_dir.

    Parameters
    ----------
    cf_neighbors : list of list of (User, float)
        Output of _build_cf_data: cf_neighbors[i] is a ranked list of
        (neighbor_User, similarity_weight) pairs for user i, sorted by
        descending similarity.  Users with no positive-similarity neighbor
        have an empty list.
    check_run_dir : str
        Path to the per-run check folder (already created by _build_cf_data).
        The PDF is saved here as ``neighbor_sim_dist_interval_{interval_idx+1}.pdf``.
    interval_idx : int
        0-based index of the current CF interval; used only in the file name
        and plot title (displayed as interval_idx + 1).
    top_n_ranks : int, optional (default=3)
        Number of rank positions to visualise (1 = most similar, 2 = second
        most similar, …).
    """
    # Palette: one colour per rank, colour-blind-friendly sequence
    COLOURS = ["#E73948", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    # Collect similarity values per rank position
    rank_values = {}   # rank (1-based) -> list of float
    for neighbors in cf_neighbors:
        for rank_idx in range(min(top_n_ranks, len(neighbors))):
            rank = rank_idx + 1          # 1-based label
            sim  = neighbors[rank_idx][1]
            rank_values.setdefault(rank, []).append(sim)

    if not rank_values:
        logger.warning(
            f"plot_cf_neighbor_similarity_distribution: no neighbor data for interval "
            f"{interval_idx + 1}; skipping plot."
        )
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    all_sims = [v for vals in rank_values.values() for v in vals]
    global_min = min(all_sims)
    global_max = max(all_sims)
    # Shared bin edges so distributions are directly comparable
    bins = np.linspace(global_min, global_max, 31)

    for rank in sorted(rank_values.keys()):
        vals   = np.array(rank_values[rank])
        colour = COLOURS[(rank - 1) % len(COLOURS)]
        ax.hist(
            vals,
            bins=bins,
            alpha=0.55,
            color=colour,
            edgecolor="white",
            linewidth=0.4,
            label=f"Rank {rank}  (n={len(vals)})",
            density=True,
        )
        # Overlay a KDE for a smoother visual 
        if len(vals) >= 5:
            kde_x = np.linspace(global_min, global_max, 300)
            kde   = scipy.stats.gaussian_kde(vals, bw_method="scott")
            ax.plot(kde_x, kde(kde_x), color=colour, linewidth=1.8)

    interval_label = interval_idx + 1
    ax.set_title(
        f"CF similarity distribution — interval {interval_label}\n"
        f"(top-{top_n_ranks} ranks, {len(cf_neighbors)} users)",
        fontsize=10,
    )
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.legend(framealpha=0.7, fontsize=9)
    ax.set_xlim(global_min - 0.01, min(global_max + 0.01, 1.0))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fname = os.path.join(
        check_run_dir,
        f"neighbor_sim_dist_interval_{interval_label:02d}.pdf"
    )
    plt.savefig(fname)
    plt.close()
    logger.info(f"Saved CF similarity distribution plot: {fname}")