import logging
import model.simulation
import model.utils
import time
from pprint import pprint

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():

    # ==========================================================
    # Choose mode
    # ==========================================================

    mode = "cp"  # Options (short aliases):
                        #   "s"     run simulation, save raw_data only
                        #   "a"    - load raw_data, compute & save analysis_data
                        #   "sa"   - simulate then analyse without reloading from disk
                        #   "p"    - load analysis_data and plot
                        #   "cp"   - load combined_analysis_data and plot (lsa output)
                        #   "ls"   - load a raw_data batch, continue simulation, save raw_data
                        #   "lsa"  load a raw_data batch, continue simulation, analyse and save
                        #   "mp"   - multipleplot: overlay multiple combined datasets on shared axes

    # ----------------------------------------------------------
    # simulate_params -- used only in "simulate" and
    # "simulate_and_analyse" modes.
    # ----------------------------------------------------------
    simulate_params = {
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.75,
        "b_param": 0.6,
        "user_number": 5,
        "time": 10000,
        "starting_position": "random",   # "centered" | "random" | "normal"
        "algorithm": "ge",               # "ce" = classic EPR (no grid)
                                         # "ge" = grid EPR
                                         # "uk" = User-KNN CF
                                         # "ik" = Item-KNN CF
                                         # "mf" = Matrix Factorization CF
                                         # "ba" = Barabási-Albert popularity CF
        "top_k": 10,                     # neighbors used in CF algorithms (uk/ik/mf)
    }

    ##LATENT FACTOR = 20 HARDCODED

    grid_params = {
        "cell_lenght": 101, 
        "cell_number": 101 #must be odd to have a central cell
    }

    # ----------------------------------------------------------
    # analyse_params -- used only in "analyse" mode.
    # Identifies which raw_data folders to process.
    # user_number is absent: analysis processes complete run
    # folders, never a subset of users.
    # ----------------------------------------------------------
    analyse_params = {
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.75,
        "b_param": 0.6,
        "time": 30000,
        "starting_position": "random",   # must match the simulate_params value used
        "algorithm": "ge",               # must match the simulate_params value used
    }

    # ----------------------------------------------------------
    # plot_params -- used only in "plot" mode.
    # Identifies which analysis_data to load.
    # user_number: positive = first N, negative = last N, absent = all
    # ----------------------------------------------------------
    plot_params = {
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.75,
        "b_param": 0.6,
        "time": 10000,
        "user_number": 5,       #For correct displaying
        "starting_position": "random",
        "algorithm": "ge",               # must match the simulate_params value used
        #"algorithm": "mf",              # uncomment to target a CF run
    }

    # ----------------------------------------------------------
    # load_params -- used in "load_simulate" and
    # "load_simulate_analyse" modes.
    #
    # Identifies WHICH existing raw_data batch to load as the
    # starting point.  Must match the parameters used when that
    # batch was originally saved.
    #
    # "batch_id" (optional): exact timestamp folder to load,
    #     e.g. "20240601_153012_004321".
    #     When absent the most recent batch is selected automatically.
    # ----------------------------------------------------------
    load_params = {
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.75,
        "b_param": 0.6,
        "time": 10000,            # time of the ORIGINAL simulation to load
        "starting_position": "random",
        "algorithm": "ge",               # must match the original run's algorithm
        # "batch_id": "20240601_153012_004321",  # omit to use the most recent batch
    }

    # ----------------------------------------------------------
    # resume_params -- used in "load_simulate" and
    # "load_simulate_analyse" modes.
    #
    # Defines HOW to continue the simulation after loading.
    # "time" is the NEW absolute clock target (must be > the
    # clock values stored in the loaded batch).
    # Other parameters (ro, gamma, a_param, b_param) can differ
    # from the original run to experiment with changed dynamics.
    # ----------------------------------------------------------
    resume_params = {
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.75,
        "b_param": 0.6,
        "time": 20000,            # absolute target: users will run until clock=time
        "algorithm": "ba",        # algorithm for the resume phase
        "top_k": 30,              # neighbors used when algorithm is uk/ik/mf
    }

    # ----------------------------------------------------------
    # combined_plot_params -- used only in "cp" mode.
    # Identifies which combined_analysis_data to load (lsa output).
    # "t0" = time of the original loaded batch (load_params["time"])
    # "t1" = time of the resume phase (resume_params["time"])
    # algorithm_0/1 mirror the alg0/alg1 tokens in the combined
    # folder name; they must match what was used in load_params and
    # resume_params respectively.
    # "variant": "p" = pure (only the resumed phase data)
    #            "m" = mixed (full concatenated data, default)
    # ----------------------------------------------------------
    combined_plot_params = {
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.75,
        "b_param": 0.6,
        "t0": 10000,           # load_params["time"]
        "t1": 20000,           # resume_params["time"]
        "user_number": 5,
        "starting_position": "random",
        "algorithm_0": "ge",   # load_params["algorithm"]
        "algorithm_1": "ba",   # resume_params["algorithm"]
        "combined": True,       # tells load_analysis_data() to use combined loaders
        "variant": "m",        # "p" = pure (resumed phase only), "m" = mixed (full)
    }

    # ----------------------------------------------------------
    # multiple_plot_params -- used only in "mp" mode.
    #
    # Same structure as combined_plot_params, but:
    #   - "label" (required): list of strings — one label per series.
    #     The number of labels must equal the cartesian product of all
    #     list/tuple values among the other keys.
    #   - Any key whose value is a list or tuple is expanded: the cartesian
    #     product of all such keys generates one parameter set per series.
    #   - Scalar values are shared across all series.
    #   - "combined": True must be set (mp works on combined_analysis_data).
    #
    # Example with 3 series (algorithm_1 has 3 values):
    #
    #   multiple_plot_params = {
    #       "label":          ["ge→mf", "ge→ik", "ge→uk"],
    #       "ro": 0.1, "gamma": 0.2, "a_param": 0.75, "b_param": 0.6,
    #       "t0": 30000,
    #       "t1": 230000,
    #       "user_number": 1000, "starting_position": "random",
    #       "algorithm_0": "ge",
    #       "algorithm_1": ["mf", "ik", "uk"],
    #       "combined": True,
    #   }
    # ----------------------------------------------------------
    multiple_plot_params = {
        "label":          ["EPR mf", "EPR ik", "EPR uk", "EPR ge"],
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.75,
        "b_param": 0.6,
        "t0": 10000,
        "t1": 20000,
        "user_number": 5,
        "starting_position": "random",
        "algorithm_0": "ge",
        "algorithm_1": ["mf", "ik", "uk", "ge"],
        "combined": True,
        "variant": "p",
    }

    plot_list = [
        #"mobility",
        "rg_distribution", #checked but still double plot   #mp
        "dist_distribution", #mp
        "freq_distribution", #mp
        "two_radius", #mp
        "s_over_time", #mp
        "rg_over_time", #mp
        #"aggregate_grid",   
        "batch_grid",  ##Problems in saving due to max lenght titole 260 characters
        "batch_stats", #mp
        "srand", #mp     
        "sunc", #mp      
        "sreal", #mp     
        "s_triple",   
    ]

    show = True #if True only shows the plots, if False only saves them to disk

    start_time = time.time()
    logger.info(f"Running mode: {mode}")

    # ==========================================================
    # MODE 1: SIMULATE only
    # ==========================================================
    if mode == "s":

        simulation = model.simulation.Simulation()
        simulation.run(simulate_params, grid_params=grid_params)
        simulation.save_raw_data()

    # ==========================================================
    # MODE 2: ANALYSE only
    # ==========================================================
    elif mode == "a":

        simulation = model.simulation.Simulation()
        simulation.run_analysis(analyse_params, grid_params=grid_params)

    # ==========================================================
    # MODE 3: SIMULATE + ANALYSE (no double disk load)
    # ==========================================================
    elif mode == "sa":

        simulation = model.simulation.Simulation()
        simulation.run(simulate_params, grid_params=grid_params)
        raw_run_folder = simulation.save_raw_data()
        simulation.run_analysis(simulate_params, raw_run_folder=raw_run_folder, grid_params=grid_params)

    # ==========================================================
    # MODE 4: PLOT only
    # ==========================================================
    elif mode == "p":

        simulation = model.simulation.Simulation()
        simulation.load_analysis_data(plot_params, grid_params=grid_params, plot_list=plot_list)
        simulation.plot(plot_list, show=show)

    # ==========================================================
    # MODE 4b: COMBINED PLOT (output of lsa mode)
    # ==========================================================
    elif mode == "cp":

        simulation = model.simulation.Simulation()
        simulation.load_analysis_data(combined_plot_params, grid_params=grid_params, plot_list=plot_list)
        simulation.plot(plot_list, show=show)

    # ==========================================================
    # MODE 4c: MULTIPLEPLOT -- overlay several combined datasets
    #
    # multiple_plot_params drives cartesian expansion of parameters
    # (list values) into one labelled series each.  All series are
    # plotted on shared axes for direct comparison.
    #
    # Only plots that make sense overlaid are supported:
    #   "rg_distribution", "dist_distribution", "rg_over_time",
    #   "s_over_time", "two_radius", "batch_stats" (printed per series)
    # ==========================================================
    elif mode == "mp":

        simulation = model.simulation.Simulation()
        simulation.load_multiple_plot_data(
            multiple_plot_params, grid_params=grid_params, plot_list=plot_list
        )
        simulation.multipleplot(plot_list, show=show)

    # ==========================================================
    # MODE 5: LOAD + SIMULATE (no analysis)
    #
    # Loads an existing raw_data batch, reconstructs User objects,
    # continues the simulation with resume_params, then saves the
    # extended trajectories as a new raw_data batch.
    #
    # Typical use-case: branch from a shared EPR baseline and run
    # one variant (e.g. pure EPR extension) without re-simulating
    # from scratch.
    # ==========================================================
    elif mode == "ls":

        simulation = model.simulation.Simulation()
        simulation._reconstruct_users_from_raw(load_params, grid_params=grid_params)
        simulation.resume(resume_params, grid_params=grid_params)
        simulation.save_combined_raw_data()

    # ==========================================================
    # MODE 6: LOAD + SIMULATE + ANALYSE
    #
    # Same as MODE 5 but also runs analysis immediately after,
    # without re-reading from disk (in-memory chaining).
    # ==========================================================
    elif mode == "lsa":

        simulation = model.simulation.Simulation()
        simulation._reconstruct_users_from_raw(load_params, grid_params=grid_params)
        simulation.resume(resume_params, grid_params=grid_params)
        raw_run_folder = simulation.save_combined_raw_data()
        simulation.run_analysis(resume_params, raw_run_folder=raw_run_folder,
                                grid_params=grid_params, combined=True)

    else:
        raise ValueError(
            "Invalid mode. Use short aliases: 's', 'a', 'sa', 'p', 'cp', 'ls', 'lsa', 'mp'\n" 
            "'s' = simulate only\n" 
            "'a' = analyse only\n" 
            "'sa' = simulate then analyse without reloading\n"
            "'p' = plot only\n"
            "'cp' = plot combined_analysis_data (lsa output)\n"
            "'ls' = load a raw_data batch, continue simulation, save raw_data\n"
            "'lsa' = load a raw_data batch, continue simulation, analyse and save\n"
            "'mp' = multipleplot: overlay multiple combined datasets on shared axes"
        )

    elapsed = time.time() - start_time
    print(f"\nElapsed time: {elapsed:.4f} seconds\n")


if __name__ == "__main__":
    main()