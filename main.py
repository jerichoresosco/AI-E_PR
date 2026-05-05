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

    # ----------------------------------------------------------
    # simulate_params -- used only in "simulate" and
    # "simulate_and_analyse" modes.
    # ----------------------------------------------------------
    simulate_params = {
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.75,
        "b_param": 0.6,
        "user_number": 1000,
        "time": 30000,
        "grid" : True,
        "starting_position": "random",   # "centered" | "random" | "normal"
        "epr_fraction": 1,      # fraction of 'time' run with pure EPR before CF (1.0 = pure EPR; 0 < x < 1 enables CF phase)
        "cf_type": "ik",          # CF algorithm: "uk" = User-KNN, "ik" = Item-KNN, "mf" = Matrix Factorization
        "top_k": 10,              # number of similar neighbors used in CF phase
    }

    ##TOPK STILL NOT IN THE SAVING/PLOT PROCEDURE!!!!!1
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
        "time": 1000,
        "grid": True,
        "starting_position": "normal",   # must match the simulate_params value used
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
        "time": 30000,
        "user_number": 1000,       #For correct displaying
        "grid": True,
        "starting_position": "random",
        "epr_fraction": 1,
        #"cf_type": "mf",         # must match the simulate_params value used
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
        "time": 30000,            # time of the ORIGINAL simulation to load
        "grid": True,
        "starting_position": "random",
        "epr_fraction": 1,      # include only if the original run used CF
        "cf_type": "ik",          # must match the cf_type of the original run
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
        "time": 230000,            # absolute target: users will run until clock=time in resume params
        #"grid": False,
        #"starting_position": "normal",
        "epr_fraction": 1,      # 1.0 = pure EPR resume; <1.0 = EPR then CF
        "cf_type": "mf",          # CF algorithm for the resume phase: "uk" or "ik"
        "top_k": 30,
    }

    # ----------------------------------------------------------
    # combined_plot_params -- used only in "cp" mode.
    # Identifies which combined_analysis_data to load (lsa output).
    # "t0" = time of the original loaded batch (load_params["time"])
    # "t1" = time of the resume phase (resume_params["time"])
    # epr_fraction_0/1 and cf_type_0/1 mirror the ef/cf tokens in
    # the combined folder name; omit or set to defaults when unused.
    # ----------------------------------------------------------
    combined_plot_params = {
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.75,
        "b_param": 0.6,
        "t0": 30000,           # load_params["time"]
        "t1": 230000,           # resume_params["time"]
        "user_number": 1000,
        "grid": True,
        "starting_position": "random",
        "epr_fraction_0": 1,    # load_params["epr_fraction"]  (1 = pure EPR)
        "epr_fraction_1": 1,    # resume_params["epr_fraction"] (1 = pure EPR)
        # "cf_type_0": "ik",    # uncomment if epr_fraction_0 < 1
        "cf_type_1": "mf",    # uncomment if epr_fraction_1 < 1
        "combined": True,       # tells load_analysis_data() to use combined loaders
    }

    plot_list = [
        #"mobility",
        #"rg_distribution", #checked but still double plot
        #"freq_distribution", 
        #"two_radius", 
        #"s_over_time", 
        "rg_over_time",
        #"aggregate_grid",   
        #"batch_grid",  ##Problems in saving due to max lenght titole 260 characters
        #"batch_stats"     
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
            "Invalid mode. Use short aliases: 's', 'a', 'sa', 'p', 'cp', 'ls', 'lsa'\n" 
            "'s' = simulate only\n" 
            "'a' = analyse only\n" 
            "'sa' = simulate then analyse without reloading\n"
            "'p' = plot only\n"
            "'cp' = plot combined_analysis_data (lsa output)\n"
            "'ls' = load a raw_data batch, continue simulation, save raw_data\n"
            "'lsa' = load a raw_data batch, continue simulation, analyse and save"
        )

    elapsed = time.time() - start_time
    print(f"\nElapsed time: {elapsed:.4f} seconds\n")


if __name__ == "__main__":
    main()