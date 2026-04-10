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

##Problems: Double saving on raw folder, ad also this error message: 2026-04-09 12:35:09 - model.simulation - INFO - Save progress: 10% (3/30 users saved)
#2026-04-09 12:35:09 - model.simulation - INFO - Save progress: 20% (6/30 users saved)
#2026-04-09 12:35:09 - model.simulation - INFO - Save progress: 30% (9/30 users saved)
#Traceback (most recent call last):
  #File "c:\Users\Jacopo\Desktop\Tesi_Magistrale_AKA_La_Fine\AI-E_PR\main.py", line 242, in <module>
  #  main()
  #File "c:\Users\Jacopo\Desktop\Tesi_Magistrale_AKA_La_Fine\AI-E_PR\main.py", line 225, in main
  #  raw_run_folder = simulation.save_combined_raw_data()
  #File "c:\Users\Jacopo\Desktop\Tesi_Magistrale_AKA_La_Fine\AI-E_PR\model\simulation.py", line 851, in save_combined_raw_data
  #  user_df.to_parquet( os.path.join(run_folder, f"user_{i}_history.parquet"), index=False)
 # File "C:\Users\Jacopo\miniconda3\envs\GSA\lib\site-packages\pandas\util\_decorators.py", line 207, in wrapper
  #  return func(*args, **kwargs)
  #File "C:\Users\Jacopo\miniconda3\envs\GSA\lib\site-packages\pandas\core\frame.py", line 2835, in to_parquet
  #  return to_parquet(
  #File "C:\Users\Jacopo\miniconda3\envs\GSA\lib\site-packages\pandas\io\parquet.py", line 420, in to_parquet
  #  impl.write(
  #File "C:\Users\Jacopo\miniconda3\envs\GSA\lib\site-packages\pandas\io\parquet.py", line 176, in write
  #    path_or_handle, handles, kwargs["filesystem"] = _get_path_or_handle(
  #File "C:\Users\Jacopo\miniconda3\envs\GSA\lib\site-packages\pandas\io\parquet.py", line 102, in _get_path_or_handle
  #    handles = get_handle(
  #File "C:\Users\Jacopo\miniconda3\envs\GSA\lib\site-packages\pandas\io\common.py", line 795, in get_handle
   # handle = open(handle, ioargs.mode)
#FileNotFoundError: [Errno 2] No such file or directory: 'c:\\Users\\Jacopo\\Desktop\\Tesi_Magistrale_AKA_La_Fine\\AI-E_PR\\combined_raw_data\\ro_0.1_gamma_0.2_a_0.75_b_0.6_time_10000_clen_21_cnum_21_rstart_yes\\ro_0.1_gamma_0.2_a_0.75_b_0.6_time_20000_clen_21_cnum_21_rstart_yes\\20260409_123509_056420\\user_10_history.parquet'
#PS C:\Users\Jacopo\Desktop\Tesi_Magistrale_AKA_La_Fine\AI-E_PR> 

def main():

    # ==========================================================
    # Choose mode
    # ==========================================================

    mode = "load_simulate_analyse"  # Options:
                        #   "simulate"             - run simulation, save raw_data only
                        #   "analyse"              - load raw_data, compute & save analysis_data
                        #   "simulate_and_analyse" - simulate then analyse without reloading from disk
                        #   "plot"                 - load analysis_data and plot
                        #   "load_simulate"        - load a raw_data batch, continue simulation, save raw_data
                        #   "load_simulate_analyse"- load a raw_data batch, continue simulation, analyse and save

    # ----------------------------------------------------------
    # simulate_params -- used only in "simulate" and
    # "simulate_and_analyse" modes.
    # ----------------------------------------------------------
    simulate_params = {
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.75,
        "b_param": 0.6,
        "user_number": 30,
        "time": 10000,
        "grid" : True,
        "random_start": True,
        "epr_fraction": 0.3,      # fraction of 'time' run with pure EPR before CF (1.0 = pure EPR; 0 < x < 1 enables CF phase)
        "top_k": 10,                # number of similar neighbors used in CF phase
    }

    ##TOPK STILL NOT IN THE SAVING/PLOT PROCEDURE!!!!!1

    grid_params = {
        "cell_lenght": 21, #must be odd to have a central cell
        "cell_number": 21
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
        "time": 1000000,
        "grid" : True
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
        "user_number": 30,       #For correct displaying
        "grid": True,
        "random_start": True,      # must match the simulate_params value used
        "epr_fraction": 0.3,      
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
        "grid": True,
        "random_start": True,
        # "epr_fraction": 0.3,   # include only if the original run used CF
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
        "time": 20000,            # absolute target: users will run until clock=20000
        "grid": True,
        "random_start": True,
        "epr_fraction": 1.0,      # 1.0 = pure EPR resume; <1.0 = EPR then CF
        "top_k": 10,
    }

    plot_list = [
        #"mobility",
        #"rg_distribution", #checked but still double plot
        #"freq_distribution", 
        #"two_radius", 
        #"s_over_time", 
        #"rg_over_time",
        #"aggregate_grid",   
        #"batch_grid",  ##Problems in saving due to max lenght titole 260 characters
        "batch_stats"     
    ]

    show = False #if True only shows the plots, if False only saves them to disk

    start_time = time.time()
    logger.info(f"Running mode: {mode}")

    # ==========================================================
    # MODE 1: SIMULATE only
    # ==========================================================
    if mode == "simulate":

        simulation = model.simulation.Simulation()
        simulation.run(simulate_params, grid_params=grid_params)
        simulation.save_raw_data()

    # ==========================================================
    # MODE 2: ANALYSE only
    # ==========================================================
    elif mode == "analyse":

        simulation = model.simulation.Simulation()
        simulation.run_analysis(analyse_params, grid_params=grid_params)

    # ==========================================================
    # MODE 3: SIMULATE + ANALYSE (no double disk load)
    # ==========================================================
    elif mode == "simulate_and_analyse":

        simulation = model.simulation.Simulation()
        simulation.run(simulate_params, grid_params=grid_params)
        raw_run_folder = simulation.save_raw_data()
        simulation.run_analysis(simulate_params, raw_run_folder=raw_run_folder, grid_params=grid_params)

    # ==========================================================
    # MODE 4: PLOT only
    # ==========================================================
    elif mode == "plot":

        simulation = model.simulation.Simulation()
        simulation.load_analysis_data(plot_params, grid_params=grid_params, plot_list=plot_list)
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
    elif mode == "load_simulate":

        simulation = model.simulation.Simulation()

        # Reconstruct User objects from the chosen raw_data batch
        simulation._reconstruct_users_from_raw(load_params, grid_params=grid_params)

        # Continue the simulation with the new parameters
        simulation.resume(resume_params, grid_params=grid_params)

        # Save the extended trajectories under combined_raw_data/
        simulation.save_combined_raw_data()

    # ==========================================================
    # MODE 6: LOAD + SIMULATE + ANALYSE
    #
    # Same as MODE 5 but also runs analysis immediately after,
    # without re-reading from disk (in-memory chaining).
    # ==========================================================
    elif mode == "load_simulate_analyse":

        simulation = model.simulation.Simulation()

        # Reconstruct User objects from the chosen raw_data batch
        simulation._reconstruct_users_from_raw(load_params, grid_params=grid_params)

        # Continue the simulation with the new parameters
        simulation.resume(resume_params, grid_params=grid_params)

        # Save extended raw trajectories under combined_raw_data/
        raw_run_folder = simulation.save_combined_raw_data()

        # Analyse and save under combined_analysis_data/
        simulation.run_analysis(resume_params, raw_run_folder=raw_run_folder,
                                grid_params=grid_params, combined=True)

    else:
        raise ValueError(
            "Invalid mode. Use: 'simulate', 'analyse', 'simulate_and_analyse', "
            "'plot', 'load_simulate', or 'load_simulate_analyse'"
        )

    elapsed = time.time() - start_time
    print(f"\nElapsed time: {elapsed:.4f} seconds\n")


if __name__ == "__main__":
    main()