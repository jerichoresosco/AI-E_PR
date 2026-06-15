import logging
import time
import model.simulation
import copy

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

grid_params = {
    "cell_lenght": 100,
    "cell_number": 15
}

# Identifies which raw_data batch to load as the starting point.
# Must match the parameters used when that batch was originally saved.
# "algorithm" must be "ge" (the EPR grid run that produced the base trajectories).
load_params = {
    "ro": 0.1,
    "gamma": 0.2,
    "a_param": 0.75,
    "b_param": 0.6,
    "time": 30000,
    "starting_position": "random",
    "algorithm": "ge",   # algorithm of the original EPR run to load
    # "batch_id": "20240601_153012_004321",  # omit to use the most recent batch
}

# Base resume parameters shared across all configurations.
# "algorithm" is injected per-configuration below.
resume_base = {
    "ro": 0.1,
    "gamma": 0.2,
    "a_param": 0.75,
    "b_param": 0.6,
    "time": 150000,   # absolute clock target (must be > load_params["time"])
    "top_k": 40,      # neighbors used by CF algorithms (uk/ik/mf); ignored by ge/ba
}

# One dict per run.  "algorithm" maps directly to the resume algorithm:
#   "ge"  → pure EPR continuation (no collaborative filtering)
#   "uk"  → User-KNN CF
#   "ik"  → Item-KNN CF
#   "mf"  → Matrix Factorization CF
#   "ba"  → Barabási-Albert popularity CF
configurations = [
    {"algorithm": "ba"},
    {"algorithm": "bv"},
    {"algorithm": "ge"},
    {"algorithm": "uk"},
    {"algorithm": "ik"},
    {"algorithm": "mf"},
    {"algorithm": "ra"}
]

# --- Build users once and reuse across all runs ---
logger.info("Reconstructing users from raw data (one-time operation)...")
base_simulation = model.simulation.Simulation()
base_simulation._reconstruct_users_from_raw(load_params, grid_params=grid_params)
logger.info("User reconstruction complete.")

total_start = time.time()

for i, cfg in enumerate(configurations):
    alg = cfg["algorithm"]
    label = "Pure EPR" if alg == "ge" else f"CF={alg.upper()}"
    logger.info(f"\n{'='*60}")
    logger.info(f"Run {i+1}/{len(configurations)}: {label}  (algorithm={alg})")
    logger.info(f"{'='*60}")

    resume_params = {**resume_base, **cfg}

    run_start = time.time()
    try:
        # Deep-copy the base simulation to avoid mutating shared user state
        simulation = copy.deepcopy(base_simulation)
        simulation.resume(resume_params, grid_params=grid_params)
        raw_run_folder = simulation.save_combined_raw_data()
        simulation.run_analysis(
            resume_params,
            raw_run_folder=raw_run_folder,
            grid_params=grid_params,
            combined=True
        )
        elapsed = time.time() - run_start
        logger.info(f"Run {i+1} completed in {elapsed:.1f}s")
    except Exception as e:
        logger.error(f"Run {i+1} FAILED: {e}", exc_info=True)

total_elapsed = time.time() - total_start
logger.info(f"\nAll runs completed in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")