# Loads the baseline results
import pickle


# Define the baseline experiment diectories
experiments = {
    "memory_on": "qwen3_14b_200_baseline_memory_on",
    "memory_off": "qwen3_14b_200_baseline_memory_off"
}


# Retrieves and load the results of the experiment
def read_results(experiment):
    with open(f"results/{experiments[experiment]}/records.pkl", "rb") as results_file:
        results = pickle.load(results_file)
    return results

# Loads memory-free results
memory_off_results_pkl = read_results("memory_off")

# INdex valid memory-free results by scenario token for direct comparison
# Exclude records with missing token, L2 error or predicted trajectory
memory_off_results = {
    record["token"]: {
        "l2_error": record["l2_error"],
        "pred_traj": record["pred_traj"]
    }
    
    for record in memory_off_results_pkl
    if record.get("token") is not None
    and record.get("l2_error") is not None
    and record.get("pred_traj") is not None
}
