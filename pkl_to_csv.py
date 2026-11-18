# This script convert the results of the experiments from pickle file into csv

import pickle
import csv

experiments = {
    "memory_on": "qwen3_14b_200_baseline_memory_on",
    "memory_off": "qwen3_14b_200_baseline_memory_off",
    "harmful_memory_sensitive": "qwen3_14b_200_harmful_memory_sensitive",
    "harmful_memory_balance": "qwen3_14b_200_harmful_memory_balance",
    "harmful_memory_conservative": "qwen3_14b_200_harmful_memory_conservative",
}

# Retrieve the results of the experiment
def read_results(experiment):
    with open(f"results/{experiments[experiment]}/records.pkl", "rb") as results_file:
        results = pickle.load(results_file)
    return results

# Convert pickle to csv
def convert_to_csv(results, filename):
    with open(filename, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
# Memory On
#memory_on_results = read_results("memory_on")
#convert_to_csv(memory_on_results, "results_csv/memory_on_results.csv")

# Memory Off
#memory_off_results = read_results("memory_off")
#convert_to_csv(memory_off_results, "results_csv/memory_off_results.csv")

# Harmful memory
#harmful_memory_sensitive = read_results("harmful_memory_sensitive")
#convert_to_csv(harmful_memory_sensitive, "results_csv/harmful_memory_sensitive.csv")

#harmful_memory_balance = read_results("harmful_memory_balance")
#onvert_to_csv(harmful_memory_balance, "results_csv/harmful_memory_balance.csv")

harmful_memory_conservative = read_results("harmful_memory_conservative")
convert_to_csv(harmful_memory_conservative, "results_csv/harmful_memory_conservative.csv")