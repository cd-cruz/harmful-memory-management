# This script is used to convert the results of experiments, memory on and off, into CSV format for easier analysis. 
# It reads the results from pickle files and writes them to CSV files.

import pickle
import csv


# Experiments to compare: memory on and off
experiments = {
    "memory_on": "qwen3_14b_200_baseline_memory_on",
    "memory_off": "qwen3_14b_200_baseline_memory_off"
}


# Retrieve results of the experiments with memory on
def read_memory_on():
    with open(f"results/{experiments['memory_on']}/records.pkl", "rb") as memory_on_results_file:
        memory_on_results = pickle.load(memory_on_results_file)
    return memory_on_results

memory_on_results = read_memory_on()


# Converting the results to CSV format for easier analysis
def write_memory_on_results_to_csv(results_memory_on, filename):
    with open(filename, mode='w', newline='') as memory_on_csv_file:   
        writer = csv.DictWriter(memory_on_csv_file, fieldnames = results_memory_on[0].keys())
        writer.writeheader()
        writer.writerows(results_memory_on)

write_memory_on_results_to_csv(memory_on_results, "results_csv/memory_on_results.csv")


# Retrieve results of the experiments with memory off
def read_memory_off():
    with open(f"results/{experiments['memory_off']}/records.pkl", "rb") as memory_off_results_file:
        memory_off_results = pickle.load(memory_off_results_file)
    return memory_off_results

memory_off_results = read_memory_off()

# Converting the results to CSV format for easier analysis
def write_memory_off_results_to_csv(results_memory_off, filename):
    with open(filename, mode='w', newline='') as memory_off_csv_file:   
        writer = csv.DictWriter(memory_off_csv_file, fieldnames = results_memory_off[0].keys())
        writer.writeheader()
        writer.writerows(results_memory_off)

write_memory_off_results_to_csv(memory_off_results, "results_csv/memory_off_results.csv")