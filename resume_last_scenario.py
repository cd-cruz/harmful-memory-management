import json
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from agentdriver.llm_core.api_keys import FINETUNE_PLANNER_NAME
from agentdriver.main.language_agent import LanguageAgent
from agentdriver.memory.memory_agent import MemoryAgent
from agentdriver.perception.perception_agent import PerceptionAgent
from agentdriver.planning.planning_agent import PlanningAgent
from agentdriver.reasoning.reasoning_agent import ReasoningAgent

# Keep the original experiment untouched; resume into a new directory.
SOURCE_EXPERIMENT_NAME = "qwen3_14b_20_test_v3"
EXPERIMENT_NAME = "qwen3_14b_20_test_v3_recovered"
COMPLETED_SCENARIOS = 19
FAILED_TOKEN = "9c9cdfaf964f4776b30f1edbdf9adb0b"
RANDOM_SEED = 0
N_SAMPLES = 20
SUCCESS_L2_THRESHOLD = 2.5  # add when l2 < this. 0.0 = no add, 99999.0 = always add
DELETION_L2_THRESHOLD = 99999.0  # del when history l2 > this. 99999.0 = no deletion    #2.5
DELETION_RETREIVAL_N_TIMES = 99999  # del when ret time >= this. 99999 = no deletion      #3
CLEANUP_RETRIEVAL_N_TIMES = 0  # clean up when ret time < this. 0 = no cleanup            #1
EXIST_TIME_THRESHOLD = 99999  # clean up when exist time > this. 99999 = no cleanup        #500
MIN_MEMORY_SIZE = 1  # won't delete when < this
MAX_MEMORY_SIZE = 99999  # force delete (ignore l2 threshold) mem when > this (but will still add)
ADD_GT = False  # WE NEVER USED THIS OPTION. TO RUN ERROR-FREE EXPERIMENT, USE `./main_noerrorprop.py`.


random.seed(RANDOM_SEED)


def convert_working_memory_to_memory(working_memory, perception_agent, planning_traj):
    """
    Convert working_memory to the format required by the memory database.

    Args:
        working_memory (dict): The working memory dictionary.
        perception_agent (PerceptionAgent): The perception agent instance.
        planning_traj (tuple): The planning trajectory output from the planning agent.

    Returns:
        dict: A dictionary in the format required by the memory database.
    """
    # Get ego_data from working_memory
    ego_data = working_memory["ego_data"]
    # Add 'objects' from perception_agent.data_dict
    ego_data["objects"] = perception_agent.data_dict.get("objects", [])
    # Add 'ego_fut_traj' from planning_traj[0]
    # Ensure that ego_hist_traj exists
    if "ego_hist_traj" in ego_data and ego_data["ego_hist_traj"].shape[0] > 0:
        last_ego_point = ego_data["ego_hist_traj"][-1]
    else:
        # If ego_hist_traj is empty
        raise ValueError("ego_hist_traj is empty")
    # Construct ego_fut_traj by adding last_ego_point to the beginning
    predicted_traj = planning_traj  # Shape (6, 2)
    ego_fut_traj = np.vstack([last_ego_point, predicted_traj])  # Shape (7, 2)
    ego_data["ego_fut_traj"] = ego_fut_traj
    # Compute ego_fut_traj_diff
    ego_fut_traj_diff = np.diff(ego_fut_traj, axis=0)  # Shape (6, 2)
    ego_data["ego_fut_traj_diff"] = ego_fut_traj_diff

    # Prepare the memory entry
    memory_entry = {"token": working_memory["token"], "ego_data": ego_data}
    return memory_entry


def eval_l2(pred_traj, gt_traj, metric="uniad"):
    assert metric in ["uniad", "stp3"], f"Invalid metric: {metric}"
    assert pred_traj.shape == gt_traj.shape, "Shape mismatch between pred and gt trajs"
    assert pred_traj.shape == (6, 2), "Invalid shape for pred_traj and gt_traj"
    l2s = np.sqrt(np.sum((pred_traj - gt_traj) ** 2, axis=1))
    assert l2s.shape == (6,), "DEBUG: Invalid shape for l2s"
    if metric == "uniad":
        avg = (l2s[1] + l2s[3] + l2s[5]) / 3
    elif metric == "stp3":
        l2s_stp3 = []
        for i in range(6):
            l2s_stp3.append(np.mean(l2s[: i + 1]))
        avg = (l2s_stp3[1] + l2s_stp3[3] + l2s_stp3[5]) / 3

    return avg


print(f"Experiment name: {EXPERIMENT_NAME}")
print(f"Experiment time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Random seed: {RANDOM_SEED}")
print(f"Number of samples: {N_SAMPLES}")
print(f"Success L2 threshold: {SUCCESS_L2_THRESHOLD}")
print(f"Deletion L2 threshold: {DELETION_L2_THRESHOLD}")
print(f"Deletion retrieval n times: {DELETION_RETREIVAL_N_TIMES}")
print(f"Cleanup retrieval n times: {CLEANUP_RETRIEVAL_N_TIMES}")
print(f"Exist time threshold: {EXIST_TIME_THRESHOLD}")
print(f"Minimum memory size: {MIN_MEMORY_SIZE}")
print(f"Maximum memory size: {MAX_MEMORY_SIZE}")
print(f"Add Ground-Truth: {ADD_GT}")
print(f"FINETUNE_PLANNER_NAME: {FINETUNE_PLANNER_NAME}")
source_dir = Path("results") / SOURCE_EXPERIMENT_NAME
output_dir = Path("results") / EXPERIMENT_NAME
if not source_dir.is_dir():
    raise FileNotFoundError(f"Original results folder not found: {source_dir}")
if output_dir.exists():
    raise FileExistsError(
        f"Recovery folder already exists: {output_dir}. "
        "Choose another EXPERIMENT_NAME rather than overwriting it."
    )
required_files = (
    "records.pkl", "prediction_results.pkl", "memory.pkl",
    "success_stats.pkl", "retrieval_records.pkl",
    "retrieval_records_reverse.pkl", "deletion_records.pkl",
    "exist_time.pkl", "retrieval_timesteps.pkl",
)
for filename in required_files:
    if not (source_dir / filename).is_file():
        raise FileNotFoundError(f"Missing checkpoint: {source_dir / filename}")

# Only the new folder will be modified during recovery.
shutil.copytree(source_dir, output_dir)
