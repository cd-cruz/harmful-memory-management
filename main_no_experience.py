import json
import os
import pickle
import random
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

EXPERIMENT_NAME = "qwen3_14b_200_baseline_memory_off"
RANDOM_SEED = 0
N_SAMPLES = 200
SUCCESS_L2_THRESHOLD = 2.5  # add when l2 < this. 0.0 = no add, 99999.0 = always add
DELETION_L2_THRESHOLD = 99999.0  # del when history l2 > this. 99999.0 = no deletion    #2.5
DELETION_RETREIVAL_N_TIMES = 99999  # del when ret time >= this. 99999 = no deletion      #3
CLEANUP_RETRIEVAL_N_TIMES = 0  # clean up when ret time < this. 0 = no cleanup            #1
EXIST_TIME_THRESHOLD = 99999  # clean up when exist time > this. 99999 = no cleanup        #500
MIN_MEMORY_SIZE = 1  # won't delete when < this
MAX_MEMORY_SIZE = 99999  # force delete (ignore l2 threshold) mem when > this (but will still add)
ADD_GT = False  # WE NEVER USED THIS OPTION. TO RUN ERROR-FREE EXPERIMENT, USE `./main_noerrorprop.py`.
USE_EXPERIENCE_MEMORY = False

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
os.makedirs(f"./results/{EXPERIMENT_NAME}", exist_ok=True)

# Set the data path and split
data_path = Path("./data/")
split = "val"

# Load the split data
with open(data_path / "split.json", "r") as f:
    split_data = json.load(f)


##### MODIFY THE SECTION BELOW #####
# The following section will decide which data split to use.
# To run a *NORMAL* experiment, uncomment the *NORMAL* section and comment the *DISTRIBUTION SHIFT* section.
# To run a *DISTRIBUTION SHIFT* experiment, uncomment the *DISTRIBUTION SHIFT* section and comment the *NORMAL* section.

#### NORMAL
val_tokens = split_data["val"]
token_list = random.sample(val_tokens, N_SAMPLES)
assert "distribshift" not in EXPERIMENT_NAME

#### DISTRIBUTION SHIFT
# val_tokens = split_data["distribshift"]
# token_list = val_tokens
# assert token_list[0] == "64d019bf33ba4dcb9eca5e5ab2ef967e"
# assert "distribshift" in EXPERIMENT_NAME

##### END OF MODIFICATION #####


assert len(token_list) == N_SAMPLES

# For evaluation, Load the ground truth trajectory
with open("./data/metrics/gt_traj.pkl", "rb") as f:
    gt_data = pickle.load(f)

# Initialize the LanguageAgent
language_agent = LanguageAgent(
    data_path,
    split,
    model_name=FINETUNE_PLANNER_NAME,
    planner_model_name=FINETUNE_PLANNER_NAME,
    verbose=False,
)
language_agent.tokens = token_list

# Initialize the results dictionary
results = {}

# Initialize MemoryAgent, ReasoningAgent, and PlanningAgent
print("Initializing agents...")
print("Initializing MemoryAgent...")
memory_agent = MemoryAgent(
    data_path=data_path,
    model_name=language_agent.model_name,
    verbose=language_agent.verbose,
)
assert memory_agent.experience_memory.get_n_remaining_memory() == 180

print("Initializing ReasoningAgent...")
reasoning_agent = ReasoningAgent(
    model_name=language_agent.model_name, verbose=language_agent.verbose
)
print("Initializing PlanningAgent...")
planning_agent = PlanningAgent(
    model_name=language_agent.planner_model_name, verbose=language_agent.verbose
)


# For each token, run the pipeline
print("Running the pipeline...")
invalid_tokens = []
success_count = 0
records = []
time_step = 0
for token in tqdm(token_list, desc="Processing tokens"):
    sys.stdout.flush()
    # Make sure the token is in the language_agent tokens
    assert token in language_agent.tokens
    time_step += 1

    try:
        # Create PerceptionAgent
        perception_agent = PerceptionAgent(
            token=token,
            split=split,
            data_path=data_path,
            model_name=language_agent.model_name,
            verbose=language_agent.verbose,
        )
        # Perception
        ego_prompts, perception_prompts, working_memory = perception_agent.run()
        # Memory
        # Commented to not use experience
        # commonsense_mem, experience_mem = memory_agent.run(working_memory)
        commonsense_mem = memory_agent.retrieve_common_sense_memory()
        experience_mem = ""
        assert not USE_EXPERIENCE_MEMORY
        
        # Reasoning
        reasoning = reasoning_agent.run(
            data_dict=perception_agent.data_dict,
            env_info_prompts=ego_prompts + perception_prompts,
            working_memory=working_memory,
            use_cot_rules=language_agent.finetune_cot,
        )
        # Planning
        planning_target = planning_agent.generate_planning_target(
            perception_agent.data_dict
        )
        data_sample = {
            "token": token,
            "ego": ego_prompts,
            "perception": perception_prompts,
            "commonsense": commonsense_mem,
            "experiences": experience_mem,
            "chain_of_thoughts": reasoning,
            "reasoning": reasoning,
            "planning_target": planning_target,
        }
        planning_traj, planning_info = planning_agent.run(
            data_dict=perception_agent.data_dict,
            data_sample=data_sample,
        )
        # Save the planning trajectory to the results dictionary
        results[token] = planning_traj  # Shape (6, 2)

        ##### Evaluation
        ground_truth = gt_data[token][0]
        # Compute L2 error
        # l2_error = np.linalg.norm(ground_truth - planning_traj, ord=2)
        l2_error = eval_l2(planning_traj, ground_truth, metric="uniad")
        # Record this retrieve. No matter if succeeded or not
        # Change to None to disable memory
        memory_token = None
        memory_traj = None
        pred_mem_l2_error = None
        mem_gt_l2_error = None
        input_similarity = None
        output_similarity_cos = None
        output_similarity_exp = None
        
        
        is_success = l2_error < SUCCESS_L2_THRESHOLD
        if is_success:
            print(F"Token {token} succeeded")
            success_count += 1
        else:
            print(f"Token {token} failed")
            
    
        # Save the results to a pickle file
        with open(f"./results/{EXPERIMENT_NAME}/prediction_results.pkl", "wb") as f:
            pickle.dump(results, f)

        records.append(
            {
                "token": token,
                "l2_error": l2_error,
                "pred_mem_l2_error": pred_mem_l2_error,
                "mem_gt_l2_error": mem_gt_l2_error,
                "success": is_success,
                "pred_traj": planning_traj,
                "gt_traj": ground_truth,
                "memory_token": memory_token,
                "memory_traj": memory_traj,
                "input_similarity": input_similarity,
                "output_similarity_cos": output_similarity_cos,
                "output_similarity_exp": output_similarity_exp,
            }
        )
        
        with open(f"./results/{EXPERIMENT_NAME}/records.pkl", "wb" as f):
            pickle.dump(records , f)
            
    except Exception as e:
        print(f"An error occurred for token {token}:")
        print(f"{type(e).__name__}: {e}")
        invalid_tokens.append(token)
        # raise e       # TO DEBUG, UNCOMMENT THIS
        continue

print(f"Invalid tokens: {invalid_tokens}")
print(
    f"Success count: {success_count}/{len(token_list)} ({success_count/len(token_list)*100:.2f}%)"
)

# Debugging
if invalid_tokens or len(records) != N_SAMPLES:
    raise RuntimeError(
        f"Incomplete experiment: {len(records)}/{N_SAMPLES} "
        f"valid scenarios; invalid tokens: {invalid_tokens}"
    )