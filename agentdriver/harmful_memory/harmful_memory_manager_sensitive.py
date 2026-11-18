# Calculates the delta l2 error
# Set threshold value for flagging harmful memory
# Track harmful memories
import math
import numbers


# Classify memory based on the delta L2 error
# High delta L2 error indicates potentially harmful memory
def memory_classifier(delta_memory):
    if delta_memory > 0.50:
        return "deletion_review"
    elif delta_memory > 0.25:
        return "reject"
    elif delta_memory > 0.10:
        return "flag"
    else:
        return "keep"

# Checks whether the current token exist in the baseline result
def token_look_up(dataset, current_token):
    if current_token in dataset:
        return True
    else:
        return False

# Retrieved the memory-free L2 error for the matching scenario
def get_l2_error(dataset, current_token):
    off_l2_error = dataset[current_token]["l2_error"]
    return off_l2_error
 
# Retrieve the predicted trajectory from the memory-free baseline
def get_pred_traj(dataset, current_token):
    off_pred_traj = dataset[current_token]["pred_traj"]
    return off_pred_traj

# Valide L2 values by rejecting non-numerical, non-finite and nagative values
def validate_l2_error(l2_error):
    if isinstance(l2_error, (bool, complex)) or not isinstance(l2_error, numbers.Real):
        return False
    if not math.isfinite(l2_error):
        return False
    if l2_error < 0:
        return False
    return True

# Compare the retrieved memory L2 error against the memory-free baseline
def calculate_memory_quality(mem_gt_l2_error, memory_off_l2_error):
        if mem_gt_l2_error is None or memory_off_l2_error is None:
            return None
        return mem_gt_l2_error - memory_off_l2_error

# Measure the impact ofmemory on planning error
def calculate_delta_l2_error(memory_on_l2_error, memory_off_l2_error):
    if memory_on_l2_error is None or memory_off_l2_error is None:
        return None
    return memory_on_l2_error - memory_off_l2_error

# Store retrieval counts and harmful outcomes for each memory
record_memory_result = {}

# Update the memory's retrieval history and harmful outcome count
def record_memory(mem_token, on_l2, off_l2, delta_l2, mem_classification):
    # Memory is harmful if the delta L2 error is greater than 0.25
    is_harmful = delta_l2 > 0.25

    if mem_token in record_memory_result:
        record = record_memory_result[mem_token]
        record["number_memory_retrieved"] += 1

        if is_harmful:
            record["number_harmful_outcomes"] += 1
        
        record["on_l2_error"] = on_l2
        record["off_l2_error"] = off_l2
        record["delta_l2_error"] = delta_l2
        record["memory_classification"] = mem_classification

    else:
        record_memory_result[mem_token]= {
            "memory_token": mem_token,
            "on_l2_error": on_l2,
            "off_l2_error": off_l2,
            "delta_l2_error": delta_l2,
            "memory_classification": mem_classification,
            "number_memory_retrieved": 1,
            "number_harmful_outcomes": 1 if is_harmful else 0
            }

# Memory deletion system
def memory_deletion_review(records, mem_token):
    if mem_token not in records:
        return False
    
    num_memory_retrieved = records[mem_token]["number_memory_retrieved"]
    num_harmful_outcomes = records[mem_token]["number_harmful_outcomes"]
    mem_classification = records[mem_token]["memory_classification"]

    # Recommend deletion only after at least three retrievals
    if num_memory_retrieved < 3:
        return False

    harmful_rate = num_harmful_outcomes / num_memory_retrieved

    #harmful_rate = num_harmful_outcomes / num_memory_retrieved
    if harmful_rate >= 0.7 and mem_classification == "deletion_review":
        return True
    else:
        return False
