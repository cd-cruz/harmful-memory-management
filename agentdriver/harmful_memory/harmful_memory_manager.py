# Calculates the delta l2 error
# Set threshold value for flagging harmful memory
# Track harmful memories

from baseline_result_loader import memory_off_results
import pickle


def memory_classifier(delta_memory):
    if delta_memory > 1.0:
        return "deletion_review"
    elif delta_memory > 0.5:
        return "reject"
    elif delta_memory > 0.25:
        return "flag"
    else:
        return "keep"


def token_look_up(dataset, current_token):
    if current_token not in dataset:
        return "Token not found"


def get_l2_error(dataset, current_token):
    if current_token in dataset:
        off_l2_error = dataset[current_token]["l2_error"]
        return off_l2_error
    else:
        return "Token not found"


def get_pred_traj(dataset, current_token):
    if current_token in dataset:
        off_pred_traj = dataset[current_token]["pred_traj"]
        return off_pred_traj
    else:
        return "Token not found"


def calculate_memory_quality(mem_gt_l2_error, memory_off_l2_error):
    return mem_gt_l2_error - memory_off_l2_error


def calculate_delta_l2_error(memory_on_l2_error, memory_off_l2_error):
    return memory_on_l2_error - memory_off_l2_error



