import openai
import pickle
import json
import ast
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
import os
import re

from agentdriver.planning.planning_prmopts import planning_system_message as system_message
from agentdriver.llm_core.chat import run_one_round_conversation
from agentdriver.reasoning.collision_check import collision_check
from agentdriver.reasoning.collision_optimization import collision_optimization
from agentdriver.llm_core.api_keys import OPENAI_ORG, OPENAI_API_KEY

openai.organization = OPENAI_ORG
openai.api_key = OPENAI_API_KEY

def generate_messages(data_sample, use_peception=True, use_short_experience=True, verbose=True, use_gt_cot=False):
    token = data_sample["token"]
    ego = data_sample["ego"]
    perception = data_sample["perception"]
    commonsense = data_sample["commonsense"]
    experiences =  data_sample["experiences"]
    reasoning = data_sample["reasoning"]
    long_experiences = data_sample["long_experiences"] if "long_experiences" in data_sample else None
    chain_of_thoughts = data_sample["chain_of_thoughts"] if "chain_of_thoughts" in data_sample else ""
    planning_target = data_sample["planning_target"] if "planning_target" in data_sample else None

    user_message = ego
    if use_peception:
        user_message += perception
    if use_short_experience:
        if experiences:
            user_message += experiences
    else:
        if long_experiences:
            user_message += long_experiences
    user_message += commonsense
    if use_gt_cot:
        user_message += chain_of_thoughts
    else:
        user_message += reasoning
    
    assistant_message = planning_target

    if verbose:
        print(user_message)
        print(assistant_message)
    
    return token, user_message, assistant_message

def planning_single_inference(
        planner_model_id, 
        data_sample, 
        data_dict=None, 
        self_reflection=True,
        safe_margin=1., 
        occ_filter_range=5.0, 
        sigma=1.0, 
        alpha_collision=5.0, 
        verbose=True
    ):

    token, user_message, assitant_message = generate_messages(data_sample, verbose=False)

    planner_prompt = (system_message + "\n"
        "Return ONLY six FUTURE (x,y) waypoints at these exact times: "
        "0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s. "
        "Do NOT include the vehicle's current position at t=0s. "
        "Coordinates must be relative to the current ego position. "
        "Use this exact Python format: "
        "[(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6)]. "
        "Replace x1,y1,etc. with actual numerical coordinates. "
        "Do not include explanations, Markdown, or equations."
        if planner_model_id == "qwen3:14b"
        else system_message
    )

    full_messages, response_message = run_one_round_conversation(
        full_messages=[],
        system_message=planner_prompt,
        user_message=user_message,
        temperature=0.0,
        model_name=planner_model_id,
    )

    result = response_message["content"]

    try:
        traj = result[result.index("["):result.index("]") + 1]
        traj = ast.literal_eval(traj)

    except (ValueError, SyntaxError):
        # Original fallback: labelled (x, y) waypoints
        lines = result.splitlines()
        waypoints = []

        for i, line in enumerate(
            line for line in lines
            if "Waypoint " in line and " : (" in line
        ):
            if f"Waypoint {i + 1} " not in line:
                raise ValueError("Unexpected waypoint numbering")

            coords = line.split(" : (", 1)[1].split(")", 1)[0]
            x, y = coords.split(",")
            waypoints.append((float(x), float(y)))

        # Additional fallback: separate x and y values
        # Additional fallback: separate x and y values
        if not waypoints:
            for i in range(1, 7):
                heading = f"**Waypoint {i} ({i * 0.5:.1f}s):**"

                if heading not in result:
                    break

                section = result.split(heading, 1)[1].split("\n\n", 1)[0]

                x = section.split("**x:**", 1)[1].strip().split()[0]
                y = section.split("**y:**", 1)[1].split("=", 1)[1].strip().split()[0]

                waypoints.append((float(x), float(y)))

        # Additional fallback: First, Second, Third... Waypoint
        if not waypoints:
            labels = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]

            for i, label in enumerate(labels, start=1):
                prefix = f"{i}. **{label} Waypoint**:"
                matches = [
                    line for line in lines
                    if line.strip().startswith(prefix)
                ]

                if len(matches) != 1 or "$(" not in matches[0]:
                    break

                coords = matches[0].split("$(", 1)[1].split(")", 1)[0]
                x, y = coords.split(",")
                waypoints.append((float(x), float(y)))

        # Qwen3 compatibility: parse Markdown-formatted waypoints.
        # Only accept the required future timestamps: 0.5s to 3.0s.
        if not waypoints and planner_model_id == "qwen3:14b":
            matches = re.findall(
                r"\*\*At\s+(\d+(?:\.\d+)?)s:\*\*"
                r".*?"
                r"\*\*Waypoint:\*\*\s*\$?\s*"
                r"\(\s*([-+]?\d*\.?\d+)\s*,\s*"
                r"([-+]?\d*\.?\d+)\s*\)",
                result,
                flags=re.DOTALL,
            )

            expected_times = np.arange(0.5, 3.1, 0.5)

            if (
                len(matches) == 6
                and np.allclose(
                    [float(t) for t, _, _ in matches],
                    expected_times,
                    atol=1e-6,
                )
            ):
                waypoints = [
                    (float(x), float(y))
                    for _, x, y in matches
                ]

        traj = waypoints

    traj = np.asarray(traj, dtype=float)

    if traj.shape != (6, 2) or not np.isfinite(traj).all():
        raise ValueError(
            f"Invalid trajectory for token {token}: shape {traj.shape}\n"
            f"Full Qwen response: {repr(result)}"
        )

    if verbose:
        print(token)
        print(f"GPT Planner:\n{result}")
        print(f"Ground Truth:\n{assitant_message}")

    output_dict = {
        "token": token,
        "Prediction": result,
        "Ground Truth": assitant_message,
    }

    if self_reflection:
        assert data_dict is not None
        collision = collision_check(traj, data_dict, safe_margin=safe_margin, token=token)
        if collision.any():
            traj = collision_optimization(traj, data_dict, occ_filter_range=occ_filter_range, sigma=sigma, alpha_collision=alpha_collision)
            if verbose:
                print("Collision detected!")
                print(f"Optimized trajectory:\n {traj}")
    return traj, output_dict

def planning_batch_inference(data_samples, planner_model_id, data_path, save_path, self_reflection=True, verbose=False):
    
    save_file_name = save_path / Path("pred_trajs_dict.pkl")
    if os.path.exists(save_file_name):
        with open(save_file_name, "rb") as f:
            pred_trajs_dict = pickle.load(f)
    else:
        pred_trajs_dict = {}
    invalid_tokens = []
    
    for data_sample in tqdm(data_samples):
        token = data_sample["token"]
        try:
            data_dict_path = Path(data_path) / Path(f"{token}.pkl")
            with open(data_dict_path, "rb") as f:
                data_dict = pickle.load(f)
            traj, output_dict = planning_single_inference(
                planner_model_id=planner_model_id, 
                data_sample=data_sample, 
                data_dict=data_dict, 
                self_reflection=self_reflection,
                safe_margin=0., 
                occ_filter_range=5.0, 
                sigma=1.265, 
                alpha_collision=7.89, 
                verbose=verbose
            )
            pred_trajs_dict[token] = traj
        except Exception as e:
            print("An error occurred:", e)
            invalid_tokens.append(token)
            print(f"Invalid token: {token}")
            continue

    print("#### Invalid Tokens ####")
    print(f"{invalid_tokens}")

    with open(save_file_name, "wb") as f:
        pickle.dump(pred_trajs_dict, f)

    return pred_trajs_dict

if __name__ == "__main__":
    current_time = time.strftime("%D:%H:%M")
    current_time = current_time.replace("/", "_")
    current_time = current_time.replace(":", "_")
    save_path = Path("experiments/outputs") / Path(current_time)

    pred_trajs_dict = planning_batch_inference(
        model_id= 'ft:gpt-3.5-turbo-0613:usc-gvl::8El3lxMY',
        save_path=save_path, 
        verbose=False,
    )