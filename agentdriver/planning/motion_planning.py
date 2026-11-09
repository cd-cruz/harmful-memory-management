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


def parse_qwen_trajectory(result):
    """Extract six future waypoints without changing their coordinates."""
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    pair = rf"\(\s*({number})\s*,\s*({number})\s*\)"
    expected_times = np.arange(0.5, 3.1, 0.5)

    def validate(coords, times=None):
        traj = np.asarray(coords, dtype=float)

        if traj.shape != (6, 2) or not np.isfinite(traj).all():
            raise ValueError("Invalid Qwen trajectory coordinates")

        if times is not None:
            if len(times) != 6 or not np.allclose(
                times, expected_times, atol=1e-6, rtol=0
            ):
                raise ValueError("Incorrect waypoint timestamps")

        return traj

    # Format A: Python list of six coordinate pairs
    for match in re.finditer(r"\[[^\[\]]*\]", result, re.DOTALL):
        try:
            coords = ast.literal_eval(match.group())
            return validate(coords)
        except (ValueError, SyntaxError, TypeError):
            continue

    # Format B: Explicit timestamp and coordinate pair
    timestamp_pair = re.findall(
        rf"(?m)^\s*(?:-\s*)?\*\*(?:At\s+)?"
        rf"({number})\s*(?:seconds|s)\s*:?\*\*:"
        rf"\s*\n\s*\$?\s*(?:\(x,\s*y\)\s*=\s*)?"
        rf"{pair}",
        result,
    )

    if timestamp_pair:
        return validate(
            [(float(x), float(y)) for _, x, y in timestamp_pair],
            [float(t) for t, _, _ in timestamp_pair],
        )

    # Format C: Numbered Markdown waypoint list
    numbered = re.findall(
        rf"(?m)^\s*([1-6])\.\s*\*\*{pair}\*\*",
        result,
    )

    if numbered:
        indices = [int(i) for i, _, _ in numbered]

        if indices != list(range(1, 7)):
            raise ValueError("Incorrect waypoint numbering")

        # Numbered lists require explicit timing context.
        has_full_times = all(
            re.search(
                rf"(?<![\d.]){t:.1f}\s*s(?:econds)?\b",
                result,
            )
            for t in expected_times
        )

        has_interval = bool(
            re.search(r"0\.5\s*s(?:econds)?\s+intervals?", result)
            and re.search(
                r"starting\s+at\s+0\.5\s*s", result
            )
        )

        if not (has_full_times or has_interval):
            raise ValueError("Ambiguous numbered waypoint timing")

        return validate(
            [(float(x), float(y)) for _, x, y in numbered]
        )

    # Format D: Timestamped separate x/y approximate values
    separate_xy = re.findall(
        rf"(?m)^\s*-\s*\*\*(?:At\s+)?"
        rf"({number})\s*(?:seconds|s)\*\*:"
        rf"[^\n]*\n[^\n]*?"
        rf"x\s*\\approx\s*({number})"
        rf"[^\n]*?"
        rf"y\s*\\approx\s*({number})",
        result,
    )

    if separate_xy:
        return validate(
            [(float(x), float(y)) for _, x, y in separate_xy],
            [float(t) for t, _, _ in separate_xy],
        )

    # Previous Qwen format: At 0.5s / Waypoint
    previous_format = re.findall(
        rf"\*\*At\s+({number})s:\*\*"
        rf".*?\*\*Waypoint:\*\*\s*\$?\s*{pair}",
        result,
        flags=re.DOTALL,
    )

    if previous_format:
        return validate(
            [(float(x), float(y)) for _, x, y in previous_format],
            [float(t) for t, _, _ in previous_format],
        )

    raise ValueError("No recognised six-waypoint trajectory")


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

    if planner_model_id == "qwen3:14b":
        try:
         traj = parse_qwen_trajectory(result)
        except ValueError:
            print(f"\nInvalid Qwen trajectory for token: {token}")
            print(f"Full Qwen response:\n{result}")
            raise

    else:
        # Original AgentDriver trajectory parser.
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
                labels = [
                    "First", "Second", "Third",
                    "Fourth", "Fifth", "Sixth"
                ]

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

            traj = waypoints

    # Convert the trajectory to a NumPy array.
    traj = np.asarray(traj, dtype=float)

    # Validate trajectory shape and coordinates.
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