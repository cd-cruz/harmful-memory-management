
import pickle
from pathlib import Path

from agentdriver.functional_tools.functional_agent import FuncAgent

data_dir = Path("data/train")

samples = sorted(data_dir.glob("*.pkl"))

if not samples:
    raise FileNotFoundError(
        f"No .pkl samples found directly inside {data_dir}"
    )

sample = samples[0]

print("Loading sample:", sample, flush=True)

with open(sample, "rb") as f:
    data_dict = pickle.load(f)

agent = FuncAgent(data_dict)

print("Testing ego states...", flush=True)
ego_prompts, ego_data = agent.get_ego_states()
print("Ego states: SUCCESS")

print("Testing object detection...", flush=True)
prompt, result = agent.get_leading_object_detection()

print("Detection result:", prompt)
print("Functional agent test completed.", flush=True)
