
import pickle
from pathlib import Path

from agentdriver.perception.perception_agent import PerceptionAgent
from agentdriver.reasoning.reasoning_agent import ReasoningAgent

# Use the same nuScenes sample from the perception test
token = "00014584c6de4789a69b4717fa3c0d22"

data_path = Path("data")
sample_path = data_path / "train" / f"{token}.pkl"

# Load the original processed nuScenes sample
with open(sample_path, "rb") as f:
    data_dict = pickle.load(f)

# Step 1: Run original AgentDriver perception
print("Running perception with Qwen3:14B...", flush=True)

perception_agent = PerceptionAgent(
    token=token,
    split="train",
    data_path=data_path,
    model_name="qwen3:14b",
    verbose=False,
)

ego_prompts, perception_prompts, working_memory = perception_agent.run()

print("Perception completed.", flush=True)

# Step 2: Pass perception results into original reasoning
env_info_prompts = ego_prompts + "\n" + perception_prompts

reasoning_agent = ReasoningAgent(
    model_name="qwen3:14b",
    verbose=True,
)

print("Running reasoning with Qwen3:14B...", flush=True)

reasoning = reasoning_agent.run(
    data_dict=data_dict,
    env_info_prompts=env_info_prompts,
    working_memory=working_memory,
    use_cot_rules=False,
)

print("\nReasoning test completed.", flush=True)
print("Reasoning result:\n", reasoning)
