
import pickle
from pathlib import Path

from agentdriver.memory.memory_agent import MemoryAgent
from agentdriver.functional_tools.functional_agent import FuncAgent

token = "00014584c6de4789a69b4717fa3c0d22"
data_path = Path("data")
sample_path = data_path / "train" / f"{token}.pkl"

# Load existing driving scenario
with open(sample_path, "rb") as f:
    data_dict = pickle.load(f)

# Retrieve ego-state information
func_agent = FuncAgent(data_dict)
ego_prompts, ego_data = func_agent.get_ego_states()

# Construct the minimum working memory required for retrieval
working_memory = {
    "token": token,
    "ego_data": ego_data,
}

print("Initialising memory agent...", flush=True)

memory_agent = MemoryAgent(
    data_path=data_path,
    model_name="qwen3:14b",
    verbose=True,
)

print("Testing memory retrieval...", flush=True)

common_sense_prompts, experience_prompt = memory_agent.run(
    working_memory
)

print("\nCommon-sense memory:")
print(common_sense_prompts)

print("\nExperience memory:")
print(experience_prompt)

print("\nMemory retrieval test completed.", flush=True)
