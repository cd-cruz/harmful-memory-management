
from pathlib import Path

from agentdriver.perception.perception_agent import PerceptionAgent

token = "00014584c6de4789a69b4717fa3c0d22"

agent = PerceptionAgent(
    token=token,
    split="train",
    data_path=Path("data"),
    model_name="qwen3:14b",
    verbose=True,
)

print("Testing PerceptionAgent with Qwen3:14B...", flush=True)

ego_prompts, perception_prompts, working_memory = agent.run()

print("Perception test completed.", flush=True)
print("Perception results:", perception_prompts)
print("Working memory keys:", list(working_memory.keys()))
