
from pathlib import Path

from agentdriver.main.language_agent import LanguageAgent

token = "00014584c6de4789a69b4717fa3c0d22"

print("Initialising full AgentDriver with Qwen3:14B...", flush=True)

agent = LanguageAgent(
    data_path=Path("data"),
    split="train",
    model_name="qwen3:14b",
    planner_model_name="qwen3:14b",
    finetune_cot=False,
    verbose=True,
)

print("Running full AgentDriver pipeline...", flush=True)

planning_traj = agent.inference_single(
    token=token,
    data_sample=None,
)

print("\nFull AgentDriver test completed.", flush=True)
print("Planning result:", planning_traj)
