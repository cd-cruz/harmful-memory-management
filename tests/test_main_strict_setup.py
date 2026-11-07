
import json
import pickle
from pathlib import Path

from agentdriver.llm_core.api_keys import (
    FINETUNE_PLANNER_NAME,
    OLLAMA_BASE_URL,
)
from agentdriver.memory.memory_agent import MemoryAgent


def main():
    data_path = Path("./data")

    print("=== ALC Experiment Setup Test ===")

    # 1. Check model configuration
    print(f"\nModel: {FINETUNE_PLANNER_NAME}")
    print(f"Ollama endpoint: {OLLAMA_BASE_URL}")

    assert FINETUNE_PLANNER_NAME == "qwen3:14b"
    print("PASS: Qwen model configured")

    # 2. Check dataset split
    split_file = data_path / "split.json"
    assert split_file.is_file(), f"Missing: {split_file}"

    with open(split_file, "r") as f:
        split_data = json.load(f)

    val_tokens = split_data["val"]
    print(f"Validation scenarios: {len(val_tokens)}")

    assert len(val_tokens) >= 2
    print("PASS: Validation split available")

    # 3. Check ground-truth trajectories
    gt_file = data_path / "metrics" / "gt_traj.pkl"
    assert gt_file.is_file(), f"Missing: {gt_file}"

    with open(gt_file, "rb") as f:
        gt_data = pickle.load(f)

    print(f"Ground-truth entries: {len(gt_data)}")

    missing = [token for token in val_tokens[:2] if token not in gt_data]
    assert not missing, f"Missing ground truth for: {missing}"
    print("PASS: Ground-truth trajectories available")

    # 4. Check memory initialization
    print("\nInitializing MemoryAgent...")

    memory_agent = MemoryAgent(
        data_path=data_path,
        model_name=FINETUNE_PLANNER_NAME,
        verbose=False,
    )

    experience_memory = memory_agent.experience_memory

    assert experience_memory is not None, (
        "ExperienceMemory was not initialized"
    )

    memory_count = experience_memory.get_n_remaining_memory()
    print(f"Initial memory count: {memory_count}")

    assert memory_count == 180, (
        f"Expected 180 initial memories, found {memory_count}"
    )

    print("PASS: Initial experience memory contains 180 entries")

    print("\n=== ALL SETUP CHECKS PASSED ===")


if __name__ == "__main__":
    main()
