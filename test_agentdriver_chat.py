
from agentdriver.llm_core.chat import run_one_round_conversation

messages = []

print("Testing AgentDriver with Qwen3:14B...", flush=True)

full_messages, response = run_one_round_conversation(
    full_messages=messages,
    system_message="You are a helpful assistant.",
    user_message="Reply with exactly: AgentDriver Qwen test successful",
    model_name="qwen3:14b",
)

print("Response:", response.get("content"))
print("Test completed.", flush=True)
