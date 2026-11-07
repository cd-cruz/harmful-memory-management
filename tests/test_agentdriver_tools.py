
from agentdriver.llm_core.chat import (
    run_one_round_conversation_with_functional_call
)

# Define a simple test function
test_functions = [
    {
        "name": "get_vehicle_speed",
        "description": "Get the current vehicle speed.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    }
]

messages = []

print("Testing Qwen function calling...", flush=True)

full_messages, response = (
    run_one_round_conversation_with_functional_call(
        full_messages=messages,
        system_message="You are a driving assistant. Use the available function to answer.",
        user_message="What is the current vehicle speed? Call the function.",
        functional_calls_info=test_functions,
        model_name="qwen3:14b",
    )
)

print("Response:", response, flush=True)

if response.get("tool_calls"):
    print("SUCCESS: Qwen returned a tool call.")
else:
    print("No tool call returned.")
