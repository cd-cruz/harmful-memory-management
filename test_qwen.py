
import openai

# Connect to Ollama instead of OpenAI's cloud API
openai.api_base = "http://localhost:11434/v1"
openai.api_key = "ollama"

response = openai.ChatCompletion.create(
    model="qwen3:14b",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: Qwen connection successful"
        }
    ],
    temperature=0,
)

print(response["choices"][0]["message"]["content"])
