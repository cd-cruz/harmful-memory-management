# Agent Memory Management and Harmful Memory Management

This repository contains the integrated source code based on the following research papers: 
- **How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior**
- **A Language Agent for Autonomous Driving**.

This project extends the original Agent-Driver memory management system to detect and mitigate harmful memory use in autonomous driving agents. It uses the Qwen3:14B LLM instead of GPT-3.5 Turbo and can run locally.

Original GitHub Repositories:
Agent-Driver: https://github.com/physical-superintelligence-lab/Agent-Driver
Agent-Driver Memory Management: https://github.com/yuplin2333/agent_memory_manage/tree/main

The dataset is available through the original Agent-Driver GItHub repository. It uses pre-cached data from the nuScenes dataset and is stored on Google Drive.

The Agent-Driver and the memory management are already integrated into this repository.

Steps:
1. Clone this repository

2. pip install -r requirements.txt

3. Install the datasets and move the files, "finetune", "memory", "metrics", "train", and "val" into the "data" folder

4. Install Ollama: https://ollama.com/download

5. Install Qwen3:14B: https://ollama.com/library/qwen3:14b

6. In Powershell, run the command: "ollama run qwen3:14b"

7. To run this locally, you need to go to "/agentdriver/llm_core" and duplicate "api_keys.py.example"
Rename the duplicate to "api_keys.py"

8. In the file "api_keys.py" modify the following:
"""
OPENAI_API_KEY = "ollama"
OPENAI_ORG = ""

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen3:14b"

FINETUNE_PLANNER_NAME = OLLAMA_MODEL
"""

The default local API is "http://localhost:11434/v1"

9. To run the baseline experiments:
Memory enabled: python main_strict.py
Memory disabled: python main_no_experience.py

10. To run the harmful memory detection and mitigation experiments:
Sensitive Classifier: python main_harmful_memory_sensitive.py
Balanced Classifier: python main_harmful_memory_balance.py
Conservative Classifier: python main_harmful_memory_conservative.py

Results
Results are saved in the "results" directory