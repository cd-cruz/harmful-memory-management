import subprocess
import sys

# Run both baseline experiments sequentially
if __name__ == "__main__":
    print("Starting memory-enabled baseline...", flush=True)

    subprocess.run(
        [sys.executable, "main_strict.py"],
        check=True
    )

    print("Starting memory-free baseline...", flush=True)

    subprocess.run(
        [sys.executable, "main_no_experience.py"],
        check=True
    )

    print("Both baseline experiments completed successfully!")