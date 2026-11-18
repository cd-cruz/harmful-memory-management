import subprocess
import sys

if __name__ == "__main__":

    experiments = [
        ("Balanced Classifier", "main_harmful_memory_balance.py"),
        ("Conservative Classifier", "main_harmful_memory_conservative.py"),
        ("Sensitive Classifier", "main_harmful_memory_sensitive.py"),
    ]

    for experiment_name, filename in experiments:
        print(f"\nStarting {experiment_name}", flush=True)

        result = subprocess.run(
            [sys.executable, filename],
            check=False
        )

        if result.returncode == 0:
            print(f"Completed {experiment_name}", flush=True)
        else:
            print(
                f"WARNING: {experiment_name} finished with errors "
                f"(exit code {result.returncode}).",
                flush=True
            )

        print("Moving to next experiment...\n", flush=True)

    print("\nAll experiments attempted!")