import subprocess
import sys

# Run multiple baseline sequentially
if __name__ == "__main__":
    print("Starting experiment X")
    
    subprocess.run([sys.executable, "main_strict.py"],
                   check = True
                   )
    
    print("Stating experiment Y")
    
    subprocess.run([sys.executable, "main_no_experience.py"],
                    check = True
                    )
    
    print("\nExperiment Completed")