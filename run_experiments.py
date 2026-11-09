import os


# Run both baseline experiments
if __name__ == "__main__":
    os.system("python main_strict.py")
    os.system("python main_no_experience.py")