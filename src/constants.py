'''Constants used in the project'''

import os

# Directories and paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
REPO_DIR = os.path.dirname(PROJECT_DIR)
BIOMODELS_DIR = os.path.join(REPO_DIR, "temp-biomodels", "final")

# Constants
START_TIME = 0.0
END_TIME = 10.0
NUM_POINTS = 10*int(END_TIME - START_TIME) + 1