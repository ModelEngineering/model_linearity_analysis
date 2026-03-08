'''Constants used in the project'''

import os

# Directories and paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
REPO_DIR = os.path.dirname(PROJECT_DIR)
BIOMODELS_DIR = os.path.join(REPO_DIR, "temp-biomodels", "final")