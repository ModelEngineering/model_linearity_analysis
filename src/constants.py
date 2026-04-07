'''Constants used in the project'''

import os
import tellurium as te  # type: ignore

# Directories and paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
REPO_DIR = os.path.dirname(PROJECT_DIR)
BIOMODELS_DIR = os.path.join(REPO_DIR, "temp-biomodels", "final")
CALCULATED_ENTIMES_PATH = os.path.join(DATA_DIR, "biomodels_endtime.csv")

# Types
TYPE_ROADRUNNER = "tellurium.roadrunner.extended_roadrunner.ExtendedRoadRunner"
NULL_ROADRUNNER = te.loada("")

# Constants
START_TIME = 0.0
END_TIME = 10.0
NUM_POINTS = 10*int(END_TIME - START_TIME) + 1

# Columns
COL_MAXCV = "max_cv"
COL_ENDTIME = "end_time"
COL_MODEL_NAME = "model_name"
COL_ENDTIME_SOURCE = "end_time_source"  # How end_time was determined (e.g. "reciprocal_min_eigenvalue", "default")
ENDTIME_SOURCE_RECIROCAL_MIN_EIGENVALUE = "reciprocal_min_eigenvalue"
ENDTIME_SOURCE_SEDML = "sedml"
ENDTIME_SOURCE_STEADYSTATE = "steadystate"
ENDTIME_SOURCE_MAX_MEDIAN_CV = "max_median_cv"
ENDTIME_SOURCE_USER_SPECIFIED = "user_specified"
COL_NAMES = [COL_MODEL_NAME, COL_MAXCV, COL_ENDTIME, COL_ENDTIME_SOURCE]