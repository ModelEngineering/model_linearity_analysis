"""
score_infos = []
Iterate on Biomodels
    trajectory  = Trajectory.makeBiomodel(model_name)
    prediction = trajectory.makeLinearPrediction()
    score_info = prediction.makeScoreInfo(trajectory.timecourse)
    score_infos.append(score_info)
"""
import src.constants as cn # type: ignore
from src.l_roadrunner import LRoadrunner # type: ignore
from src.score import Score # type: ignore
from src.trajectory import Trajectory # type: ignore
from src.biomodels_iterator import BiomodelsIterator # type: ignore

import matplotlib.pyplot as plt # type: ignore
import numpy as np  # type: ignore
import os

""" EXCLUDED_MODELS: list[str] = [
    "BIOMD0000000002",  # Model with no species
    "BIOMD0000000014",  # Long processing time
    "BIOMD0000000019",  # Long processing time
    "BIOMD0000000020",  # No floating species
    "BIOMD0000000023",  # Long processing time
    "BIOMD0000000024",  # Delay differential equation model, which is not supported by our linear predictor
    "BIOMD0000000025",  # Delay differential equation model, which is not supported by our linear predictor
    "BIOMD0000000028",  # Long processing time
    "BIOMD0000000035",  # Errors "too much work"
    "BIOMD0000000036",  # Errors "too much work"
    "BIOMD0000000054",  # Long processing time
] """
EXCLUDED_MODELS: list[str] = [
    "BIOMD0000000035",  # Errors "too much work"
    "BIOMD0000000036",  # Errors "too much work"
    "BIOMD0000000079",  # Errors "too much work"
    "BIOMD0000000088",  # Errors "too much work"
]


score = Score(serialization_path=os.path.join(cn.DATA_DIR, "linear_predictor_scores2.csv"))
if os.path.exists(score._serialization_path):
    existing_models = set(score.score_df["description"].unique())
else:
    existing_models = set()
excluded_models = list(set(EXCLUDED_MODELS) | existing_models)
iterator = BiomodelsIterator(excluded_models=excluded_models)
for idx, item in enumerate(iterator):
    model_name = item.model_name
    try:
        trajectory = Trajectory.makeBiomodel(model_name=model_name)
    except Exception as e:
        print(f"Error occurred while processing model {model_name}: {e}")
        continue
    prediction_df = trajectory.predictLinear()
    if np.any(np.isnan(prediction_df.values)):
        # Handle large Jacobians
        trajectory = Trajectory.makeBiomodel(model_name=model_name)
        prediction_df = trajectory.predictLinear(is_adjust_fitted_jacobian=True)
    score.addTestResult(trajectory.timecourse, prediction_df, description = model_name)

import pdb; pdb.set_trace()
score.plot(is_model_aggregation=True, column_name="mean", num_bin=20)