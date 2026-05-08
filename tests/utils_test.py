"""Commonly used code in tests."""

import src.constants as cn

import os


def makeBiomdName(model_num: int) -> str:
    """Construct the model name"""
    return f"{model_num:04d}"

def makeBiomdPath(model_num: int) -> str:
    """Construct the file path for a BioModel SBML file given its model number."""
    model_str = makeBiomdName(model_num)
    return os.path.join(
        cn.BIOMODELS_DIR, f"BIOMD000000{model_str}", f"BIOMD000000{model_str}_url.xml"
    )