'''Script that serializes Timecourse (timecourse_df + Jacobians) for all BioModels.'''

import src.constants as cn  # type: ignore
from src.biomodels_iterator import BiomodelsIterator  # type: ignore
from src.model import Model  # type: ignore
from src.timecourse import Timecourse  # type: ignore

import os
from typing import List

EXCLUDED_MODELS: List[str] = [
    "BIOMD0000000268",
    "BIOMD0000000625",
]


def main(
        is_report: bool = True,
        first_model_num: int = 0,
        last_model_num: int = int(1e9),
        excluded_models: List[str] = EXCLUDED_MODELS,
        is_initialize: bool = True, # Ignore existing serialized Timecourse when initializing (for testing).
) -> None:
    '''Serialize timecourses for all BioModels.

    Parameters
    ----------
    is_report : bool
        Whether to print progress.
    first_model_num : int
        First model number to include (inclusive).
    last_model_num : int
        Last model number to include (inclusive).
    excluded_models : List[str]
        Model names to skip.
    '''
    os.makedirs(cn.TIMECOURSE_SERIALIZATION_DIR, exist_ok=True)
    for item in BiomodelsIterator(
            excluded_models=excluded_models,
            is_report=is_report,
            first_model_num=first_model_num,
            last_model_num=last_model_num):
        model_name = item.model_name
        if not item.sbml_paths:
            print(f"Skipping {model_name} (no SBML files)")
            continue
        pkl_path = os.path.join(cn.TIMECOURSE_SERIALIZATION_DIR,
                f"{model_name}_timecourse.pkl")
        if os.path.isfile(pkl_path) and (not is_initialize):
            print(f"Skipping {model_name} (already serialized)")
            continue
        try:
            model = Model.makeBiomodel(model_name)
            timecourse = Timecourse(model=model, end_time=item.end_time)
            _ = timecourse.jacobian_collection_arr  # Force calculations
            path = timecourse.serialize()
            serialized_timecourse = Timecourse.deserialize(path=path)
            if not serialized_timecourse == timecourse:
                raise ValueError(f"Deserialized timecourse does not match original for {model_name}.")  
        except Exception as e:
            print(f"Error processing {model_name}: {e}")


if __name__ == "__main__":
    main()
