'''Script that creates a CSV file of BioModels end times.'''

import src.constants as cn  # type: ignore
from src.l_roadrunner import LRoadrunner # type: ignore

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import os
from typing import List, Dict, Tuple, Optional

EXCLUDED_MODELS = ["BIOMD0000000268",
        "BIOMD0000000625",]


def findFilesWithExtension(model_name: str, extension: str) -> List[str]:
    '''
    Find files with a specific extension in a directory.
    Args:
        model_name: The name of the BioModel directory to search in.
        extension: The file extension to look for (e.g. ".xml", ".sedml").

    Returns:
        A list of file paths with the specified extension.
    '''
    directory = os.path.join(cn.BIOMODELS_DIR, model_name)
    results = [f for f in os.listdir(directory)
            if (extension in f) and (not "manifest" in f)]
    results = [os.path.join(directory, f) for f in results]
    if len(results) == 0:
        print(f"No {extension} file found in {directory}")
    return results

def main(output_path: str = cn.CALCULATED_ENTIMES_PATH) -> pd.DataFrame:
    '''
    Main function.
    '''
    # Initialize the processed models
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path,
                usecols=[cn.COL_MODEL, cn.COL_ENDTIME_SOURCE, cn.COL_ENDTIME])
    else:
        existing_df = pd.DataFrame(
                columns=[cn.COL_MODEL, cn.COL_ENDTIME_SOURCE, cn.COL_ENDTIME])   
    processed_models = existing_df[cn.COL_MODEL].tolist()
    # Get the list of BioModel directories
    model_names = np.sort([d for d in os.listdir(cn.BIOMODELS_DIR) if "BIOMD" in d]).tolist()
    # Save the results
    result_dct: dict = {c:[] for c in [cn.COL_MODEL, cn.COL_ENDTIME_SOURCE, cn.COL_ENDTIME]}
    # Create a list to store the end times
    # Loop through each SBML directory and get the end time
    endtime_df = pd.DataFrame(result_dct)
    for model_name in model_names:
        if model_name in processed_models:
            print(f"Model {model_name} already processed, skipping.")
            continue
        if model_name in EXCLUDED_MODELS:
            print(f"Model {model_name} is excluded, skipping.")
            continue
        print("Processing model: ", model_name)
        # Find the SBML file in the directory
        sbml_files = [f for f in findFilesWithExtension(model_name, '.xml') if "_url" in f]
        if len(sbml_files) == 0:
            continue
        with open(sbml_files[0], 'r') as f:
            sbml_str = f.read()
        # Find the end time
        # Find the SEDML file in the directory
        sedml_files = findFilesWithExtension(model_name, '.sedml')
        sedml_str : Optional[str] = None
        if len(sedml_files) > 0:
            with open(sedml_files[0], 'r') as f:
                sedml_str = f.read()
        # Find the end time
        try:
            lrr = LRoadrunner(sbml_str, sedml_str=sedml_str)
        except Exception as e:
            print(f"Error occurred while processing model {model_name}: {e}")
            continue
        result_dct[cn.COL_MODEL].append(model_name)
        result_dct[cn.COL_ENDTIME].append(lrr.end_time)
        result_dct[cn.COL_ENDTIME_SOURCE].append(lrr.end_time_source)
        df = pd.DataFrame(result_dct)
        if len(df) > 0 and len(existing_df) > 0:
            endtime_df= pd.concat([existing_df, df], ignore_index=True)
        elif len(df) > 0:
            endtime_df = df
        else:
            endtime_df = existing_df
        endtime_df.to_csv(output_path, index=False)
    # Create a DataFrame from the list of end times
    return endtime_df


if __name__ == "__main__":
    main()