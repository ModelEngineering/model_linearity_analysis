"""
Module for iterating over BioModels SBML model directories.
"""
import src.constants as cn  # type: ignore

import os
import pandas as pd  # type: ignore
from typing import Iterator, List, Optional, Tuple

NUM_TEST_MODELS = 5


############################################
class BiomodelsItem:
    """Represents a single BioModel with its associated file paths."""

    def __init__(self, model_name: str, sbml_paths: List[str], sedml_paths: List[str],
            existing_df: pd.DataFrame = pd.DataFrame()) -> None:
        """
        Initialize a BiomodelsItem.

        Parameters
        ----------
        model_name : str
            The BioModel identifier (e.g. 'BIOMD0000000001').
        sbml_paths : List[str]
            Absolute paths to SBML (.xml) files in the model directory,
            excluding manifest.xml.
        sedml_paths : List[str]
            Absolute paths to SED-ML (.sedml) files in the model directory.
        existing_df : pd.DataFrame
            The DataFrame containing existing processed models.
        """
        self.model_name = model_name
        self.sbml_paths = sbml_paths
        self.sedml_paths = sedml_paths
        if existing_df is None:
            self.existing_df = pd.DataFrame()
        else:
            self.existing_df = existing_df

    def __repr__(self) -> str:
        return (
            f"BiomodelsItem(model_name={self.model_name!r}, "
            f"sbml_paths={self.sbml_paths!r}, "
            f"sedml_paths={self.sedml_paths!r}, "
            f"existing_df={self.existing_df!r}"
        )


############################################
class BiomodelsIterator:
    """Iterates over all BioModel directories in the BioModels repository."""

    def __init__(self,
                biomodels_dir: str = cn.BIOMODELS_DIR,
                excluded_models: List[str] = [],
                existing_csv_path: Optional[str] = None,
                is_report: bool = True,
                is_test: bool = False,
                ) -> None:
        """
        Initialize a BiomodelsIterator.

        Parameters
        ----------
        biomodels_dir : str
            Path to the directory containing BioModel subdirectories.
            Defaults to cn.BIOMODELS_DIR.
        excluded_models : List[str]
            List of model names to exclude from iteration.
        is_report : bool
            Whether to print progress reports.
        existing_csv_path : Optional[str]
            Path to an existing CSV file containing processed models. If provided,
            models listed in this file will be added to the excluded_models list.
            The column cn.COL_MODEL_NAME will be used to identify processed models.
        is_test : bool
            Whether to run in test mode, which may limit the number of models processed or alter behavior
        """
        self.biomodels_dir = biomodels_dir
        self.excluded_models = excluded_models
        self._is_report = is_report
        self._existing_csv_path = existing_csv_path
        self._existing_df, self._processed_models = self._getProcessedModelsFromCSV()
        self._is_test = is_test

    def _getProcessedModelsFromCSV(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Get a list of processed model names from an existing CSV file.

        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            A tuple containing the DataFrame read from the CSV file and a list of model names that have already been processed.
            If no existing CSV path is provided or the file does not exist, returns an empty DataFrame and an empty list.
        """
        if self._existing_csv_path is None or not os.path.exists(self._existing_csv_path):
            return pd.DataFrame(), []
        df = pd.read_csv(self._existing_csv_path)
        if not cn.COL_MODEL_NAME in df.columns:
            raise ValueError(f"Expected column '{cn.COL_MODEL_NAME}' not found in existing CSV file: {self._existing_csv_path}")
        processed_models = df[cn.COL_MODEL_NAME].tolist()
        return df, processed_models

    @classmethod
    def _findFilesWithExtension(cls, model_dir: str, extension: str) -> List[str]:
        """
        Find files with a given extension in a model directory, excluding manifest.xml.

        Parameters
        ----------
        model_dir : str
            Absolute path to the model directory.
        extension : str
            File extension to search for (e.g. '.xml', '.sedml').

        Returns
        -------
        List[str]
            Sorted list of absolute file paths matching the extension.
        """
        paths = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.endswith(extension) and f != "manifest.xml"
        ]
        return sorted(paths)
    
    def _msg(self, text: str) -> None:
        if self._is_report:
            print(text)

    @classmethod
    def getBiomodelInfo(cls, model_dir: str) -> BiomodelsItem:
        model_name = os.path.basename(model_dir)
        sbml_paths = cls._findFilesWithExtension(model_dir, ".xml")
        sedml_paths = cls._findFilesWithExtension(model_dir, ".sedml")
        return BiomodelsItem(
            model_name=model_name,
            sbml_paths=sbml_paths,
            sedml_paths=sedml_paths,
            existing_df=pd.DataFrame()
        )

    def __iter__(self) -> Iterator[BiomodelsItem]:
        """
        Yield a BiomodelsItem for each BioModel directory.

        Yields
        ------
        BiomodelsItem
            Item containing the model name and paths to its SBML and SED-ML files.
        """
        model_names = sorted(
            d for d in os.listdir(self.biomodels_dir)
            if os.path.isdir(os.path.join(self.biomodels_dir, d)) 
            and "BIOMD" in d
        )
        for idx, model_name in enumerate(model_names):
            if idx > NUM_TEST_MODELS and self._is_test:
                self._msg(f"Test mode enabled, stopping after {NUM_TEST_MODELS} models.")
                break
            model_dir = os.path.join(self.biomodels_dir, model_name)
            if model_name in self._processed_models:
                self._msg(f"Skipping processed model: {model_name}")
                continue
            if model_name in self.excluded_models:
                self._msg(f"Skipping excluded model: {model_name}")
                continue
            self._msg(f"Processing model: {model_name}")
            item = self.getBiomodelInfo(model_dir)
            item.existing_df = self._existing_df
            yield item