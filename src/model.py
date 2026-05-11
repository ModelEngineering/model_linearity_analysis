"""Represents the static (specification-derived) properties of a model."""

import src.constants as cn  # type: ignore

import os
import tellurium as te  # type: ignore
from typing import List


class Model(object):
    """Static properties of a model derived from its specification, not its execution.

    Accepts an SBML or Antimony string. Antimony is converted to SBML on
    construction. RoadRunner is used transiently and not stored.
    """

    def __init__(self, model_str: str, model_name: str = "") -> None:
        """
        Parameters
        ----------
        model_str : str
            SBML XML string or Antimony model string.
        model_name : str
            Optional identifier for the model (e.g. 'BIOMD0000000001').
        """
        if not isinstance(model_str, str):
            raise ValueError("model_str must be a string.")
        self.model_name = model_name
        self.sbml_str = self._toSBML(model_str)
        self._species_names: List[str] = []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Model):
            return NotImplemented
        return self.sbml_str == other.sbml_str and self.model_name == other.model_name

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def species_names(self) -> List[str]:
        """Floating species names, cached after the first query."""
        if not self._species_names:
            rr = te.loadSBMLModel(self.sbml_str)
            ids = rr.getFloatingSpeciesIds()
            self._species_names = [
                s[1:-1] if s.startswith("[") and s.endswith("]") else s
                for s in ids
            ]
        return self._species_names

    @property
    def num_species(self) -> int:
        """Number of floating species in the model."""
        return len(self.species_names)

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def makeBiomodel(cls, model_name: str) -> "Model":
        """
        Create a Model from a BioModels SBML file in cn.BIOMODELS_DIR.

        Parameters
        ----------
        model_name : str
            BioModel identifier (e.g. 'BIOMD0000000001'). Must start with 'BIOMD'.

        Returns
        -------
        Model
        """
        if not model_name.startswith("BIOMD"):
            raise ValueError(f"model_name must start with 'BIOMD', got '{model_name}'.")
        model_dir = os.path.join(cn.BIOMODELS_DIR, model_name)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        xml_files = [
            f for f in os.listdir(model_dir)
            if f.endswith(".xml") and f != "manifest.xml"
        ]
        if not xml_files:
            raise FileNotFoundError(f"No SBML file found in {model_dir}")
        path = os.path.join(model_dir, sorted(xml_files)[0])
        with open(path) as fh:
            sbml_str = fh.read()
        return cls(sbml_str, model_name=model_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _toSBML(self, model_str: str) -> str:
        """Return an SBML string, converting from Antimony if necessary."""
        stripped = model_str.strip()
        if "<?xml" in stripped or "<sbml" in stripped:
            te.loadSBMLModel(model_str)  # validate
            return model_str
        try:
            rr = te.loada(model_str)
            return rr.getSBML()
        except Exception as e:
            raise ValueError(f"Could not load model: {e}")
