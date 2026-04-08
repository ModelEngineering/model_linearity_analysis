'''Multiple linear predictor for chemical reaction network trajectories.'''

from src.clustered_jacobian_collection import ClusteredJacobianCollection  # type: ignore
from src.l_roadrunner import LRoadrunner  # type: ignore
from src.linear_predictor import LinearPredictor  # type: ignore

import matplotlib.axes as maxes  # type: ignore
import matplotlib.figure as mfigure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from typing import List, Optional


class MultipleLinearPredictor(object):
    """Predicts species concentrations across a sequence of linear models.

    Uses a ClusteredJacobianCollection to build one LinearPredictor per cluster,
    propagating the predicted state from the end of each cluster as the initial
    condition for the next.  Forced inputs for each cluster are computed from
    the LRoadrunner instance via: u = f(x0) - J_mean @ x0, where f(x0) is the
    instantaneous rate of change of floating species at the current state x0.
    """

    def __init__(self,
            clustered_jacobian_collection: ClusteredJacobianCollection,
            l_roadrunner: Optional[LRoadrunner] = None,
            ) -> None:
        """
        Parameters
        ----------
        clustered_jacobian_collection : ClusteredJacobianCollection
            Clustered Jacobian collections defining the sequence of linear models.
        l_roadrunner : LRoadrunner, optional
            LRoadrunner instance used to obtain initial concentrations and to
            compute instantaneous rates of change for forced-input estimation.
            If not provided, falls back to the l_roadrunner from the collection.
        """
        self.clustered_jacobian_collection = clustered_jacobian_collection
        self.l_roadrunner = l_roadrunner if l_roadrunner is not None else self.clustered_jacobian_collection.l_roadrunner
        if self.l_roadrunner is not None:
            self.species_names = self.l_roadrunner.getRoadrunner().getFloatingSpeciesIds()
        else:
            self.species_names = []

    def predict(self) -> np.ndarray:
        """Predict floating species concentrations at the end of each cluster.

        For each JacobianCollection in the ClusteredJacobianCollection:
        1. Compute the forced input u = f(x0) - J_mean @ x0, where f(x0) is
            the instantaneous rate of change at the current state x0.
        2. Build a LinearPredictor with the current state as initial condition.
        3. Predict concentrations at the last timepoint of the cluster.
        4. Use the predicted concentrations as the initial state for the next cluster.

        Returns
        -------
        np.ndarray
            Array of shape (n_clusters, n_species) containing the predicted
            floating species concentrations at the last timepoint of each cluster.
        """
        rr = self.l_roadrunner.getRoadrunner()
        rr.reset()
        current_x = np.array(rr.getFloatingSpeciesConcentrations())

        predictions: List[np.ndarray] = []
        for jc in self.clustered_jacobian_collection.jacobian_collections:

            # Compute forced input: u = f(x0) - J_mean @ x0
            # Set floating species to current_x and read instantaneous rates.
            rr.reset()
            species_ids = rr.getFloatingSpeciesIds()
            for idx, sp_id in enumerate(species_ids):
                rr[sp_id] = float(current_x[idx])
            f_arr = np.array(rr.getRatesOfChange())
            forced_input_arr = f_arr - jc.jacobian_mean_arr @ current_x

            # Predict at the end of this cluster (duration from cluster start)
            duration = float(jc.timepoint_arr[-1] - jc.timepoint_arr[0])
            linear_predictor = LinearPredictor(jc, current_x, forced_input_arr)
            predicted_arr = linear_predictor.predict(np.array([0.0, duration])).prediction_arr
            current_x = predicted_arr[-1]
            predictions.append(current_x)

        return np.array(predictions)  # shape: (n_clusters, n_species)

    def plot(self,
            ax: Optional[maxes.Axes] = None,
            title: str = "Multiple Linear Prediction vs Simulation",
            ylim: Optional[tuple] = None,
            xlim: Optional[tuple] = None,
            ) -> mfigure.Figure:
        """Plot predicted and simulated species concentrations over time.

        Simulation is drawn as solid lines; the piecewise linear prediction as
        dashed lines.  One colour per floating species is used so that prediction
        and simulation for the same species share a colour.  A vertical dotted
        line marks the start of each cluster after the first, indicating where a
        new Jacobian takes over.

        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to draw on.  A new figure and axes are created when omitted.
        title : str, optional
            Title for the axes.
        ylim : tuple[float, float], optional
            y-axis limits as (ymin, ymax).
        xlim : tuple[float, float], optional
            x-axis limits as (xmin, xmax).

        Returns
        -------
        plt.Figure
            The figure containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # --- simulated trajectory ---
        sim_arr = self.l_roadrunner.simulate(is_with_timepoints=True)
        sim_times = sim_arr[:, 0]
        sim_conc_arr = sim_arr[:, 1:]
        n_species = sim_conc_arr.shape[1]

        for i in range(n_species):
            label = self.species_names[i] if i < len(self.species_names) else f"species_{i}"
            ax.plot(sim_times, sim_conc_arr[:, i], color=f"C{i}",
                    label=f"{label} (simulation)")

        # --- piecewise linear prediction ---
        rr = self.l_roadrunner.getRoadrunner()
        rr.reset()
        current_x = np.array(rr.getFloatingSpeciesConcentrations())

        for cluster_idx, jc in enumerate(self.clustered_jacobian_collection.jacobian_collections):
            # Draw vertical boundary line at the start of each new cluster.
            if cluster_idx > 0:
                ax.axvline(x=float(jc.timepoint_arr[0]), color="gray",
                linestyle=":", linewidth=1.0)

            # Forced input at current state.
            rr.reset()
            species_ids = rr.getFloatingSpeciesIds()
            for idx, sp_id in enumerate(species_ids):
                rr[sp_id] = float(current_x[idx])
            f_arr = np.array(rr.getRatesOfChange())
            forced_input_arr = f_arr - jc.jacobian_mean_arr @ current_x

            # Predict at every timepoint in the cluster (relative times).
            abs_times = jc.timepoint_arr
            rel_times = abs_times - abs_times[0]
            linear_predictor = LinearPredictor(jc, current_x, forced_input_arr)
            predicted_arr = linear_predictor.predict(rel_times).prediction_arr

            for i in range(n_species):
                label = (self.species_names[i] if i < len(self.species_names)
                         else f"species_{i}")
                if cluster_idx == 0:
                    ax.plot(abs_times, predicted_arr[:, i], color=f"C{i}",
                            linestyle="--", label=f"{label} (prediction)")
                else:
                    ax.plot(abs_times, predicted_arr[:, i], color=f"C{i}",
                            linestyle="--")

            current_x = predicted_arr[-1]

        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        ax.set_title(title)
        ax.legend()
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
        return fig  # type: ignore