# Model-Based Design

This writeup describes a new design for modules in the project based on a better understanding of the following requirements:

  * separating static and dynamic aspects of models;
  * the need to use the same dynamics when splitting a model since we are not guaranteed to get the same numerical outputs on subsequent runs; and
  * better balancing the size of modules.
  
Refer to archive/l_roadrunner.py and archive/trajectory.py for algorithmic approaches. Use @plot_options for plotting. Do not change the files in archive; they are for reference only.

## Class Architecture

### ``StaticModel``

This class contains properties of the static model, those properties derived from the model specification (not its execution).
Included are the properties:

* ``model_name``
* ``sbml_str`` is an SBML string for the model
* Various static properties, such as ``species_names``, ``num_species``

### ``DynamicModel``

This class is a container for properties obtained running a simulation.

The constructor has arguments for: static model, and the following:

* ``start_time``, ``end_time`, ``num_point``
* ``jacobian_collection_arr`` the Jacobians at each of the timepoints
* ``timepoint_arr`` is the times at which simulation results are reported
* ``forced_input_collection_arr`` an array of arrays of forced inputs calculated at each timepoint
* ``timecourse_df`` is the timecourse for the dynamics

The following are properties calculated from the properties above:

* ``jacobian_median_arr`` is the median of the values in ``jacobian_collection_arr``, a computed property.
* ``jacobian_std_arr`` is the standard deviation of Jacobian values, a computed property.

The following are external methods:

* ``makeSubmodel(start_time, end_time)`` uses a subset of the dynamical data (as specified by start and end) to construct a new dynamical model that copies of subset of the data in the current model into the new model.
* ``makeModel(start_time, end_time, num_point, StaticModel)`` runs simulations to collect

### ``LinearModelPredictor``

This class does linear prediction and evaluations of these predictions. It is constructed with a ``DynamicModel`` and ``num_step``, the number of steps ahead to do the prediction. It has the following methods

* ``predict`` provides predictions for the timecourse of the ``DynamicModel``. 
* ``score`` scores the prediction using the Score class.
* ``plotPrediction`` plots the timecourse and the prediction from the ``start_time`` to the ``end_time``

### ``Score``

### ``MultipleLinearPredictor``

This class performs piece-wise linear prediction. It is constructed wtih a DynamicModel, ``num_step``, and a collection of timepoints a which a new dynamic model is constructed.

* ``predict``
* ``score``
* ``plotPrediction`` The plot shows predicted and actual (simulated) values with vertical dashed lines to indicate regions for submodels.

## To Do

### Prompts


1. Review the design and provide comments on where confusions exist as well as where improvements can be made. Do not implement yet.
1. Implement StaticModel as described in @model_based_design.md and associated tests.
2. Implement DynamicModel as described in @model_based_design.md and associated tests. 
3.  
4. Update the modules in scripts to use ...