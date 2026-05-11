# Model-Based Design

This writeup describes a new design for modules in the project based on a better understanding of the following requirements:

* separating static and dynamic aspects of models;
* the need to use the same dynamics when splitting a model since we are not guaranteed to get the same numerical outputs on subsequent runs; and
* better balancing the size of modules.

Refer to archive/l_roadrunner.py and archive/trajectory.py for algorithmic approaches. Use @plot_options for plotting. Do not change the files in archive; they are for reference only.

## Class Architecture

### ``Model``

This class contains properties of the static model, those properties derived from the model specification (not its execution).
Included are the properties:

* ``model_name``
* ``sbml_str`` is an SBML string for the model. Antimony models are converted to SBML using roadrunner.
* Various static properties, such as ``species_names``, ``num_species``
* ``__eq__`` which checks the ``model_name`` and ``sbml_str``

StaticModel uses roadrunner transiently to query the underlying model (e.g., names of floating species).

### ``Trajectory``

This class is a container for properties obtained running a simulation.

The constructor has arguments for: static model, and the following:

* ``jacobian_collection_arr`` the Jacobians at each of the timepoints
* ``timepoint_arr`` is the times at which simulation results are reported. The first timepoint is used to obtain initial values for the Trajectory.
* ``forcing_input_collection_arr`` an array of arrays of forced inputs calculated at each timepoint
* ``timecourse_df`` is the timecourse for the dynamics

The following are properties calculated from the properties above:

* ``jacobian_median_arr`` is the median of the values in ``jacobian_collection_arr``, a computed property.
* ``jacobian_std_arr`` is the standard deviation of Jacobian values, a computed property.
* ``start_time``, ``end_time``, ``num_point``

The following are external methods:

* ``makeSubmodel(start_time, end_time)`` uses a subset of the dynamical data (as specified by start and end) to construct a new dynamical model that copies of subset of the data in the current model into the new model. This method uses slicing to obtain values for constructing a new DynamicModel.
* ``makeFromSimulation(start_time, end_time, num_point, StaticModel)`` is a class method that runs simulations to obtain the arguments for the DynamicModel constructor. This is the only method in DynamicModel that uses roadrunner. If the model_name begins with "BIOMD", then: (a) the model is in BIOMODELS_DIR and (b) ``end_time`` is obtained from the module ``biomodels_iterator``. ``end_time=None`` triggers autodetect of ``end_time``.
* ``_makeEndtime`` is a method formerly in LRoadrunner.
* ``__eq__`` which checks that the constructor arguments are the same

### ``LinearPredictor``

This class does linear prediction and evaluations of these predictions. It is constructed with a ``DynamicModel``, ``jacobian_selection`` and ``num_step``, the number of steps ahead to do the prediction. It has the following methods

* ``predict`` provides predictions for the timecourse of the ``DynamicModel``.
* ``score`` scores the prediction using the Score class.
* ``plotPrediction`` plots the timecourse and the prediction from the ``start_time`` to the ``end_time``

### ``Score``

* We will use ``makeScoreInfo`` to construct the various score metrics to evaluate the quality of a prediction.

### ``TrajectoryCollection``

This class represents a collection of ``Trajectory`` with the same
``Model``. Methods include:

* ``plotTimecourse`` pieces together the timecourse for each ``DynamicModel``, and plots it with a vertical dashed line separating each ``DynamicModel``.
* ``__eq__`` which checks if it's the same as another ``DynamicModelCollection`` by comparing each ``DynamicModel``.
* ``__lt`` if the last timepoint of the first Trajector equals the first timepoint of the second trajectory.
* ``split`` takes as input timepoints to create multiple Trajectory objects. A timepoint specifies the last time for the preceeding Trajectory and the first time for the next Trajectory.

### ``MultipleLinearPredictor``

This class performs piece-wise linear prediction. The times at which
there is a partition of the linear model is a "split point". Using slicing, we can easily construct the DynamicModels for a collection of split points. (Of course, all will have the same StaticModel.) If there are n split points, then there are n + 1 linear models.
Note that the old Trajectory.sequentialPartition / nonsequentialPartition aren't mentioned beause their implementation is deferred.

* When splitting a ``DynamicModel``, slicing is used, not simulation.
* ``split`` can be called with specific split times or without any split time specified. A split time t1 specifies the time at which the previous DynamicModel ends and the second timepoint of the new Dynamics model.
**Does this belong here?**
* ``predict``
* ``score``
* ``plotPrediction`` The plot shows predicted and actual (simulated) values with vertical dashed lines to indicate regions for submodels.

### Prompts

1. Review the design and provide comments on where confusions exist as well as where improvements can be made. Do not implement yet.
1. Implement StaticModel as described in @model_based_design.md and associated tests.
1. Implement DynamicModel as described in @model_based_design.md and associated tests.
1. Update the modules in scripts to use.
