# Tasks

1. Trajectory can be constructed with a LRoadrunner, in which case it has values for
   end_time, species_names, num_species (Maybe a class called Model?) and obtains from
   lRoadrunner: timecourse_df, Jacobians, forced_inputs, initial_values
1. Consider separating Trajectory into Trajectory (a container) and TrajectoryAnalyzer (predictions, plots)
2. Compare predictions with and without splits.
3. Calculation of weighted eigenvectors is too slow for repeated use. Precalculate all jacobian eigenvalues?
4. Implement end_time calculation where try to maximize the median ratio of the cv of the timecourses.
5. JacobianCluster has visualization for clusters that show max mean-normalized distance from cluster mean for each timepoint
