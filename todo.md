# Tasks

1. Test processBioModels
1. Finding an end time reflective of the system dynamics is challenging. Only about 15% of models have a steady state. Need another criteria. Check for perodicities and use the longest period.
   1. See if there is a human crafted end time in the SEDML. Doesn't end at 10. Doesn't have:    ``<uniformTimeCourse id="auto_ten_seconds" initialTime="0" outputStartTime="0" outputEndTime="10" numberOfSteps="1000">``
   2. Try steady state from initial conditions in simulation.
   3. If (2) fails, run longer simulation, then try steady state.
   4. If (3) fails, randomize initial values of chemical species, run long simulation, then look at steady state.
2. JacobianCluster has visualization for clusters that show max mean-normalized distance from cluster mean for each timepoint
3. BioModels path is a constant
4. Create JacobianCluster. Knows Jacobians and their timepoints.
