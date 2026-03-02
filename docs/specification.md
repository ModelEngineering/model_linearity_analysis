# Specification for Model Linearity Analysis

## Architecture

* Class LinearAnalyzer. This class reads and runs models, collecting their Jacobians and constructs a heatmap for the result.
  * Constructed with an SBML or Antimony model and a default specification of the simulate options for start (0), end(10), and number of points (100)
  * Method collectJacobians. Runs the model, collecting a Jacobian at each timepoint. The Jacobian should contain only floating species in the model.
  * Method plot creates a heatmap for the Jacobian matrix where the entries are the coefficient of variation for each Jacobian.
  * Class method processBioModels for the SBML models in /Users/jlheller/home/Technical/repos/temp-biomodels/final
  
## Coding styles

* Functions are camel case.
* Variables are lower case with underline separators
* Document all functions
* Use typing in function signatures.

## Tests

* There is a test for each method of every class

## Repository Structure

* All code is in src
* All tests are in tests
