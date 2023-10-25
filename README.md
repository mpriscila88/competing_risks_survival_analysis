# competing_risks_survival_analysis

# Code for the original publication "Brain Health Scores to Predict Neurological Outcomes from Electronic Health Records"
https://doi.org/10.1016/j.ijmedinf.2023.105270

# This repository contains the folders:

# Python

- Preprocessing: code for generation of data for train and test, including survival time and binary event variables, and all covariates used in the analysis. MICE imputation code is also included.
  For the preprocessing we provide the code only, which is replicable and reproducible with the investigator's cohort data.

- Postpreprocessing: code for generation of data for survival analysis with competing risks (main folder) and code for generating the statistics resulting from the modeling performed in R (stats folder).

- Plots: code for generation of plots in manuscript and appendix.

# Rstudio
- Survival analysis code for each of the outcomes of interest. We include the deidentified data for train and test, where a dummy column "MRN" is given as the matrix index for coding purposes only.

