# 0.4.1 (2025-05-18)
### Fix
- Tornado dev dependency it logs a warning but continues trying to parse the remainder of the data. This allows remote attackers to generate an extremely high volume of logs, constituting a DoS attack vulnerability, fixed in 6.5.0

# 0.4.0
### Refactor
- These "fit" parameters are now in XNB constructor:
    - bw_function
    - kernel
    - algorithm
    - n_sample
- Enum parameters (bw_function, kernel, algorithm) allow str types as well.
### Style
- HTML representation when display the XNB object in a Jupyter notebook.

# 0.3.1 (2025-03-20)
### Fix
- Hellinger threshold fixed to ensure it always returns features.
- Tornado dev dependency has an HTTP cookie parsing DoS vulnerability, fixed in 6.4.2.

# 0.3.0 (2024-11-20)
### Feat
- New predict_proba method.

### Refactor
- Predict methods refactored to improve performance.

### Fix
- Bug in hash method of _ClassFeatureDistance class.
- Bug in best estimator with rang==0.
- Bug in HSJ method.


# 0.2.3 (2024-10-02)
### Docs
- README updated with repository links.
- Metadata added to pyproject.toml

# 0.2.2 (2024-08-19)
### Docs
- New installation section in README.md
- Example modified in README.md
- New test and coverage badges in README.md

# 0.2.1 (2024-08-18)
### Docs
- README updated.

# 0.2.0 (2024-08-05)
### Refactor
- All methods have been refactored. Ready to be uploaded along with the paper.


# 0.1.3 (2023-02-26)
### Fix
- Error in "_calculate_feature_selection" function fixed. The solution is to copy the map object.
- NotFittedError class coded for "predict" function.
### Privated changes
- Adding a new attribute to the KDE class
- Due to memory problems, we remove the "_kernel_density_dict" attribute and the KDE is recalculated only from the variables needed during the prediction.
- The Pool class is used to obtain parallelization in the KDE calculation.
- The content of the "_calculate_divergence" function was changed to optimize the calculation.
### Style
- Renaming tests in "performance_test.py" file

# 0.1.2 (2023-01-04)
### Refactor
- 'iris.csv' file included in the Git project.

# 0.1.1 (2022-11-27)
### Docs
- CHANGELOG created.
- [build.md](docs/build.md) file created.
### Style
- Internal functions and scripts renamed.
- bw_best_estimator method moved into "_bandwidth_functions.py" file.
### Test
- General "calculate bandwidth" function tested in performance.
- Enhanced bandwidth test Asserts
### Build
- pandarallel dependency removed.
- sympy dependency removed.
- matplotlib dependency removed.
- ipykernel dependency changes to dev-depdenency.
- seaborn dependency removed.
- tqdm dependency removed.
- psutil dependency removed.
- multiprocesspandas dependency removed.
### Fix
- ValueError coded for bandwidth functions.

# 0.1.0 (2022-11-26)
### Feat
- Original code
