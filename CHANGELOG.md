# [1.2.2] - 2021-08-11
  ### Added
  - `pairwise_error` function in `utilities` module
  - Full scMNC motor data run in `example.ipynb`
  - Enhanced error printout in `example.ipynb`

# [1.2.1] - 2021-08-05
  ### Added
  - MMD-MA auto kernel calculation feature
  - scMNC sample data in `example.ipynb`
  

# [1.2.0] - 2021-07-29
  ### Added
  - ManiNetCluster 5 remaining methods
  - Increased optimization for automated correlation/weight creation for ManiNetCluster
  - MAGAN
  - `utilities` module for visualizations

  ### Changed
  - Code and specification standardization methods for MMD-MA
  - Added new technique to `example.ipynb`

  ### Fixed
  - ManiNetCluster args for linear methods
  - ManiNetCluster returns for linear methods

# [1.1.0] - 2021-07-19
  ### Added
  - ManiNetCluster method
    - Handler has autofill corrplot and weight functionality
    - Handles 7/12 alignments for now
  - UnionCom method
    - Slightly modified for stat tracking
  - `updated` option in wrapper to use the currently installed library version of an algorithm, if available
  - Verbosity adjustment in `mmd_combine.py`
  - Direct mapping output for MMD-MA
  - Documentation on main wrapper methods
  - More secure/reliable module pulling
  - Comparison of algorithms to `example.ipynb`

  ### Changed
  - Output parameters have been standardized further
  - `example.ipynb`
    - Implements multiple techniques
    - Preview of alignment using PCA

# [1.0.0] - 2021-07-12
  ### Added
  - `mmd_combine.py` wrapper
    - Currently only implements MMD-MA
  - MMD-MA
    - `mmd_ma_helper.py` module.  This implements the parts of MMD-MA not in functions already
  - Wrapper test module `running`
  - Jupyter notebook showcasing a sample run of MMD-MA
  - `build-test.yml` workflow
