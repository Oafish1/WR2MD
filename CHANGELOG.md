# [1.2.5] - 2023-08-18
  ### Fixed
  - Remove reference to `sklearn` package
  - Removed deprecated `np.int` usage in `Unioncom` implementation

# [1.2.4] - 2021-09-05
  ### Added
  - `Relative distance` metric.  For any pair, this metric is the average number of points closer than a point's partner

  ### Changed
  - Requirement formatting.  Now stored in `setup.py` rather than `requirements.in`
  - Additional requirements `dev` and `notebooks`
  - `boxplot` function replaces `pairwise_boxplot` and similarly specific functions

  ### Fixed
  - MAGAN output, formatting


# [1.2.3] - 2021-08-25
  ### Added
  - `pairwise_boxplot` function in `utilities` module

  ### Refactored
  - `pairwise_error` function
  - `_pairwise` helper function


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
