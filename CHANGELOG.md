# Upcoming
  ### Ongoing
  - Add more techniques

  ### Features
  - Add startup details to `README.md`
  - Add transformations/projections to output
  - Add `pairs` to correlation class
  - Add return documentation to `mmd_combine.py`
  - Add tests to algorithms
  - Add 3D plotting to examples
  - Add verbosity to MMD-MA
  - Convert MMD-MA to eager execution
  - Add library features such as UnionCom's `Visualize` (?)
  
  ### QOL
  - Add visualizations to library
  - `README.md` badge displaying segmented checks
  - Potentially switch main output method to `yield`
  - Add text file output
  - Test verbosity method for thread-safety
  - Switch verbosity method to stream rather than temp file

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
