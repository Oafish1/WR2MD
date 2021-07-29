# Wrapper for Reconciling Multi-Modal Data

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/Oafish1/WR2MD/Build%20and%20Test?label=tests&style=plastic)

This wrapper facilitates the use of the following techniques for reconciling multi-modal data:

- MAGAN
- ManiNetCluster
- MMD-MA
- UnionCom


# Notes

- If MAGAN is interrupted, `tf.reset_default_graph()` may need to be used to avoid errors.
- MMD-MA currently only works without eager execution, meaning that programs requiring eager execution may require reimporting `tensorflow`.
