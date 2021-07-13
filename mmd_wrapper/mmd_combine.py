from .algs import *


def mmd_combine(*datasets, method: str = 'mmd_ma', **kwargs):
    """
    Performs `method` using `kwargs` on the specified `datasets` to
    provide a combined representation of the multi-modal data

    Parameters
    ----------
    *datasets: np.arrays
        Any number of np.arrays, each meant to represent data from one modality
    method: str
        Specification of which method to use
    **kwargs: kwargs
        Any parameters to use while running the specified method

    Returns
    -------
    Combined representation of multi-modal input data
    """

    query = method.lower()

    if query == '_running':
        return running.func()

    if query == 'mmd_ma':
        assert len(datasets) == 2, 'MMD-MA only accepts two datasets'

        k1_matrix, k2_matrix = datasets
        return mmd_ma_helper(k1_matrix, k2_matrix, **kwargs)
