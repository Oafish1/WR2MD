from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as sd
from sklearn.decomposition import PCA


def plot_training(hist,
                  keys,
                  alg_name: Optional[str],
                  iter_key: str = 'iteration',
                  log_plot: bool = False
                  ):
    """
    Plots specified 'keys' from dictionary 'hist' over time

    Parameters
    ----------
    hist: Dictionary
        Dictionary containing arrays under the keys in 'keys' as
        well as an equal-length array in 'iter_key' within 'hist'
    keys: Iterable
        List of keys to plot in 'hist'
    alg_name: str
        String containing the name of the algorithm.  To be used
        in the title of the plot
    iter_key: str
        Key to use for the x-axis

    Plots
    -------
    'hist['keys']' over 'hist['iter_key']'
    """
    for k in keys:
        plt.plot(hist[iter_key], hist[k], label=k)

    plt.xlabel('Iteration')
    plt.ylabel('Error Function')
    if log_plot:
        plt.yscale('log')

    title = 'Error over Time'
    if alg_name is not None:
        title = alg_name + ' ' + title
    plt.title(title)

    plt.legend(loc='upper right')


def alignment_visualize(*mapping,
                        alg_name: Optional[str],
                        legend_loc: Union[str, int] = 'best'):
    """
    Plots mappings provided in a PCA visualization

    Parameters
    ----------
    *mapping: Array
        Contains np.array types.  Any number of mappings to be
        used in a PCA visualization.  The first mapping sets the
        PCA target
    alg_name: str
        String containing the name of the algorithm.  To be used
        in the title of the plot
    legend_loc: str
        Location of the legend

    Plots
    -------
    PCA visualization of each point in the provided mappings
    """
    pca = PCA(n_components=2)
    pca.fit(mapping[0])

    for i, m in enumerate(mapping):
        m_pca = pca.transform(m)
        if i == 0:
            label = 'Native'
        else:
            label = 'Mapped ' + str(i)
        plt.scatter(m_pca[:, 0], m_pca[:, 1], label=label)

    title = 'PCA Plot'
    if alg_name is not None:
        title = alg_name + ' ' + title
    plt.title(title)

    plt.legend(loc=legend_loc)


def _default_normalize(*maps):
    map_max = max(map.max() for map in maps)
    return (map/map_max for map in maps)


def pairwise_error(map1,
                   map2,
                   metric=sd.euclidean,
                   normalize_method=_default_normalize,
                   normalize_by_feature=False):
    """
    Calculate pairwise error between two maps of equal dimension

    Parameters
    ----------
    map1, map2: array
        Maps to compare
    metric: function
        Function array, array -> float that calculates the
        distance/difference between two (potentially) high
        dimensional points
    normalize_method: function
        Normalization method for use on the maps
    normalize_by_feature: bool
        If true, divides the final result by the number of features

    Returns
    -------
    Cumulative pairwise error
    """
    diff = _pairwise(map1,
                     map2,
                     metric=metric,
                     normalize_method=normalize_method,
                     normalize_by_feature=normalize_by_feature)
    return diff.sum()


def pairwise_boxplot(*maps,
                     labels=None,
                     metric=sd.euclidean,
                     normalize_method=_default_normalize,
                     normalize_by_feature=False):
    """
    Calculate pairwise error between two maps of equal dimension

    Parameters
    ----------
    maps: arrays
        Maps to compare.  Will be evaulated in pairs starting from
        the first two
    labels: array
        Strings that contain the names of each map pair to compute.
        Will be used for text display only
    metric: function
        Function array, array -> float that calculates the
        distance/difference between two (potentially) high
        dimensional points
    normalize_method: function
        Normalization method for use on the maps
    normalize_by_feature: bool
        If true, divides the final result by the number of features

    Plots
    -------
    'hist['keys']' over 'hist['iter_key']'
    """
    assert len(maps) % 2 == 0, 'The number of maps provided must be even'

    errors = [_pairwise(*maps[i-2:i],
                        metric=metric,
                        normalize_method=normalize_method,
                        normalize_by_feature=normalize_by_feature)
              for i in range(2, len(maps)+1, 2)]

    plt.boxplot(errors)
    plt.title('Pairwise Error')
    ax = plt.gca()
    if labels is not None:
        ax.set_xticklabels(labels)


def _pairwise(map1,
              map2,
              metric=sd.euclidean,
              normalize_method=_default_normalize,
              normalize_by_feature=False):
    """
    Small helper function for computing pairwise error

    Parameters
    ----------
    map1, map2: array
        Maps to compare.  Will be evaulated in pairs starting from
        the first two
    metric: function
        Function array, array -> float that calculates the
        distance/difference between two (potentially) high
        dimensional points
    normalize_method: function
        Normalization method for use on the maps
    normalize_by_feature: bool
        If true, divides the final result by the number of features

    Returns
    -------
    Pairwise error for each point
    """
    map1, map2 = normalize_method(map1, map2)
    diff = np.array([metric(row1, row2) for row1, row2 in zip(map1, map2)])
    if normalize_by_feature:
        diff = diff/map1.shape[1]
    return diff
