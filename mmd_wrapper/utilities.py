from typing import Optional, Union

import matplotlib.pyplot as plt
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
    Plots specified 'keys' from dictionary 'hist' over time

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
    'hist['keys']' over 'hist['iter_key']'
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
