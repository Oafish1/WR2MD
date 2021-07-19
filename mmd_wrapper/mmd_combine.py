import contextlib
import os

from .algs import *


@contextlib.contextmanager
def stuff_output(verbose: int):
    if not verbose:
        with open('_trash.notafile', 'w') as f_null:
            with contextlib.redirect_stdout(f_null):
                yield
        os.remove('_trash.notafile')
    else:
        yield


def mmd_combine(*datasets,
                method: str = '_running',
                verbose: int = 1,
                updated: bool = False,
                **kwargs):
    """
    Performs `method` using `kwargs` on the specified `datasets` to
    provide a combined representation of the multi-modal data

    Parameters
    ----------
    *datasets: np.arrays
        Any number of np.arrays, each meant to represent data from one modality
    method: str
        Specification of which method to use
    verbose: int
        Specification of whether or not to output
    updated: bool
        Whether or not to use the library version of the method, if available.
        This can result in less functionality
    **kwargs: kwargs
        Any parameters to use while running the specified method.  Everything
        other than the parameters above must be handled as a kwarg

    Returns
    -------
    Combined representation of multi-modal input data, generally i
    ([mapped datasets], history, auxiliary) format
    """

    query = method.lower()

    if query == '_running':
        return running.func()

    elif query == 'mmd_ma':
        """
        Returns ([mapped datasets], history)

        `mmd_ma_helper.py`:
        "...History object with 'loss' (, _mmd, _penalty, _distortion),
        'alpha', 'beta', 'iteration' entries"
        """
        assert len(datasets) == 2, 'MMD-MA only accepts two datasets'

        k1_matrix, k2_matrix = datasets
        # asdf add mapping
        return mmd_ma_helper(*datasets, **kwargs)

    elif query == 'unioncom':
        """
        Returns ([mapped datasets], history, UnionCom object)

        `unioncom_helper.py`:
        "History object with various entries returned depending on
        the parameters chosen
        tsne:
            'loss', 'loss_align'
        MultiOmics:
            'corr_alpha'
        All Methods:
            'corr_err', 'corr_iteration', 'iteration'"

        NOTE: stdout suppression NOT tested with multi-threading
        """
        if updated:
            uc = source.unioncom_updated.UnionCom.UnionCom(**kwargs)
        else:
            uc = unioncom_helper(**kwargs)

        with stuff_output(verbose):
            mmd = uc.fit_transform(dataset=[*datasets])

        return (mmd, uc._history, uc)

    elif query == 'maninetcluster':
        """
        Parameters
        ----------
        alignment: str
            Type of alignment to perform, found in `alignment.py`
        **kwargs: kwargs
            Any parameters to use while running the specified method

        Returns
        -------
        (
            [mapped datasets],
            {'pairwise_error': float},
            Class with methods project(X, Y, num_dims=none) and
                               apply_transform(other),
        )

        In the case of 'manifold warping', only returns ([mapped datasets], class)
        """
        assert len(datasets) == 2, 'ManiNetCluster only accepts two datasets'

        return maninetcluster_helper(*datasets, **kwargs)
