import numpy as np

from .source.maninetcluster import *
from .source.maninetcluster.correspondence import Correspondence
from .source.maninetcluster.distance import SquaredL2
from .source.maninetcluster.neighborhood import neighbor_graph
from .source.maninetcluster.util import pairwise_error


def maninetcluster_helper(*args,
                          alignment: str = 'no alignment',
                          **kwargs):
    assert len(args) == 2, 'ManiNetCluster takes two dataset arguments: X and Y'

    # TODO: Case checking, save cycles
    if 'corr' not in kwargs:
        kwargs['corr'] = Correspondence(matrix=np.eye(len(args[0])))
    elif type(kwargs['corr']).__module__ == np.__name__:
        kwargs['corr'] = Correspondence(matrix=np.asmatrix(kwargs['corr']))
    if 'Wx' not in kwargs and 'Wy' not in kwargs:
        kwargs['Wx'] = neighbor_graph(args[0], k=5)
        kwargs['Wy'] = neighbor_graph(args[1], k=5)

    # NOTE: **kwargs needs to include num_dims for use in project()
    query = alignment.lower()
    if query == 'no alignment':
        aln = TrivialAlignment(*args, **kwargs)
    elif query == 'affine':
        aln = Affine(*args, **kwargs)
    elif query == 'procrustes':
        aln = Procrustes(*args, **kwargs)
    elif query == 'cca':
        aln = CCA(*args, **kwargs)
    elif query == 'cca_v2':
        aln = CCAv2(*args, **kwargs)
    elif query == 'cca_v3':
        # Added but not in R implementation
        aln = CCAv3(*args, **kwargs)
    elif query == 'linear manifold':
        aln = ManifoldLinear(*args, **kwargs)
    elif query == 'nonlinear manifold aln':
        proj = manifold_nonlinear(*args, **kwargs)
        return (proj, {'pairwise_error': pairwise_error(*proj, metric=SquaredL2)})
    else:
        raise Exception('Invalid alignment: ' + str(alignment))
    """
    elif query == 'ctw':
        aln = asdf(*args, **kwargs)
    elif query == 'manifold warping':
        aln = asdf(*args, **kwargs)
    elif query == 'nonlinear manifold warp':
        pass
    elif query == 'ctw':
        pass
    elif query == 'manifold warping two-step':
        pass
    """
    return (proj, {'pairwise_error': pairwise_error(*proj, metric=SquaredL2)}, aln)
