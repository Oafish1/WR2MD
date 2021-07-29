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

    corr_exclude = ['cca_v2', 'cca_v3', 'ctw', 'ctw two-step', 'manifold warping',
                    'manifold warping two-step', 'no alignment', 'nonlinear manifold warp']
    W_exclude = ['affine', 'cca', 'cca_v2', 'cca_v3', 'ctw', 'ctw two-step', 'no alignment',
                 'procrustes']

    query = alignment.lower()

    if query not in corr_exclude:
        if 'corr' not in kwargs:
            kwargs['corr'] = Correspondence(matrix=np.eye(len(args[0])))
        elif type(kwargs['corr']).__module__ == np.__name__:
            kwargs['corr'] = Correspondence(matrix=np.asmatrix(kwargs['corr']))

    if query not in W_exclude:
        if 'Wx' not in kwargs and 'Wy' not in kwargs:
            kwargs['Wx'] = neighbor_graph(args[0], k=5)
            kwargs['Wy'] = neighbor_graph(args[1], k=5)

    # Standard alignments
    match = True
    # NOTE: **kwargs needs to include num_dims for use in project()
    if query == 'no alignment':
        aln = TrivialAlignment(*args, **kwargs)
    elif query == 'affine':
        # NOTE: X, Y must have same number of columns
        aln = Affine(*args, **kwargs)
    elif query == 'procrustes':
        # NOTE: X, Y must have same number of columns
        aln = Procrustes(*args, **kwargs)
    elif query == 'cca':
        aln = CCA(*args, **kwargs)
    elif query == 'cca_v2':
        aln = CCAv2(*args, **kwargs)
    elif query == 'cca_v3':
        # Added but not in R implementation
        # NOTE: Can fail to init
        aln = CCAv3(*args, **kwargs)
    elif query == 'linear manifold':
        aln = ManifoldLinear(*args, **kwargs)
    else:
        match = False

    if match:
        proj = aln.project(*args, kwargs['num_dims'])
        return (proj, {'pairwise_error': pairwise_error(*proj, metric=SquaredL2)}, aln)

    # Rolled return warps
    match = True
    if query == 'ctw':
        proj = ctw(*args, **kwargs)
    elif query == 'manifold warping':
        proj = manifold_warping_linear(*args, **kwargs)
    else:
        match = False

    if match:
        corr, aln = proj
        proj = aln.project(*args, kwargs['num_dims'])
        return (proj,
                {'pairwise_error': pairwise_error(*proj, metric=SquaredL2),
                 'corr': corr
                 },
                aln)

    # Standalone
    if query == 'nonlinear manifold aln':
        proj = manifold_nonlinear(*args, **kwargs)
        return (proj, {'pairwise_error': pairwise_error(*proj, metric=SquaredL2)})

    # Unrolled return warps
    match = True
    if query == 'ctw two-step':
        proj = ctw_twostep(*args, **kwargs)
    elif query == 'manifold warping two-step':
        proj = manifold_warping_twostep(*args, **kwargs)
    elif query == 'nonlinear manifold warp':
        proj = manifold_warping_nonlinear(*args, **kwargs)
    else:
        match = False

    if match:
        return (proj[1:],
                {'pairwise_error': pairwise_error(*proj[1:], metric=SquaredL2),
                 'corr': proj[0]
                 }
                )

    raise Exception('Invalid alignment: ' + str(alignment))
