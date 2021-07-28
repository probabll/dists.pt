from itertools import chain

from .bernoulli import bernoulli_probs_from_logit, bernoulli_log_probs_from_logit
from .concrete import BinaryConcrete, RelaxedOneHotCategoricalStraightThrough
from .d01c01 import D01C01, MixtureD01C01, Rectified01
from .exponential import RightTruncatedExponential
from .kumaraswamy import Kumaraswamy
from .mixture import MixtureSameFamily, MixtureOfGaussians
from .product import ProductOfDistributions
from .stretched import Stretched
from .truncated import Truncated01
from .dirichlet import MaskedDirichlet
from .bitvector import NonEmptyBitVector, MaxEntropyFaces
from .deterministic import Delta
from .mixed import MixedDirichlet


import torch.distributions as torchd


def get_named_params(p):
    """Return a sequence of pairs (param_name: str, param_value: tensor) based on the type of the distribution p"""
    if isinstance(p, torchd.Normal):
        return ('loc', p.loc), ('scale', p.scale)
    elif isinstance(p, torchd.Beta):
        return ('a', p.concentration1), ('b', p.concentration0)
    elif isinstance(p, BinaryConcrete):
        return ('temp', p.temperature), ('logits', p.logits)
    elif isinstance(p, RightTruncatedExponential):
        return ('rate', p.base.rate), ('upper', p.upper), ('normaliser', p.normaliser)
    elif isinstance(p, Kumaraswamy):
        return ('a', p.a), ('b', p.b)
    elif isinstance(p, torchd.Independent):
        return get_named_params(p.base_dist)
    elif isinstance(p, MixtureSameFamily):
        return (('log_w', p.log_weights),) + get_named_params(p.components)
    elif isinstance(p, ProductOfDistributions):
        return tuple(chain(*(get_named_params(c) for c in p.distributions)))
    else:
        return tuple()


__all__ = [
    "bernoulli_log_probs_from_logit", 
    "bernoulli_probs_from_logit",
    "BinaryConcrete",
    "D01C01",
    "Delta",
    "MixtureD01C01",
    "Rectified01",
    "RightTruncatedExponential",
    "Kumaraswamy",
    "MaxEntropyFaces",
    "MixtureSameFamily",
    "MixtureOfGaussians", 
    "ProductOfDistributions",
    "RelaxedOneHotCategoricalStraightThrough",
    "Stretched",
    "Truncated01",
    "get_named_params",
    "MaskedDirichlet",
    "NonEmptyBitVector",
    "MixedDirichlet"
]


