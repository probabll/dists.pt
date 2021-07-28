import torch
import torch.distributions as td
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from itertools import product
from collections import Counter
import numpy as np
from probabll.distributions import NonEmptyBitVector, MaxEntropyFaces
from probabll.distributions import MaskedDirichlet


class MixedDirichlet(td.Distribution):
    """
    This class manipulates distributions over bit-vectors with the constraint that the outcome 'all zeros' is not in the sample space.
    """

    has_rsample = True
    has_enumerate_support = False
    arg_constraints = {'concentration': td.constraints.independent(td.constraints.greater_than_eq(0.), 1)}
    support = td.constraints.simplex
    
    def __init__(self, *, concentration, scores=None, pmf_n=None, validate_args=None):
        """
        Parameters

            scores: [B, K] real scores
            concentration: [B, K] strictly concentrations
        """
        if scores is not None:
            if pmf_n is not None:
                raise ValueError("Provide scores or pmf_n, not both.")
            batch_shape, event_shape = scores.shape[:-1], scores.shape[-1:]
        elif pmf_n is not None:
            batch_shape, event_shape = pmf_n.shape[:-1], pmf_n.shape[-1:]
        else:
            raise ValueError("Provide scores or pmf_n")
        self._concentration = concentration
        super().__init__(batch_shape, event_shape, validate_args)
        if scores is None:
            self.faces = MaxEntropyFaces(pmf_n)
        else:
            self.faces = NonEmptyBitVector(scores)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MixedDirichlet, _instance)
        batch_shape = torch.Size(batch_shape)
        new._concentration = self._concentration.expand(batch_shape + self.event_shape)
        new.faces = self.faces.expand(batch_shape)
        super(MixedDirichlet, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def dim(self):
        return self._K 

    @property
    def support_size(self):
        return 2 ** self._K - 1

    @property
    def scores(self):
        return self._scores
    
    @property
    def concentration(self):
        return self._concentration
    
    def sample(self, sample_shape=torch.Size()):
        if sample_shape is not None:
            sample_shape = torch.Size(sample_shape)

        f = self.faces.sample(sample_shape)
        Y = self.Y(f)
        return Y.sample()
    
    def rsample(self, sample_shape=torch.Size()):
        if sample_shape is not None:
            sample_shape = torch.Size(sample_shape)
        f = self.faces.sample(sample_shape)
        Y = self.Y(f)
        return Y.rsample()
    
    def log_prob(self, value):
        y = value
        f = (value > 0)
        log_prob_f = self.faces.log_prob(f)
        log_prob_y = self.Y(f).log_prob(y)
        return log_prob_f + log_prob_y

    def Y(self, f):
        return MaskedDirichlet(f.bool(), self.concentration.expand(f.shape))

    def cross_entropy(self, other):
        """
        Compute H(p, q) = - \sum_x p(x) \log q(x)
         where p = self, and q = other.

        :return: [B]        
        """
        f = self.faces.enumerate_support()
        w = self.faces.log_prob(f).exp()
        try:
            f_part = self.faces.cross_entropy(other.faces)
        except: 
            f_part = -(w * other.faces.log_prob(f)).sum(0)
        return f_part + (w * self.Y(f).cross_entropy(other.Y(f))).sum(0)
    
    def entropy(self):
        return self.cross_entropy(self)
    

@td.register_kl(MixedDirichlet, MixedDirichlet)
def _kl_mixeddir_mixeddir(p, q):
    return p.cross_entropy(q) - p.entropy()


