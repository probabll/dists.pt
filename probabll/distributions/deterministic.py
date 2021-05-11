# This is adapted from Pyro

# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import numbers

import torch
import torch.distributions as td
from torch.distributions import Distribution
from torch.distributions import constraints


def sum_rightmost(value, dim):
    """
    Sum out ``dim`` many rightmost dimensions of a given tensor.
    If ``dim`` is 0, no dimensions are summed out.
    If ``dim`` is ``float('inf')``, then all dimensions are summed out.
    If ``dim`` is 1, the rightmost 1 dimension is summed out.
    If ``dim`` is 2, the rightmost two dimensions are summed out.
    If ``dim`` is -1, all but the leftmost 1 dimension is summed out.
    If ``dim`` is -2, all but the leftmost 2 dimensions are summed out.
    etc.
    :param torch.Tensor value: A tensor of ``.dim()`` at least ``dim``.
    :param int dim: The number of rightmost dims to sum out.
    """
    if isinstance(value, numbers.Number):
        return value
    if dim < 0:
        dim += value.dim()
    if dim == 0:
        return value
    if dim >= value.dim():
        return value.sum()
    return value.reshape(value.shape[:-dim] + (-1,)).sum(-1)




class Delta(Distribution):
    """
    Degenerate discrete distribution (a single point).

    Discrete distribution that assigns probability one to the single element in
    its support. Delta distribution parameterized by a random choice should not
    be used with MCMC based inference, as doing so produces incorrect results.

    :param torch.Tensor v: The single support element.
    :param torch.Tensor log_density: An optional density for this Delta. This
        is useful to keep the class of :class:`Delta` distributions closed
        under differentiable transformation.
    :param int event_dim: Optional event dimension, defaults to zero.
    """
    has_rsample = True
    arg_constraints = {'v': constraints.dependent,
                       'log_density': constraints.real}

    def __init__(self, v, log_density=0.0, event_dim=0, validate_args=None):
        if event_dim > v.dim():
            raise ValueError('Expected event_dim <= v.dim(), actual {} vs {}'.format(event_dim, v.dim()))
        batch_dim = v.dim() - event_dim
        batch_shape = v.shape[:batch_dim]
        event_shape = v.shape[batch_dim:]
        if isinstance(log_density, numbers.Number):
            log_density = torch.full(batch_shape, log_density, dtype=v.dtype, device=v.device)
        elif validate_args and log_density.shape != batch_shape:
            raise ValueError('Expected log_density.shape = {}, actual {}'.format(
                log_density.shape, batch_shape))
        self.v = v
        self.log_density = log_density
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return constraints.independent(constraints.real, len(self.event_shape))

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Delta, _instance)
        batch_shape = torch.Size(batch_shape)
        new.v = self.v.expand(batch_shape + self.event_shape)
        new.log_density = self.log_density.expand(batch_shape)
        super(Delta, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.v.shape
        return self.v.expand(shape)

    def log_prob(self, x):
        v = self.v.expand(self.batch_shape + self.event_shape)
        log_prob = (x == v).type(x.dtype).log()
        log_prob = sum_rightmost(log_prob, len(self.event_shape))
        return log_prob + self.log_density

    @property
    def mean(self):
        return self.v

    @property
    def variance(self):
        return torch.zeros_like(self.v)


@td.register_kl(Delta, Delta)
def _kl_delta_delta(p, q):
    if (p.v == q.v).all():
        return torch.zeros(p.batch_shape, device=p.v.device)
    else:
        raise ValueError("The locations of p and q differ")
    
