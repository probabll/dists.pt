import torch
import torch.nn as nn
from torch.distributions import Distribution
from torch.distributions import register_kl, kl_divergence


class ProductOfDistributions(Distribution):

    def __init__(self, distributions: list, validate_args=None):
        assert all(p.batch_shape == distributions[0].batch_shape for p in distributions), "All distributions must have the same batch_shape"
        assert all(len(p.event_shape) == 1 for p in distributions), "I can only deal with vector-valued events"
        super().__init__(
            batch_shape=distributions[0].batch_shape, 
            event_shape=torch.Size([sum(p.event_shape[0] for p in distributions)]), 
            validate_args=validate_args
        )
        self.sizes = [p.event_shape[0] for p in distributions]
        self.n_dists = len(distributions)
        self.distributions = distributions
        
    def split(self, value):
        """Split value [...,E_1+...+E_K] into a tuple of K tensors [...,E_k]"""
        values = torch.split(value, self.sizes, -1)
        return values

    def rsample(self, sample_shape=torch.Size()):
        # [K, ...]
        return torch.cat([p.rsample(sample_shape) for p in self.distributions], -1)

    def sample(self, sample_shape=torch.Size()):
        return torch.cat([p.sample(sample_shape) for p in self.distributions], -1)

    @property
    def mean(self):
        return torch.cat([p.mean for p in self.distributions], -1)

    def log_prob(self, value):
        values = self.split(value)
        return torch.cat([p.log_prob(values[i]) for i, p in enumerate(self.distributions)], -1).sum(-1)

    def cdf(self, value):
        raise NotImplementedError("You may be able to get the cdf of my parts instead")

    def icdf(self, value):
        raise NotImplementedError("You may be able to get the icdf of my parts instead")

    def entropy(self):
        raise NotImplementedError("You may be able to get the entropy of my parts instead")

    
@register_kl(ProductOfDistributions, ProductOfDistributions)
def _prod_prod(p, q):
    assert p.n_dists == q.n_dists, "I need the same number of distributions"
    return torch.cat([kl_divergence(pi, qi).unsqueeze(-1) for pi, qi in zip(p.distributions, q.distributions)], -1).sum(-1)

