import torch
from torch.distributions.uniform import Uniform
from torch.distributions.kl import register_kl

EPS = 1e-5

class BinaryConcrete(torch.distributions.relaxed_bernoulli.RelaxedBernoulli):
    
    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        super(BinaryConcrete, self).__init__(temperature, probs=probs, logits=logits, validate_args=validate_args)
        
    def cdf(self, value):
        return torch.sigmoid((torch.log(value + EPS) - torch.log(1. - value + EPS)) * self.temperature - self.logits)
    
    def icdf(self, value):
        return torch.sigmoid((torch.log(value + EPS) - torch.log(1. - value + EPS) + self.logits) / self.temperature)
        
    def rsample_truncated(self, k0, k1, sample_shape=torch.Size()):        
        shape = self._extended_shape(sample_shape)
        probs = torch.distributions.utils.clamp_probs(self.probs.expand(shape))
        uniforms = Uniform(self.cdf(torch.full_like(self.logits, k0)), 
                           self.cdf(torch.full_like(self.logits, k1))).rsample(sample_shape)
        x = (uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()) / self.temperature
        return torch.sigmoid(x)


def kl_concrete_concrete(p, q, n_samples=1):
    """
    KL is estimated for the logits of the concrete distribution to avoid underflow.
    """
    x_logit = p.base_dist.rsample(torch.Size([n_samples]))
    return (p.base_dist.log_prob(x_logit) - q.base_dist.log_prob(x_logit)).mean(0)


@register_kl(BinaryConcrete, BinaryConcrete)
def _kl_concrete_concrete(p, q):
    return kl_concrete_concrete(p, q, n_samples=1)
