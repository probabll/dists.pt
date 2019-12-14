import torch
from torch.distributions.uniform import Uniform
from torch.distributions.exponential import Exponential
from torch.distributions.utils import broadcast_all
from torch.distributions.kl import register_kl


EPS = 1e-5


class RightTruncatedExponential(torch.distributions.Distribution):
    
    def __init__(self, rate, upper):
        self.base = Exponential(rate)
        self._batch_shape = self.base.rate.size()
        self._upper = upper
        self.upper = torch.full_like(self.base.rate, upper)        
        # normaliser
        self.normaliser = self.base.cdf(self.upper)
        self.uniform = Uniform(torch.zeros_like(self.upper), self.normaliser)
    
    def rsample(self, sample_shape=torch.Size()):        
        # sample from truncated support (0, normaliser)
        # where normaliser = base.cdf(upper)
        u = self.uniform.rsample(sample_shape)
        x = self.base.icdf(u)
        return x
    
    def log_prob(self, value):
        return self.base.log_prob(value) - torch.log(self.normaliser)
    
    def cdf(self, value):
        return self.base.cdf(value) / self.normaliser
    
    def icdf(self, value):
        return self.base.icdf(value * self.normaliser)
    
    def cross_entropy(self, other):
        assert isinstance(other, RightTruncatedExponential)
        assert type(self.base) is type(other.base) and self._upper == other._upper 
        a = torch.log(other.base.rate) - torch.log(other.normaliser)
        log_b = torch.log(self.base.rate) + torch.log(other.base.rate) - torch.log(self.normaliser)
        b = torch.exp(log_b)
        c = (torch.exp(-self.base.rate) * (- self.base.rate - 1) + 1) / (self.base.rate ** 2)
        return -(a - b * c)

    def entropy(self):
        return self.cross_entropy(self)    
        
def kl_righttruncexp_righttruncexp(p, q):
    assert type(p.base) is type(q.base) and p._upper == q._upper
    h = - p.entropy() 
    c = p.cross_entropy(q)
    return h + c

@register_kl(RightTruncatedExponential, RightTruncatedExponential)
def _kl(p, q):
    return kl_righttruncexp_righttruncexp(p, q)

@register_kl(RightTruncatedExponential, Uniform)
def _kl(p, q):
    # E_p[log U] is a constant
    #  the constant corresponds to 1 if U is U(0,1)
    #  to be sure we compute it over the (0, 1) interval in terms of the Uniform's cdf
    constant = - torch.log(q.cdf(1) - q.cdf(0))
    return - p.entropy() - constant

