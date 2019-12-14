import torch
from torch.distributions import Distribution, Uniform


EPS = 1e-4


class Truncated01(Distribution):
    """
    Truncate a base distribution to the support (0, 1), 
        for this to work the base must have a support wider than (0, 1)
        and it must have a closed-form cdf (necessary for normalisation) 
        and inverse cdf (necessary for sampling).

    The result is itself a properly normalised density.         
    """
    
    def __init__(self, base: Distribution, validate_args=None):
        super(Truncated01, self).__init__(
            base.batch_shape, 
            base.event_shape, 
            validate_args=validate_args)
        
        self.base = base
        # this is used to get the shape and device of the base
        x = self.base.sample()
        self._base_cdf0 = self.base.cdf(torch.zeros_like(x))
        self._normaliser = self.base.cdf(torch.ones_like(x)) - self._base_cdf0
        self._uniform = Uniform(torch.zeros_like(x) + EPS, torch.ones_like(x) - EPS)
    
    def log_prob(self, value):
        log_p = self.base.log_prob(value) - torch.log(self._normaliser + EPS)
        log_p = torch.where((value < 0) + (value > 1), torch.full_like(log_p, float('-inf')), log_p)       
        return log_p
    
    def cdf(self, value):
        cdf = (self.base.cdf(value) - self._base_cdf0) / (self._normaliser + EPS)
        cdf = torch.where(value < 0, torch.zeros_like(cdf), cdf)  # flat at zero for x < 0
        cdf = torch.where(value > 1, torch.ones_like(cdf), cdf)   # flat at one for x > 1
        return cdf
            
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)
        
    def rsample(self, sample_shape=torch.Size()):        
        # Sample from a uniform distribution
        #  and transform it to the corresponding truncated uniform distribution
        # Let F be the cdf of the base and I its inverse. Let G be the cdf of the truncated distribution and J its inverse. Let U be a uniform random variable over (0, 1).
        # We can sample from the truncated distribution via
        # X = I( F(0) + U * (F(1) - F(0)) )
        # Proof:
        #  Pr{ I( F(0) + U * (F(1) - F(0)) ) <= x }           Let's apply F to both sides of the inequality
        #  = Pr{ F(0) + U*(F(1) - F(0) } <= F(x) }
        #  = Pr{ U <= (F(x) - F(0))/(F(1)-F(0)) }             Note that the right-hand side corresponds to G(x)
        #  = Pr{ U <= G(x) }                                  Now let's apply J to both sides
        #  = Pr{ J(U) <= x }                                  Done!
        u = self._uniform.sample(sample_shape)
        u = self._base_cdf0 + u * self._normaliser 
        # Map to sample space using the base's inverse cdf
        return self.base.icdf(torch.clamp(u, min=EPS, max=1-EPS))
    
    
