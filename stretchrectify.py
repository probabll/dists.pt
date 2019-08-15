import torch
from torch.distributions import Distribution
from torch.distributions.uniform import Uniform
from torch.distributions.kl import register_kl

EPS = 1e-5
    
    
def stretched_distribution(distribution, lower=-0.1, upper=1.1):
    assert lower < upper
    return torch.distributions.TransformedDistribution(distribution,
        torch.distributions.AffineTransform(loc=lower, scale=upper - lower))
    
    
class StretchedAndRectifiedDistribution(torch.distributions.Distribution):
    """  
    
    # KL
    
    Let p(h) and q(h) be two stretched-and-rectified distributions
     which share the same stretching parameters (same affine), same support, and same rectifier. 
    
    Recall that the hard variable H is obtained by transforming a base variable K via an affine transformation with 
     positive slope, this produces the stretched variable T, and then rectifying the result. 
    
    I will use f(t) for the (stretched) density used to specify p(h)
     and g(t) for the density used to specify q(h).
    
    Then
        KL(p||q) = P(0)\log (P(0)/Q(0)) + P(1)\log (P(1)/Q(1)) + \int_0^1 f_T(t) \log (f(t)/g(t)) \dd t
        
    Note that the integral is summing over a *truncated* support, namely, (0, 1) rather than the 
     complete stretched support (lower, upper). Also note that within this support f_T is not properly normalised, 
     though we could write
     
     KL(p||q) = P(t<0)\log (P(0)/Q(0)) + P(t>1)\log (P(1)/Q(1)) + P(0<t<1) \int_0^1 f_T(t)/P(0<t<1) \log (f(t)/g(t)) \dd t
     
    
    Strategies:
    
    We can compute the first two terms exactly, but we need to approximate the third.
    
    1. In one case we approximate it via importance sampling by sampling from a properly normalised distribution
    over (0, 1). This proposal can be for example the base density b(k) used to specify p.
    
    2. Another idea is to sample from f(t) constrained to the truncated support (0, 1). 
    This second idea requires being able to sample from the truncated support.
    
    3. A last idea is to simply estimate on a sample from the base distribution.
    
    Option (2) is recommended.
    """ 
    
    def __init__(self, distribution, lower=-0.1, upper=1.1):
        super(StretchedAndRectifiedDistribution, self).__init__()
        self.base = distribution
        self.stretched = stretched_distribution(self.base, lower=lower, upper=upper)                
        self.loc = lower
        self.scale = upper - lower
        
    def log_prob(self, value):
        # x ~ p(x), and y = hardtanh(x) -> y ~ q(y)
        
        # log_q(x==0) = cdf_p(0) and log_q(x) = log_q(x)
        zeros = torch.zeros_like(value)
        log_p = torch.where(value == zeros,
                            #torch.log(self.base.cdf((value - self.loc) / self.scale)),
                            torch.log(torch.clamp(self.stretched.cdf(zeros), min=EPS)),
                            self.stretched.log_prob(value))
        
        # log_q(x==1) = 1 - cdf_p(1)
        ones = torch.ones_like(value)
        log_p = torch.where(value == ones,
                            torch.log(torch.clamp(1 - self.stretched.cdf(ones), min=EPS)),
                            log_p)
        
        return log_p
    
    def cdf(self, value):
        """
        Note that HardKuma.cdf(0) = HardKuma.pdf(0) by definition of HardKuma.pdf(0),
         also note that HardKuma.cdf(1) = 1 by definition because
         the support of HardKuma is the *closed* interval [0, 1]
         and not the open interval (left, right) which is the support of the stretched variable.
        """
        cdf = torch.where(
            value < torch.ones_like(value),  
            #self.base.cdf((value - self.loc) / self.scale),            
            self.stretched.cdf(value),       # self.base.cdf((value - self.loc) / self.scale),            
            torch.ones_like(value)           # all of the mass
        )
        return cdf
            
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)
        
    def rsample(self, sample_shape=torch.Size()):
        return torch.nn.functional.hardtanh(self.stretched.rsample(sample_shape), 0., 1.)


def kl_base_sampling(p, q, n_samples=1):
    x = p.rsample(sample_shape=torch.Size([n_samples]))  
    return (p.log_prob(x) - q.log_prob(x)).mean(0)


def kl_importance_sampling(p, q, n_samples=1):    
    assert type(p.base) is type(q.base) and p.loc == q.loc and p.scale == q.scale    
    # wrt (lower, upper)   
    # Here I sample from the continuous curve between (0, 1)
    
    # Importance sampling estimate
    proposal = p.base
    x = proposal.rsample(sample_shape=torch.Size([n_samples]))
    klc = (torch.exp(p.log_prob(x) - proposal.log_prob(x)) * (p.log_prob(x) - q.log_prob(x))).mean(0)
    
    # wrt lower
    zeros = torch.zeros_like(klc)
    log_p0 = p.log_prob(zeros)
    p0 = torch.exp(log_p0)
    log_q0 = q.log_prob(zeros)
    kl0 = p0 * (log_p0 - log_q0)
    
    # wrt upper
    ones = torch.ones_like(klc)
    log_p1 = p.log_prob(ones)
    p1 = torch.exp(log_p1)
    log_q1 = q.log_prob(ones)
    kl1 = p1 * (log_p1 - log_q1)
    
    return kl0 + kl1 + klc


def kl_truncated_sampling(p, q, n_samples=1):
    assert type(p.base) is type(q.base) and p.loc == q.loc and p.scale == q.scale    
           
    # Here I sample from the continuous curve between (0, 1)
    # TODO: use the affine and its inverse from p.stretched
    # Base(a,b) variable that corresponds to 0 drawn from StretchedBase(a,b)
    k0 = (0. - p.loc) / p.scale
    # Base(a,b) variable that corresponds to 1 drawn from StretchedBase(a,b)
    k1 = (1. - p.loc) / p.scale
    # x ~ Base(a,b) where x \in (k0, k1)
    x = p.base.rsample_truncated(k0, k1, sample_shape=torch.Size([n_samples]))
    # x ~ Stretched(loc, scale) though constrained to (0, 1)
    x = p.loc + x * p.scale 
    # wrt (lower, upper)       
    log_ratio_cont = (p.stretched.log_prob(x) - q.stretched.log_prob(x)).mean(0)
    
    zeros = torch.zeros_like(log_ratio_cont)
    ones = torch.ones_like(log_ratio_cont)
    
    log_p0 = p.log_prob(zeros)    
    p0 = p.stretched.cdf(zeros)  # torch.exp(log_p0)
    log_p1 = p.log_prob(ones)    
    p1 = ones - p.stretched.cdf(ones)  # torch.exp(log_p1)
    #p_cont = ones - p0 - p1
    p_cont = p.stretched.cdf(ones) - p.stretched.cdf(zeros)
    
    log_q0 = q.log_prob(zeros) 
    log_q1 = q.log_prob(ones)

    # wrt lower    
    log_ratio_0 = log_p0 - log_q0
    
    # wrt upper    
    log_ratio_1 = log_p1 - log_q1
    
    kl_0 = torch.where(p0 > 0., p0 * log_ratio_0, p0)
    kl_1 = torch.where(p1 > 0., p1 * log_ratio_1, p1)
    kl_c = torch.where(p_cont > 0., p_cont * log_ratio_cont, p_cont)
    
    return kl_0 + kl_1 + kl_c


@register_kl(StretchedAndRectifiedDistribution, StretchedAndRectifiedDistribution)
def _kl_stretchredandrectified_stretchredandrectified(p, q):    
    return kl_truncated_sampling(p, q, n_samples=1)
