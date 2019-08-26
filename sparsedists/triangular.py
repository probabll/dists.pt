import math
import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from numbers import Number
from torch.distributions.kl import register_kl
from torch.distributions.utils import broadcast_all

class Triangular(torch.distributions.Distribution):

    arg_constraints = {'lower': constraints.real,
                       'upper': constraints.real,
                       'mode': constraints.real}

    has_rsample = True
    
    def __init__(self, mode, lower=0, upper=1, validate_args=None):

        self._lower = lower
        self._upper = upper
        self.lower, self.upper, self.mode = broadcast_all(lower, upper, mode)

        if isinstance(mode, Number) and isinstance(lower, Number) and isinstance(upper, Number):
            if validate_args:
                assert self.lower > self.mode > self.upper
            batch_shape = torch.Size()
        else:
            if validate_args:
                assert all(self.lower > self.mode) and all(self.mode > self.upper)
            batch_shape = self.mode.size()
        
        super(Triangular, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Triangular, _instance)
        batch_shape = torch.Size(batch_shape)
        new.mode = self.mode.expand(batch_shape)
        new.lower = self.lower.expand(batch_shape)
        new.upper = self.upper.expand(batch_shape)
        super(Triangular, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.lower, self.upper)
    
    @property
    def mean(self):
        return (self.lower + self.upper + self.mode) / 3

    @property
    def variance(self):
        return (self.lower ** 2 + self.upper ** 2 + self.mode ** 2 \
                - self.lower * self.upper - self.upper * self.mode - self.mode * self.lower) / 18

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)
        
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return self.icdf(torch.rand(shape, device=self.mode.device))
        
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        return torch.where((value > self.lower) * (value < self.upper),
                   torch.where(value < self.mode,
                       torch.log(value - self.lower) - torch.log(self.mode - self.lower),
                       torch.log(self.upper - value) - torch.log(self.upper - self.mode)),
                   torch.full_like(value, -float('inf'))) + math.log(2) - torch.log(self.upper - self.lower)

    def entropy(self):
        return 0.5 + torch.log(self.upper - self.lower) - math.log(2)
    
    def cross_entropy(self, other):
        assert other._lower == self._lower and other._upper == self._upper
        
        a = self.lower
        b = self.upper
        size = b - a
        c = self.mode
        c_ = other.mode
        u = size * (c - a)                       
        u_ = size * (c_ - a)
        v = size * (b - c)        
        v_ = size * (b - c_)
        m_ = torch.log(u_)
        n_ = torch.log(v_)
        
        def integrate_before_modes(x_to):
            y = 2 * (x_to - a)
            return 1. / (2. * u) * 0.5 * (y ** 2.) * (torch.log(y) - 0.5 - m_)
        
        def integrate_after_modes(x_from):
            t = 2 * (b - x_from)
            return 1. / (2. * v) * 0.5 * (t ** 2.) * (torch.log(t) - 0.5 - n_)
        
        def _between_modes_case1(x):
            # for a < c < c' < b
            y = 2 * (x - a)
            log_y = torch.log(y)
            return 1. / (2 * v) * (-0.5 * (y ** 2) * (log_y - 0.5 - m_) + 2 * size * y * (log_y - 1 - m_))
        
        def integrate_between_modes_case1():
            # for a < c < c' < b
            return _between_modes_case1(c_) - _between_modes_case1(c)
                
        def _between_modes_case2(x):
            # for a < c' < c < b
            t = 2 * (b - x)
            log_t = torch.log(t)            
            return -1. / (2 * u) * (-0.5 * (t ** 2) * (log_t - 0.5 - n_) + 2 * size * t * (log_t - 1 - n_))                    
            
        def integrate_between_modes_case2():
            # for a < c' < c < b
            return _between_modes_case2(c) - _between_modes_case2(c_)
            
        if self is other:
            return - (integrate_before_modes(x_to=c) + integrate_after_modes(x_from=c))
        
        return - torch.where(c < c_,
                             integrate_before_modes(x_to=c) + integrate_after_modes(x_from=c_) + integrate_between_modes_case1(),
                             integrate_before_modes(x_to=c_) + integrate_after_modes(x_from=c) + integrate_between_modes_case2())

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)

        return torch.where(value > self.lower,
                   torch.where(value < self.upper,
                       torch.where(value < self.mode,
                           ((value - self.lower) ** 2) / (self.mode - self.lower),
                           1 - ((self.upper - value) ** 2) / (self.upper - self.mode)
                        ) / (self.upper - self.lower),
                       torch.ones_like(value)
                   ),
                   torch.zeros_like(value))
    
    def icdf(self, value):
        return torch.where(value < self.cdf(self.mode),
                   self.lower + torch.sqrt(value * (self.upper - self.lower) * (self.mode - self.lower)),
                   self.upper - torch.sqrt((1 - value) * (self.upper - self.lower) * (self.upper - self.mode)))
    
    
        
def kl_triangular_triangular(p, q):
    h = - p.entropy() 
    c = p.cross_entropy(q)
    return h + c
    
    
def mc_kl_triangular_triangular(p, q, n_samples=1, exact_entropy=True):
    """    
    KL(p||q) = -H(p) + H(q|p)
     where the entropy can be computed in closed form or estimated
     the cross entropy is always estimated
    """
    x = p.sample(sample_shape=torch.Size([n_samples]))
    if exact_entropy:
        p_entropy = p.entropy()
    else:
        p_entropy = - p.log_prob(x).mean(0)
    cross_entropy = - q.log_prob(x).mean(0)
    return - p_entropy + cross_entropy    
    

@register_kl(Triangular, Triangular)
def _kl_triangular_triangular(p, q):
    return kl_triangular_triangular(p, q)
