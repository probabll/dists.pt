"""
Distributions of the kind:

1.    \alpha_0 * \delta(x) + (1 - \alpha_0) * f(x) [0 < x < 1]
    
2.    \alpha_0 * \delta(x) + \alpha_1 * \delta(1-x) + (1 - \alpha_0 - \alpha_1) * f(x) [0 < x < 1]
   
where \alpha_i \in (0, 1)
and f(x) is a properly normalised density in the open (0, 1)
"""

import torch
from torch.distributions.uniform import Uniform
from torch.distributions.utils import broadcast_all
from torch.distributions.kl import register_kl, kl_divergence
import torch.nn.functional as F
from torch.distributions.utils import probs_to_logits, logits_to_probs

from .bernoulli import bernoulli_probs_from_logit, bernoulli_log_probs_from_logit

EPS = 1e-5


class MixtureD0C01(torch.distributions.Distribution):
    
    def __init__(self, cont, logits0=None, probs0=None, validate_args=None):
        """
        - with probability p_0 = sigmoid(logits0) this returns 0
        - with probability 1 - p_0 this returns a sample in the open interval (0, 1)
        
        logits0: logits for p_0
        cont: a (properly normalised) distribution over (0, 1)
            e.g. RightTruncatedExponential
        """
        if logits0 is None and probs0 is None:
            raise ValueError("You must specify either logits0 or probs0")
        if logits0 is not None and probs0 is not None:
            raise ValueError("You cannot specify both logits0 and probs0")            
        shape = cont.batch_shape
        super(MixtureD0C01, self).__init__(batch_shape=shape, validate_args=validate_args)
        if logits0 is None:
            self.logits = probs_to_logits(probs0, is_binary=True)
        else:
            self.logits = logits0
        self.cont = cont
        self.p0, self.pc = bernoulli_probs_from_logit(self.logits)
        self.log_p0, self.log_pc = bernoulli_log_probs_from_logit(self.logits)        
        self.uniform = Uniform(torch.zeros(shape).to(self.logits.device), 
                               torch.ones(shape).to(self.logits.device))
                
    def rsample(self, sample_shape=torch.Size()): 
        # sample from (0, 1) uniformly
        u = self.uniform.rsample(sample_shape)  
        # affine transform to project from (p_0, 1) to (0, 1)
        # note that only where p_0 < u < 1 this is correct
        to_cont = (u - self.p0) / self.pc    
        # c ~ ContinuousDist()
        # note where p_0 < u < 1, c is valid and is in (0,1)
        c = self.cont.icdf(to_cont)
        # inverse cdf of mixture model
        # 0 if u < p_0
        # c otherwise
        x = torch.where(u <= self.p0, torch.zeros_like(u), c)
        return x
    
    def log_prob(self, value):    
        log_prob_cont = self.cont.log_prob(value)
        log_prob = torch.where(value == 0., self.log_p0, self.log_pc + log_prob_cont)
        return log_prob
    
    def cdf(self, value):
        cdf_cont = self.cont.cdf(value)
        cdf = torch.where(value == 0., self.p0, self.p0 + self.pc * cdf_cont)
        return cdf
    
    def entropy(self):
        h = self.p0 * ( - self.log_p0) + self.pc * (- self.log_pc) + self.pc * self.cont.entropy()
        return h

    def kl_divergence(self, other: 'MixtureD0C01'):
        if not isinstance(other, MixtureD0C01):
            raise ValueError("I can only estimate KL(MixtureD0C01(f, p) || MixtureD0C01(g, q))")
        kl = self.p0 * (self.log_p0 - other.log_p0)
        kl = kl + self.pc * (self.log_pc - other.log_pc)
        kl = kl + self.pc * kl_divergence(self.cont, other.cont)
        return kl

class MixtureD01C01(torch.distributions.Distribution):

    def __init__(self, cont, logits=None, probs=None, validate_args=None):
        """
        cont: a (properly normalised) distribution over (0, 1)
            e.g. RightTruncatedExponential, Uniform(0, 1)
        logits: [..., 3] 
        probs: [..., 3]
        """
        if logits is None and probs is None:
            raise ValueError("You must specify either logits or probs")
        if logits is not None and probs is not None:
            raise ValueError("You cannot specify both logits and probs")                        
        shape = cont.batch_shape
        super(MixtureD01C01, self).__init__(batch_shape=shape, validate_args=validate_args)
        if logits is None:
            self.logits = probs_to_logits(probs, is_binary=False)
            self.probs = probs
        else:
            self.logits = logits
            self.probs = logits_to_probs(logits, is_binary=False)
        
        self.logprobs = F.log_softmax(self.logits, dim=-1)
        self.cont = cont
        self.p0, self.p1, self.pc = [t.squeeze(-1) for t in torch.split(self.probs, 1, dim=-1)]
        self.log_p0, self.log_p1, self.log_pc = [t.squeeze(-1) for t in torch.split(self.logprobs, 1, dim=-1)]
        self.uniform = Uniform(torch.zeros(shape).to(self.logits.device),
                               torch.ones(shape).to(self.logits.device))

    def rsample(self, sample_shape=torch.Size()):
        # sample from (0, 1) uniformly
        u = self.uniform.rsample(sample_shape)
        # affine transform to project from (p_0, 1) to (0, 1)
        # note that only where p_0 < u < 1 this is correct
        # print(u.size(), self.p0.size(), self.pc.size())
        to_cont = (u - self.p0) / self.pc
        # c ~ ContinuousDist()
        # note where p_0 < u < 1, c is valid and is in (0,1)
        x = self.cont.icdf(to_cont)
        # inverse cdf of mixture model
        # 0 if u < p_0
        # c otherwise
        x = torch.where(u <= self.p0, torch.zeros_like(u), x)
        x = torch.where(u >= self.p0 + self.pc, torch.ones_like(u), x)
        return x

    def log_prob(self, value):
        log_prob = self.log_pc + self.cont.log_prob(value)
        log_prob = torch.where(value == 0., self.log_p0, log_prob)
        log_prob = torch.where(value == 1., self.log_p1, log_prob)
        return log_prob

    def cdf(self, value):
        cdf = self.p0 + self.pc * self.cont.cdf(value)
        cdf = torch.where(value == 0., self.p0, cdf)
        cdf = torch.where(value == 1., self.p0 + self.pc + self.p1, cdf)
        return cdf

    def entropy(self):
        h = - self.p0 * self.log_p0 - self.p1 * self.log_p1 - self.pc * self.log_pc + self.pc * self.cont.entropy()
        return h

    def kl_divergence(self, other: 'MixtureD01C01'):
        if not isinstance(other, MixtureD01C01):
            raise ValueError("I can only estimate KL(MixtureD01C01(f, p) || MixtureD01C01(g, q))")
        kl = self.p0 * (self.log_p0 - other.log_p0)     
        kl = kl + self.p1 * (self.log_p1 - other.log_p1)
        kl = kl + self.pc * (self.log_pc - other.log_pc)
        kl = kl + self.pc * kl_divergence(self.cont, other.cont)
        return kl


@register_kl(MixtureD0C01, MixtureD0C01)
def _kl(p, q):
    return p.kl_divergence(q)

@register_kl(MixtureD01C01, MixtureD01C01)
def _kl(p, q):
    return p.kl_divergence(q)
