"""
Distributions of the kind:

1.    \alpha_0 * \delta(x) + (1 - \alpha_0) * f(x) [0 < x < 1]
    
2.    \alpha_0 * \delta(x) + \alpha_1 * \delta(1-x) + (1 - \alpha_0 - \alpha_1) * f(x) [0 < x < 1]
   
where \alpha_i \in (0, 1)
and f(x) is a properly normalised density in the open (0, 1)
"""

import torch
from torch.distributions import Distribution
from torch.distributions.uniform import Uniform
from torch.distributions.utils import broadcast_all
from torch.distributions.kl import register_kl, kl_divergence
import torch.nn.functional as F
from torch.distributions.utils import probs_to_logits, logits_to_probs

from .bernoulli import bernoulli_probs_from_logit, bernoulli_log_probs_from_logit
from .stretched import Stretched
from .truncated import Truncated01

EPS = 1e-4

class D01C01:
    """
    Properties:
    - p0
    - p1
    - pc
    - log_p0
    - log_p1
    - log_pc
    - cont
    """
        
    def entropy(self):
        h = - self.p0 * self.log_p0 - self.p1 * self.log_p1 - self.pc * self.log_pc + self.pc * self.cont.entropy()
        return h

    def kl(p, q, n_samples=1, exact_entropy=False):
        """
        """
        if not (isinstance(p, D01C01) or isinstance(q, D01C01)):
            raise ValueError("p and q must mix delta(x), delta(1-x), and a density over (0, 1)")

        kl0 = p.p0 * (p.log_p0 - q.log_p0) 
        kl1 = p.p1 * (p.log_p1 - q.log_p1)
        klc = p.pc * (p.log_pc - q.log_pc)
        # Here we estimate the last term by sampling in the continuous support (0, 1)
        x = p.cont.rsample(sample_shape=torch.Size([n_samples]))
        if exact_entropy:
            H = p.cont.entropy()
        else:            
            H = - p.cont.log_prob(x).mean(0)
        C = - q.cont.log_prob(x).mean(0)
        klf = p.pc * (-H + C)
        return kl0 + kl1 + klc + klf

    
@register_kl(D01C01, D01C01)
def _kl(p, q):
    return p.kl(q, n_samples=1, exact_entropy=False)


class MixtureD01C01(D01C01,Distribution):

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
        log_prob = torch.where((value < 0.) + (value > 1.), torch.full_like(log_prob, float('-inf')), log_prob)
        return log_prob

    def cdf(self, value):
        cdf = self.p0 + self.pc * self.cont.cdf(value)
        cdf = torch.where(value == 0., self.p0, cdf)
        cdf = torch.where(value == 1., self.p0 + self.pc + self.p1, cdf)
        cdf = torch.where(value < 0., torch.zeros_like(cdf), cdf)
        cdf = torch.where(value > 1., torch.ones_like(cdf), cdf)
        return cdf
    
    
class Rectified01(D01C01,Distribution):
    
    def __init__(self, base: Distribution, validate_args=None):
        """
        Truncate a base distribution to the support (0, 1), 
            for this to work the base must have a support wider than (0, 1)
            and it must have a closed-form cdf (necessary for normalisation) 
            and inverse cdf (necessary for sampling).
            
        """
        super(Rectified01, self).__init__(
            base.batch_shape, 
            base.event_shape, 
            validate_args=validate_args)
        
        self.base = base
        # this is used to get the shape and device of the base
        x = self.base.sample()
        # How the mass is partitioned 
        self.p0 = torch.clamp(self.base.cdf(torch.zeros_like(x)), min=EPS, max=1-EPS)
        self.pc = torch.clamp(self.base.cdf(torch.ones_like(x)) - self.p0, min=EPS, max=1-EPS)
        self.p1 = torch.clamp(1 - self.p0 - self.pc, min=EPS, max=1-EPS)
        # Log probs
        self.log_p0 = torch.log(self.p0 + EPS)
        self.log_p1 = torch.log(self.p1 + EPS)
        self.log_pc = torch.log(self.pc + EPS)
        
        self.cont = Truncated01(base)
        
        self._uniform = Uniform(torch.zeros_like(x) + EPS, torch.ones_like(x) - EPS)
    
    def log_prob(self, value):
        # x \in (0, 1)
        log_p = self.log_pc + self.cont.log_prob(value)
        # x = 0
        log_p = torch.where(value == 0., self.log_p0, log_p)
        # x = 1
        log_p = torch.where(value == 1., self.log_p1, log_p)
        # everything else
        log_p = torch.where((value < 0.) + (value > 1.), torch.full_like(log_p, float('-inf')), log_p)
        
        return log_p
    
    def cdf(self, value):
        """
        Note that HardKuma.cdf(0) = HardKuma.pdf(0) by definition of HardKuma.pdf(0),
         also note that HardKuma.cdf(1) = 1 by definition because
         the support of HardKuma is the *closed* interval [0, 1]
         and not the open interval (left, right) which is the support of the stretched variable.
        """
        cdf = torch.where(
            value < 1.,  
            self.base.cdf(value), 
            torch.ones_like(value)           # all of the mass
        )
        cdf = torch.where(value < 0., torch.zeros_like(cdf), cdf)
        return cdf 
            
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)
        
    def rsample(self, sample_shape=torch.Size()):        
        x = self.base.rsample(sample_shape)
        x = torch.nn.functional.hardtanh(x, 0., 1.)        
        return x
     
